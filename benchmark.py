"""
Stage B modeli için kapsamlı değerlendirme — ``app.py``'deki "Model Başarımı" sekmesinin veri kaynağı.

Çalıştırma:
    python benchmark.py                          # varsayılan örneklem (~15 dk, CPU)
    python benchmark.py --n-aiforge 200 --ocr    # daha büyük örneklem + OCR kural motoru

Değerlendirme kümesi (modelin eğitimde görmediği görseller):
  * ``synthetic_test``  — Stage B ``splits.csv`` içindeki sentetik **test** bölümü
    (tahrifat türü, dil ve belge tipi ``metadata.csv``'den gelir).
  * ``synthetic_unseen`` — ``data/4_synthetic_tr_en`` içinde Stage B bölümlerinde hiç
    geçmeyen belge gruplarından örneklem (tahrifat türü dosya adından okunur).
  * ``aiforgedoc_ood``  — AIForge-Doc TestingSet (dağılım dışı; sahte + orijinal).

Çıktılar (``--out``, varsayılan ``results/stageB``):
  * ``per_image.csv``      — görsel başına skorlar, GT istatistikleri, süre, JPEG kalitesi.
  * ``pixel_counts.npz``   — görsel × eşik ızgarası için piksel TP/FP/FN/TN sayıları.
  * ``pixel_samples.npz``  — görsel başına düzgün rastgele piksel örneklemi (ROC/PR için).
  * ``robustness.csv``     — JPEG kalite faktörüne göre yeniden sıkıştırılmış alt kümede sonuçlar.
  * ``meta.json``          — checkpoint, sıcaklık, eşik ızgarası, örneklem boyutları.

Not: Piksel olasılıkları ``calibration.json`` sıcaklığıyla, belge skoru ise **ham logit**
olarak kaydedilir; panel farklı sıcaklıkları sonradan karşılaştırabilir.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from predict import load_calibration, predict_raw
from src.dataset import read_image_rgb, read_mask
from src.models import load_model

logger = logging.getLogger("benchmark")

SYN_ROOT = Path("data/4_synthetic_tr_en")
AIFORGE_TEST = Path("data/finetune_aiforgedoc/TestingSet")
STAGEB_SPLITS = Path("checkpoints/synthetic/splits.csv")
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
PIXEL_SAMPLES_PER_IMAGE = 4000
SYN_NAME = re.compile(r"(syn\d+)__\d+_(.+)\.\w+$")

# libjpeg standart parlaklık niceleme tablosu (Q=50); kalite tahmini için referans
STD_LUMA_Q50 = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage B model başarımı değerlendirmesi")
    p.add_argument("--checkpoint", default="checkpoints/finetuned_multilingual.pth")
    p.add_argument("--mode", choices=("sliding", "resize"), default="sliding")
    p.add_argument("--out", default="results/stageB")
    p.add_argument("--n-synthetic-unseen", type=int, default=40, help="Görülmemiş sentetik görsel sayısı")
    p.add_argument("--n-aiforge", type=int, default=60, help="AIForge-Doc görsel sayısı (yarı sahte, yarı orijinal)")
    p.add_argument("--robust-n", type=int, default=24, help="JPEG dayanıklılık alt kümesi boyutu")
    p.add_argument("--qualities", default="100,90,80,70,60,50", help="JPEG kalite faktörleri")
    p.add_argument("--ocr", action="store_true", help="OCR + kural motorunu da çalıştır (yavaş)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Değerlendirme kümesi
# ---------------------------------------------------------------------------

def build_eval_set(n_unseen: int, n_aiforge: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    meta = pd.read_csv(SYN_ROOT / "metadata.csv")
    meta["name"] = meta["image"].str.split("/").str[-1]
    splits = pd.read_csv(STAGEB_SPLITS)
    syn = splits[splits["source"] == "synthetic"].copy()
    syn["name"] = syn["image_path"].str.split("/").str[-1]
    seen_groups = set(syn["name"].str.extract(SYN_NAME)[0])

    test = syn[syn["split"] == "test"].merge(meta, on="name", how="left")
    for r in test.itertuples():
        rows.append(dict(subset="synthetic_test", image_path=str(SYN_ROOT / "images" / r.name),
                         mask_path=str(SYN_ROOT / "masks" / Path(r.name).with_suffix(".png")),
                         label=int(r.label_x), tamper_type=r.tamper_type, lang=r.lang, doc_type=r.doc_type))

    unseen = []
    for img in sorted((SYN_ROOT / "images").iterdir()):
        m = SYN_NAME.match(img.name)
        if m and m.group(1) not in seen_groups:
            unseen.append((img, m.group(2)))
    for i in rng.choice(len(unseen), size=min(n_unseen, len(unseen)), replace=False):
        img, ttype = unseen[i]
        rows.append(dict(subset="synthetic_unseen", image_path=str(img),
                         mask_path=str(SYN_ROOT / "masks" / img.with_suffix(".png").name),
                         label=int(ttype != "orig"), tamper_type=ttype, lang=None, doc_type=None))

    forged = sorted((AIFORGE_TEST / "images").glob("*.png"))
    authentic = sorted((AIFORGE_TEST / "authentic").glob("*"))
    for i in rng.choice(len(forged), size=min(n_aiforge // 2, len(forged)), replace=False):
        f = forged[i]
        rows.append(dict(subset="aiforgedoc_ood", image_path=str(f), mask_path=str(AIFORGE_TEST / "masks" / f.name),
                         label=1, tamper_type="aiforgedoc (bilinmiyor)", lang=None, doc_type=None))
    for i in rng.choice(len(authentic), size=min(n_aiforge - n_aiforge // 2, len(authentic)), replace=False):
        a = authentic[i]
        rows.append(dict(subset="aiforgedoc_ood", image_path=str(a), mask_path="", label=0, tamper_type="orig",
                         lang=None, doc_type=a.stem.rsplit("_", 1)[0]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def jpeg_quality(data: bytes) -> float:
    """JPEG dosyasının kalite faktörünü parlaklık niceleme tablosundan tahmin eder (PNG vb. → 100)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            tables = getattr(im, "quantization", None)
            if im.format != "JPEG" or not tables:
                return 100.0
            table = np.array(tables[0], dtype=float)
    except OSError:
        return float("nan")
    # libjpeg: tablo = STD · s / 100, s = 5000/Q (Q<50) veya 200 − 2Q (Q≥50)
    scale = float(np.median(table / STD_LUMA_Q50) * 100)
    quality = 5000 / scale if scale > 100 else (200 - scale) / 2
    return float(np.clip(quality, 1, 100))


def threshold_counts(prob: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(len(THRESHOLDS), 4) — her eşik için TP, FP, FN, TN piksel sayıları."""
    pos = gt.astype(bool)
    n_pos, n = int(pos.sum()), pos.size
    out = np.zeros((len(THRESHOLDS), 4), dtype=np.int64)
    for i, t in enumerate(THRESHOLDS):
        pred = prob >= t
        tp = int((pred & pos).sum())
        fp = int(pred.sum()) - tp
        out[i] = (tp, fp, n_pos - tp, n - n_pos - fp)
    return out


def region_count(gt: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(gt.astype(np.uint8))
    return int(n - 1)


def run_ocr_rules(image: np.ndarray) -> dict:
    from src.ocr_checker import FAIL, PASS, check_document_text, run_ocr

    report = check_document_text(run_ocr(image, gpu=torch.cuda.is_available()))
    return dict(ocr_fail=sum(f.status == FAIL for f in report.findings),
                ocr_pass=sum(f.status == PASS for f in report.findings),
                ocr_risk=report.risk if report.has_evidence else np.nan,
                ocr_evidence=report.has_evidence)


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, config = load_model(args.checkpoint, args.device)
    temperature, threshold, calib_path = load_calibration()
    logger.info("Model: %s | T=%.3f eşik=%.2f (%s) | mod=%s", args.checkpoint, temperature, threshold,
                calib_path, args.mode)

    frame = build_eval_set(args.n_synthetic_unseen, args.n_aiforge, args.seed)
    logger.info("Değerlendirme kümesi: %s", frame.groupby(["subset", "label"]).size().to_dict())
    rng = np.random.default_rng(args.seed)

    records, counts, samples_p, samples_y, samples_img = [], [], [], [], []
    for idx, row in enumerate(frame.itertuples()):
        data = Path(row.image_path).read_bytes()
        image = read_image_rgb(row.image_path)
        h, w = image.shape[:2]
        gt = read_mask(row.mask_path or None, (h, w)) if row.label == 1 else np.zeros((h, w), np.uint8)

        start = time.perf_counter()
        prob, doc_logit = predict_raw(model, image, config, temperature, args.device, args.mode)
        elapsed = time.perf_counter() - start

        c = threshold_counts(prob, gt)
        counts.append(c)
        k = min(PIXEL_SAMPLES_PER_IMAGE, prob.size)
        flat = rng.choice(prob.size, size=k, replace=False)
        samples_p.append(prob.ravel()[flat].astype(np.float16))
        samples_y.append(gt.ravel()[flat].astype(np.uint8))
        samples_img.append(np.full(k, idx, dtype=np.int32))

        rec = dict(row._asdict())
        rec.pop("Index")
        rec.update(width=w, height=h, megapixels=w * h / 1e6, jpeg_quality=jpeg_quality(data),
                   file_kb=len(data) / 1024, infer_s=elapsed, doc_logit=doc_logit, max_pixel=float(prob.max()),
                   mean_pixel=float(prob.mean()), gt_ratio=float(gt.mean()), gt_regions=region_count(gt))
        if args.ocr:
            rec.update(run_ocr_rules(image))
        records.append(rec)
        logger.info("[%3d/%d] %-16s y=%d logit=%+.2f maxpix=%.3f %.1fs  %s", idx + 1, len(frame), row.subset,
                    row.label, doc_logit, rec["max_pixel"], elapsed, Path(row.image_path).name)

    per_image = pd.DataFrame(records)
    per_image.to_csv(out / "per_image.csv", index=False)
    np.savez_compressed(out / "pixel_counts.npz", counts=np.stack(counts), thresholds=THRESHOLDS)
    np.savez_compressed(out / "pixel_samples.npz", prob=np.concatenate(samples_p), label=np.concatenate(samples_y),
                        image=np.concatenate(samples_img))

    # --- JPEG dayanıklılığı ---------------------------------------------------------
    qualities = [int(q) for q in args.qualities.split(",")]
    forged = per_image[(per_image["label"] == 1) & (per_image["gt_ratio"] > 0)]
    clean = per_image[per_image["label"] == 0]
    half = args.robust_n // 2
    subset = pd.concat([forged.sample(min(half, len(forged)), random_state=args.seed),
                        clean.sample(min(args.robust_n - half, len(clean)), random_state=args.seed)])
    robust = []
    for j, row in enumerate(subset.itertuples()):
        image = read_image_rgb(row.image_path)
        h, w = image.shape[:2]
        gt = read_mask(row.mask_path or None, (h, w)) if row.label == 1 else np.zeros((h, w), np.uint8)
        for q in qualities:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
            degraded = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            prob, doc_logit = predict_raw(model, degraded, config, temperature, args.device, args.mode)
            ti = int(np.argmin(np.abs(THRESHOLDS - threshold)))
            tp, fp, fn, tn = threshold_counts(prob, gt)[ti]
            robust.append(dict(image_path=row.image_path, subset=row.subset, label=row.label, quality=q,
                               doc_logit=doc_logit, max_pixel=float(prob.max()), tp=tp, fp=fp, fn=fn, tn=tn))
        logger.info("Dayanıklılık [%2d/%d] %s", j + 1, len(subset), Path(row.image_path).name)
    pd.DataFrame(robust).to_csv(out / "robustness.csv", index=False)

    (out / "meta.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "mode": args.mode, "temperature": temperature, "threshold": threshold,
        "calibration_file": str(calib_path), "thresholds": THRESHOLDS.tolist(), "qualities": qualities,
        "robustness_threshold": float(THRESHOLDS[int(np.argmin(np.abs(THRESHOLDS - threshold)))]),
        "n_images": len(per_image), "ocr": args.ocr, "seed": args.seed,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Sonuçlar kaydedildi: %s", out.resolve())


if __name__ == "__main__":
    main()
