"""
Kapsamlı değerlendirme — ``app.py``'deki "Model Başarımı" sekmesinin ayrıntılı analiz bölümünün veri kaynağı.

Çalıştırma:
    python benchmark.py                          # Stage E, test protokolü (alt küme başına ≤ 60 görüntü, ~15 dk CPU)
    python benchmark.py --splits results/protocol/stageE_kaggle_test_real.csv --max-per-subset 0   # tamamı
    python benchmark.py --splits results/protocol/test_v1.csv --ocr
    python benchmark.py --checkpoint checkpoints/finetuned_multilingual_stageB.pth   # karşılaştırma: Stage B
    python benchmark.py --legacy-sets            # eski sabit örneklem (sentetik v1 + AIForge-Doc)

Çıktı klasörü modelin adını taşır (``results/benchmark_<model>``) ve ``meta.json`` checkpoint yolunu kaydeder;
dashboard sonuçları kenar çubuğunda seçili modelle bu kayıt üzerinden eşleştirir. (Eski ``results/stageB``
klasörü, 22.10'da ``finetuned_multilingual.pth`` ile üretilmiştir; o tarihte bu dosya bir Stage C ara
modeliydi — bkz. ``src/model_registry.py``.)

Değerlendirme kümesi — test protokolüne bağlı (``--splits``, varsayılan
``results/protocol/stageE_kaggle_test.csv``): ``--split`` (varsayılan ``test``) satırları alınır, alt kümeler
``source`` sütunundan gelir (ör. ``real_forged``, ``synthetic``). Tahrifat türü dosya adındaki ``__NN_<tür>``
ekinden, dil ``lang`` sütunundan okunur. ``--max-per-subset N`` alt küme ve sınıf başına seed'li alt örneklem
alır (CPU süresi için; 0 = tamamı). JPEG ve WhatsApp dayanıklılık eğrileri **bu kümeden** seçilen alt kümede
hesaplanır. Böylece dashboard ile ``evaluate.py`` aynı protokolü kullanır.

``--legacy-sets`` eski sabit kodlu örneklemi korur (modelin eğitimde görmediği görseller):
  * ``synthetic_test``  — Stage B ``splits.csv`` içindeki sentetik **test** bölümü,
  * ``synthetic_unseen`` — ``data/4_synthetic_tr_en`` içinde Stage B bölümlerinde hiç geçmeyen gruplar,
  * ``aiforgedoc_ood``  — AIForge-Doc TestingSet (dağılım dışı; sahte + orijinal).

Çıktılar (``--out``, varsayılan ``results/benchmark_<model>``):
  * ``per_image.csv``      — görsel başına skorlar, GT istatistikleri, süre, JPEG kalitesi.
  * ``pixel_counts.npz``   — görsel × eşik ızgarası için piksel TP/FP/FN/TN sayıları ve eşiği geçen
    en büyük bağlantılı bölgenin sayfa oranı (``max_region``; ``app.is_forged`` piksel koşulu için).
  * ``pixel_samples.npz``  — görsel başına düzgün rastgele piksel örneklemi (ROC/PR için).
  * ``robustness.csv``     — JPEG kalite faktörüne göre (``kanal = jpeg``) ve WhatsApp benzetimiyle
    (``kanal = whatsapp``: uzun kenar ≤ 1600 + JPEG q75) yeniden kaydedilmiş alt kümede sonuçlar.
  * ``meta.json``          — checkpoint, ayrı belge/piksel kalibrasyonu, splits, eşik ızgarası, örneklem boyutları.

Not: Piksel olasılıkları modelin **piksel** sıcaklığıyla (``calibration.json`` → ``pixel_temperature``; eski
şemada tek T), belge skoru ise **ham logit** olarak kaydedilir; panel belge sıcaklığını sonradan uygular.
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

from predict import largest_region_fraction, load_calibration_info, predict_raw
from src.dataset import read_image_rgb, read_mask
from src.models import load_model

logger = logging.getLogger("benchmark")

SYN_ROOT = Path("data/4_synthetic_tr_en")
AIFORGE_TEST = Path("data/finetune_aiforgedoc/TestingSet")
STAGEB_SPLITS = Path("checkpoints/synthetic/splits.csv")
DEFAULT_SPLITS = Path("results/protocol/stageE_kaggle_test.csv")
TAMPER_RE = re.compile(r"__\d{2}_([A-Za-z_]+)\.\w+$")
WHATSAPP_MAX_SIDE, WHATSAPP_QUALITY = 1600, 75
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
PIXEL_SAMPLES_PER_IMAGE = 4000
SYN_NAME = re.compile(r"(syn\d+)__\d+_(.+)\.\w+$")

# libjpeg standart parlaklık niceleme tablosu (Q=50); kalite tahmini için referans
STD_LUMA_Q50 = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model başarımı değerlendirmesi (dashboard ayrıntılı analiz)")
    p.add_argument("--checkpoint", default="checkpoints/stageE/best.pth")
    p.add_argument("--mode", choices=("sliding", "resize"), default="sliding")
    p.add_argument("--out", default=None, help="Varsayılan: results/benchmark_<model> (ör. benchmark_stageE)")
    p.add_argument("--splits", default=str(DEFAULT_SPLITS),
                   help="Test protokolü (image_path, mask_path, label, source, split); alt kümeler source sütunundan")
    p.add_argument("--split", default="test", help="--splits içinden alınacak bölüm")
    p.add_argument("--max-per-subset", type=int, default=60,
                   help="Alt küme ve sınıf başına en fazla görüntü (seed'li örneklem; 0 = tamamı)")
    p.add_argument("--legacy-sets", action="store_true",
                   help="Eski sabit örneklem: sentetik v1 test/görülmemiş + AIForge-Doc (--splits yok sayılır)")
    p.add_argument("--n-synthetic-unseen", type=int, default=40, help="(--legacy-sets) Görülmemiş sentetik görsel sayısı")
    p.add_argument("--n-aiforge", type=int, default=60, help="(--legacy-sets) AIForge-Doc görsel sayısı")
    p.add_argument("--robust-n", type=int, default=24, help="JPEG dayanıklılık alt kümesi boyutu")
    p.add_argument("--qualities", default="100,90,80,70,60,50", help="JPEG kalite faktörleri")
    p.add_argument("--ocr", action="store_true", help="OCR + kural motorunu da çalıştır (yavaş)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Değerlendirme kümesi
# ---------------------------------------------------------------------------

def build_protocol_set(splits_path: str | Path, split: str, max_per_subset: int, seed: int) -> pd.DataFrame:
    """Test protokolünden değerlendirme kümesi: alt küme = ``source``; tür dosya adından, dil ``lang``'dan."""
    splits = pd.read_csv(splits_path, keep_default_na=False)
    frame = splits[splits["split"] == split].copy()
    if frame.empty:
        raise SystemExit(f"{splits_path} içinde '{split}' bölümü yok.")
    if max_per_subset > 0:
        frame = pd.concat([g.sample(min(len(g), max_per_subset), random_state=seed)
                           for _, g in frame.groupby(["source", "label"])])
    paths = frame["image_path"].astype(str).str.replace("\\", "/", regex=False)
    tamper = [m.group(1) if (m := TAMPER_RE.search(p)) else ("orig" if int(l) == 0 else "bilinmiyor")
              for p, l in zip(paths, frame["label"])]
    return pd.DataFrame({
        "subset": frame["source"].to_numpy(), "image_path": paths.to_numpy(),
        "mask_path": frame["mask_path"].astype(str).str.replace("\\", "/", regex=False).to_numpy(),
        "label": frame["label"].astype(int).to_numpy(), "tamper_type": tamper,
        "lang": frame["lang"].replace("", None).to_numpy() if "lang" in frame else None,
        "doc_type": frame["doc_type"].replace("", None).to_numpy() if "doc_type" in frame else None,
    }).sort_values(["subset", "label", "image_path"]).reset_index(drop=True)


def whatsapp(image: np.ndarray) -> np.ndarray:
    """WhatsApp benzetimi: uzun kenar ≤ 1600 + JPEG q75 (``generate_synthetic.DIGITAL_CHANNELS`` ile uyumlu)."""
    h, w = image.shape[:2]
    scale = WHATSAPP_MAX_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, WHATSAPP_QUALITY])
    out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR) if out.shape[:2] != (h, w) else out


def build_eval_set(n_unseen: int, n_aiforge: int, seed: int) -> pd.DataFrame:
    """``--legacy-sets``: eski sabit kodlu örneklem (sentetik v1 + AIForge-Doc)."""
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


def region_fractions(prob: np.ndarray) -> np.ndarray:
    """(len(THRESHOLDS),) — her eşikte eşiği geçen en büyük bağlantılı bölgenin sayfa oranı."""
    return np.array([largest_region_fraction(prob, t) for t in THRESHOLDS], dtype=np.float32)


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
    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else Path("results") / f"benchmark_{ckpt.parent.name if ckpt.stem in ('best', 'last') else ckpt.stem}"
    out.mkdir(parents=True, exist_ok=True)

    model, config = load_model(args.checkpoint, args.device)
    cal = load_calibration_info(checkpoint=args.checkpoint)
    # Haritalar piksel sıcaklığıyla; dayanıklılık eşiği piksel eşiğidir.
    temperature, threshold, calib_path = cal.pixel_temperature, cal.pixel_threshold, cal.path
    logger.info("Model: %s | %s (%s) | mod=%s", args.checkpoint, cal.summary(), calib_path, args.mode)

    if args.legacy_sets:
        frame = build_eval_set(args.n_synthetic_unseen, args.n_aiforge, args.seed)
    else:
        frame = build_protocol_set(args.splits, args.split, args.max_per_subset, args.seed)
    logger.info("Değerlendirme kümesi: %s", frame.groupby(["subset", "label"]).size().to_dict())
    rng = np.random.default_rng(args.seed)

    records, counts, regions, samples_p, samples_y, samples_img = [], [], [], [], [], []
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
        regions.append(region_fractions(prob))
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
    np.savez_compressed(out / "pixel_counts.npz", counts=np.stack(counts), max_region=np.stack(regions),
                        thresholds=THRESHOLDS)
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
        variants = []
        for q in qualities:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
            variants.append(("jpeg", q, cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))
        variants.append(("whatsapp", WHATSAPP_QUALITY, whatsapp(image)))
        ti = int(np.argmin(np.abs(THRESHOLDS - threshold)))
        for kanal, q, degraded in variants:
            prob, doc_logit = predict_raw(model, degraded, config, temperature, args.device, args.mode)
            tp, fp, fn, tn = threshold_counts(prob, gt)[ti]
            robust.append(dict(image_path=row.image_path, subset=row.subset, label=row.label, kanal=kanal, quality=q,
                               doc_logit=doc_logit, max_pixel=float(prob.max()), tp=tp, fp=fp, fn=fn, tn=tn))
        logger.info("Dayanıklılık [%2d/%d] %s", j + 1, len(subset), Path(row.image_path).name)
    pd.DataFrame(robust).to_csv(out / "robustness.csv", index=False)

    (out / "meta.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "mode": args.mode,
        # "temperature" / "threshold": eski okuyucular için piksel tarafı (haritaların üretildiği değerler).
        "temperature": temperature, "threshold": threshold,
        "doc_temperature": cal.doc_temperature, "doc_threshold": cal.doc_threshold,
        "pixel_temperature": cal.pixel_temperature, "pixel_threshold": cal.pixel_threshold,
        "calibrated": cal.calibrated, "splits": None if args.legacy_sets else Path(args.splits).as_posix(),
        "split": None if args.legacy_sets else args.split, "max_per_subset": args.max_per_subset,
        "calibration_file": str(calib_path), "thresholds": THRESHOLDS.tolist(), "qualities": qualities,
        "whatsapp": {"max_side": WHATSAPP_MAX_SIDE, "quality": WHATSAPP_QUALITY},
        "robustness_threshold": float(THRESHOLDS[int(np.argmin(np.abs(THRESHOLDS - threshold)))]),
        "n_images": len(per_image), "ocr": args.ocr, "seed": args.seed,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Sonuçlar kaydedildi: %s", out.resolve())


if __name__ == "__main__":
    main()
