"""
Karar kuralı, havuzlama, kalibrasyon ve OCR ablasyonu — modeli yeniden çalıştırmadan.

``evaluate.py`` her test/OOD kümesi için ``predictions_<küme>.csv`` ve ``arrays_<küme>.npz``
(görüntü başına eşik taramalı en büyük bölge oranları + pencere logit'leri) yazar. Bu betik o
dosyalardan aşağıdaki seçenekleri **aynı eşikte** karşılaştırır:

======================  =====================================================================
Boyut                   Seçenekler
======================  =====================================================================
Belge havuzlaması       max · top-%5 · **top-%10** (varsayılan) · top-%25 · ortalama
Skor füzyonu            ``fuse_doc_score`` açık (varsayılan) / kapalı
Karar kuralı            **hibrit + bölge** (varsayılan) · yalnız belge skoru · yalnız piksel bölgesi ·
                        yalnız hibrit koşul
En küçük bölge alanı    %0,01 · **%0,05** · %0,1 · %0,5 (sayfa alanına oranı)
Sıcaklık                val'de öğrenilen T (varsayılan) / T = 1
OCR (varsa)             kapalı / noisy-OR hibrit ``1 − (1 − p)(1 − r_OCR)`` (``evaluate.py --ocr``)
======================  =====================================================================

Her satırda eşikteki karar metrikleri (TPR, FPR, precision, F1) ve eşikten bağımsız belge AUC /
TPR@FPR=%2 raporlanır. Varsayılan yapılandırmadan yalnızca **tek bir boyut** değiştirilir; böylece
her bileşenin katkısı ayrı görülür.

Kullanım:
    python ablation_decision.py --results results/stageE
    python ablation_decision.py --results results/stageE --threshold 0.6   # summary.json eşiği yerine
Çıktı: ``<results>/ablation_decision.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from predict import CLEAN_PIXEL_LIMIT, DOC_WEIGHT, MIN_REGION_FRACTION
from src.metrics import binary_decision_metrics, roc_auc, sigmoid, tpr_at_fpr

POOLINGS = {"max": 0.0, "top5": 0.05, "top10": 0.10, "top25": 0.25, "mean": 1.0}
DEFAULT = {"pooling": "top10", "fuse": True, "rule": "hibrit+bolge", "min_region": MIN_REGION_FRACTION,
           "temperature": "val", "ocr": False}
VARIANTS = (
    [("pooling", v) for v in POOLINGS if v != "top10"]
    + [("fuse", False)]
    + [("rule", v) for v in ("yalniz_belge", "yalniz_bolge", "yalniz_hibrit")]
    + [("min_region", v) for v in (0.0001, 0.001, 0.005)]
    + [("temperature", "1")]
    + [("ocr", True)]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Karar kuralı / havuzlama / OCR ablasyonu (yeniden çıkarım yok)")
    p.add_argument("--results", default="results", help="evaluate.py --out klasörü")
    p.add_argument("--threshold", type=float, default=None, help="Karar eşiği (varsayılan: summary.json)")
    return p.parse_args()


def pool(windows: np.ndarray, ratio: float) -> float:
    windows = np.asarray(windows, float)
    if ratio <= 0:
        return float(windows.max())
    k = max(1, math.ceil(ratio * windows.size))
    return float(np.sort(windows)[-k:].mean())


def evaluate_variant(pred: pd.DataFrame, arrays: dict, cfg: dict, threshold: float, temperature: float
                     ) -> dict[str, float]:
    labels = pred["label"].to_numpy().astype(int)
    t = temperature if cfg["temperature"] == "val" else 1.0
    logits = np.array([pool(w, POOLINGS[cfg["pooling"]]) for w in arrays["windows"]])
    probs = sigmoid(logits / t)
    max_pixel = pred["max_pixel"].to_numpy(float)
    if cfg["fuse"]:
        probs = np.where(max_pixel < CLEAN_PIXEL_LIMIT, DOC_WEIGHT * probs + (1 - DOC_WEIGHT) * max_pixel, probs)
    ti = int(np.argmin(np.abs(arrays["thresholds"] - threshold)))
    region = arrays["regions"][:, ti]
    hybrid = (probs >= threshold) & (max_pixel >= CLEAN_PIXEL_LIMIT)
    pixel = region >= cfg["min_region"]
    decided = {"hibrit+bolge": hybrid | pixel, "yalniz_belge": probs >= threshold,
               "yalniz_bolge": pixel, "yalniz_hibrit": hybrid}[cfg["rule"]]
    if cfg["ocr"]:
        if "ocr_risk" not in pred:
            return {}
        risk = pred["ocr_risk"].fillna(0).to_numpy(float)
        probs = 1 - (1 - probs) * (1 - risk)  # noisy-OR hibrit skor
        # Kural ihlali bulunan belgede hibrit skor eşiği geçerse piksel izi olmasa da "sahte" denir.
        decided = decided | ((probs >= threshold) & (risk > 0))
    m = binary_decision_metrics(decided, labels)
    return {"tpr": m["recall"], "fpr": 1 - m["specificity"] if m["specificity"] == m["specificity"] else np.nan,
            "precision": m["precision"], "f1": m["f1"], "accuracy": m["accuracy"],
            "auc": roc_auc(probs, labels), "tpr_at_fpr02": tpr_at_fpr(probs, labels, 0.02)}


def main() -> None:
    args = parse_args()
    root = Path(args.results)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    temperature = float(summary.get("temperature", 1.0))
    threshold = args.threshold if args.threshold is not None else float(summary.get("threshold", 0.5))
    rows = []
    for pred_path in sorted(root.glob("predictions_*.csv")):
        key = pred_path.stem.removeprefix("predictions_")
        arr_path = root / f"arrays_{key}.npz"
        if not arr_path.is_file():
            print(f"{key}: arrays_{key}.npz yok (eski evaluate.py çıktısı), atlanıyor.")
            continue
        arrays = dict(np.load(arr_path, allow_pickle=True))
        pred = pd.read_csv(pred_path)
        for col in ("max_pixel", "label", "ocr_risk"):
            if col in pred:
                pred[col] = pd.to_numeric(pred[col], errors="coerce")
        if len(pred) != len(arrays["regions"]):
            raise ValueError(f"{key}: tahmin ve dizi satır sayıları farklı")
        for dim, value in [(None, None)] + VARIANTS:
            cfg = dict(DEFAULT)
            if dim:
                cfg[dim] = value
            result = evaluate_variant(pred, arrays, cfg, threshold, temperature)
            if not result:
                continue
            rows.append({"kume": key, "degisen": dim or "(varsayılan)", "deger": "" if dim is None else value,
                         **result})
    table = pd.DataFrame(rows)
    out = root / "ablation_decision.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"T = {temperature:.3f}, eşik = {threshold:.2f}")
        print(table.round(3).to_string(index=False))
    print(f"Yazıldı: {out}")


if __name__ == "__main__":
    main()
