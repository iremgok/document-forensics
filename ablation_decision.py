"""
Karar kuralı, havuzlama, kalibrasyon, OCR ve kopyala-taşı ablasyonu — modeli yeniden çalıştırmadan.

``evaluate.py`` her test/OOD kümesi için ``predictions_<küme>.csv`` ve ``arrays_<küme>.npz``
(görüntü başına eşik taramalı en büyük bölge oranları + pencere logit'leri; yeni çıktılarda birden çok
piksel sıcaklığı için) yazar. Bu betik o dosyalardan aşağıdaki seçenekleri karşılaştırır:

======================  =====================================================================
Boyut                   Seçenekler
======================  =====================================================================
Belge havuzlaması       max · top-%5 · **top-%10** (varsayılan) · top-%25 · ortalama
Skor füzyonu            ``fuse_doc_score`` açık (varsayılan) / kapalı
Karar kuralı            **hibrit + bölge** (varsayılan) · yalnız belge skoru · yalnız piksel bölgesi ·
                        yalnız hibrit koşul
En küçük bölge alanı    %0,01 · **%0,05** · %0,1 · %0,5 (sayfa alanına oranı)
Belge sıcaklığı         val'de öğrenilen T (varsayılan) / T = 1 (yalnız belge skoru; piksel tarafı değişmez)
Kalibrasyon             **ayrı** (summary.json'daki belge/piksel T ve eşikleri) · ``tek_T_eski`` (piksel
                        tarafı belge T'si ve belge eşiğiyle; eski şema) · ``piksel_T1`` (piksel T = 1)
OCR (varsa)             kapalı / noisy-OR hibrit ``1 − (1 − p)(1 − r_OCR)`` (``evaluate.py --ocr``)
Kopyala-taşı (varsa)    kapalı / açık: ``copymove_risk ≥ COPYMOVE_DECISION_RISK`` (``evaluate.py --copymove``)
======================  =====================================================================

Her satırda eşikteki karar metrikleri (TPR, FPR, precision, F1) ve eşikten bağımsız belge AUC /
TPR@FPR=%2 raporlanır. Varsayılan yapılandırmadan yalnızca **tek bir boyut** değiştirilir; böylece
her bileşenin katkısı ayrı görülür. Eşikler ve sıcaklıklar val'de seçilmiş değerlerdir (``summary.json``);
bu betik hiçbir değeri test kümesine göre seçmez.

``tek_T_eski`` / ``piksel_T1`` satırları, ilgili sıcaklığın bölge oranları ``arrays_*.npz`` içinde
(``pixel_temperatures``) varsa hesaplanır; eski ``evaluate.py`` çıktılarında yalnızca ana sıcaklık vardır.

``--informative``: (piksel T, belge eşiği, piksel eşiği) ızgarasında TPR/FPR tablosunu **test kümesinde**
hesaplayıp ``kalibrasyon_bilgi_amacli.csv``'ye yazar. Val kümesi yerelde olmadığında ayrı kalibrasyonun
olası etkisini göstermek içindir; buradaki "en iyi" noktalar test kümesine bakılarak bulunduğu için
iyimserdir ve **eşik seçiminde kullanılmamalıdır**.

Kullanım:
    python ablation_decision.py --results results/stageE
    python ablation_decision.py --results results/stageE --threshold 0.6   # summary.json belge eşiği yerine
    python ablation_decision.py --results results/compare_real/stageE --informative
Çıktı: ``<results>/ablation_decision.csv`` (+ ``kalibrasyon_bilgi_amacli.csv``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from predict import CLEAN_PIXEL_LIMIT, DOC_WEIGHT, MIN_REGION_FRACTION
from src.copymove import COPYMOVE_DECISION_RISK
from src.metrics import binary_decision_metrics, roc_auc, sigmoid, tpr_at_fpr

POOLINGS = {"max": 0.0, "top5": 0.05, "top10": 0.10, "top25": 0.25, "mean": 1.0}
DEFAULT = {"pooling": "top10", "fuse": True, "rule": "hibrit+bolge", "min_region": MIN_REGION_FRACTION,
           "temperature": "val", "kalibrasyon": "ayri", "ocr": False, "copymove": False}
VARIANTS = (
    [("pooling", v) for v in POOLINGS if v != "top10"]
    + [("fuse", False)]
    + [("rule", v) for v in ("yalniz_belge", "yalniz_bolge", "yalniz_hibrit")]
    + [("min_region", v) for v in (0.0001, 0.001, 0.005)]
    + [("temperature", "1")]
    + [("kalibrasyon", v) for v in ("tek_T_eski", "piksel_T1")]
    + [("ocr", True)]
    + [("copymove", True)]
)
INFORMATIVE_NOTE = "TEST KÜMESİNDE HESAPLANDI — YALNIZCA BİLGİ AMAÇLI; EŞİK/SICAKLIK SEÇİMİNDE KULLANMAYIN"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Karar kuralı / havuzlama / kalibrasyon / OCR ablasyonu (yeniden çıkarım yok)")
    p.add_argument("--results", default="results", help="evaluate.py --out klasörü")
    p.add_argument("--threshold", type=float, default=None, help="Belge eşiği (varsayılan: summary.json)")
    p.add_argument("--informative", action="store_true",
                   help="Test kümesinde (piksel T, belge eşiği, piksel eşiği) ızgarası — yalnızca bilgi amaçlı")
    return p.parse_args()


def read_calibration(summary: dict, threshold: Optional[float] = None) -> dict[str, float]:
    """``summary.json``'dan belge/piksel kalibrasyonu (eski çıktılarda tek ``temperature`` / ``threshold``)."""
    doc_t = float(summary.get("doc_temperature", summary.get("temperature", 1.0)))
    doc_thr = float(summary.get("doc_threshold", summary.get("threshold", 0.5)))
    return {"doc_temperature": doc_t, "doc_threshold": doc_thr if threshold is None else threshold,
            "pixel_temperature": float(summary.get("pixel_temperature", doc_t)),
            "pixel_threshold": float(summary.get("pixel_threshold", doc_thr)),
            "min_region_fraction": float(summary.get("min_region_fraction", MIN_REGION_FRACTION))}


def pool(windows: np.ndarray, ratio: float) -> float:
    windows = np.asarray(windows, float)
    if ratio <= 0:
        return float(windows.max())
    k = max(1, math.ceil(ratio * windows.size))
    return float(np.sort(windows)[-k:].mean())


def pixel_side(pred: pd.DataFrame, arrays: dict, temperature: float, main_temperature: float
               ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """İstenen piksel sıcaklığında (bölge oranları (N, T), maks. piksel (N,)); dizilerde yoksa None."""
    temps = arrays.get("pixel_temperatures")
    if temps is not None:
        hits = np.flatnonzero(np.isclose(np.asarray(temps, float), temperature, rtol=1e-6, atol=1e-9))
        if len(hits):
            k = int(hits[0])
            return arrays["regions_by_t"][:, k, :], arrays["max_pixel_by_t"][:, k]
    if np.isclose(temperature, main_temperature, rtol=1e-6, atol=1e-9):
        return arrays["regions"], pred["max_pixel"].to_numpy(float)
    return None


def evaluate_variant(pred: pd.DataFrame, arrays: dict, cfg: dict, cal: dict[str, float]) -> dict[str, float]:
    labels = pred["label"].to_numpy().astype(int)
    doc_thr = cal["doc_threshold"]
    t = cal["doc_temperature"] if cfg["temperature"] == "val" else 1.0
    pix_t, pix_thr = {"ayri": (cal["pixel_temperature"], cal["pixel_threshold"]),
                      "tek_T_eski": (cal["doc_temperature"], cal["doc_threshold"]),
                      "piksel_T1": (1.0, cal["pixel_threshold"])}[cfg["kalibrasyon"]]
    side = pixel_side(pred, arrays, pix_t, cal["pixel_temperature"])
    if side is None:
        return {}
    regions, max_pixel = side
    logits = np.array([pool(w, POOLINGS[cfg["pooling"]]) for w in arrays["windows"]])
    probs = sigmoid(logits / t)
    if cfg["fuse"]:
        probs = np.where(max_pixel < CLEAN_PIXEL_LIMIT, DOC_WEIGHT * probs + (1 - DOC_WEIGHT) * max_pixel, probs)
    ti = int(np.argmin(np.abs(arrays["thresholds"] - pix_thr)))
    region = regions[:, ti]
    hybrid = (probs >= doc_thr) & (max_pixel >= CLEAN_PIXEL_LIMIT)
    pixel = region >= cfg["min_region"]
    decided = {"hibrit+bolge": hybrid | pixel, "yalniz_belge": probs >= doc_thr,
               "yalniz_bolge": pixel, "yalniz_hibrit": hybrid}[cfg["rule"]]
    if cfg["ocr"]:
        if "ocr_risk" not in pred:
            return {}
        risk = pred["ocr_risk"].fillna(0).to_numpy(float)
        probs = 1 - (1 - probs) * (1 - risk)  # noisy-OR hibrit skor
        # Kural ihlali bulunan belgede hibrit skor eşiği geçerse piksel izi olmasa da "sahte" denir.
        decided = decided | ((probs >= doc_thr) & (risk > 0))
    if cfg["copymove"]:
        if "copymove_risk" not in pred:
            return {}
        risk = pred["copymove_risk"].fillna(0).to_numpy(float)
        probs = 1 - (1 - probs) * (1 - risk)
        # Sabit (modülde tanımlı, test kümesine göre seçilmemiş) risk eşiğini geçen eşleşme tek başına yeterli.
        decided = decided | (risk >= COPYMOVE_DECISION_RISK)
    m = binary_decision_metrics(decided, labels)
    return {"tpr": m["recall"], "fpr": 1 - m["specificity"] if m["specificity"] == m["specificity"] else np.nan,
            "precision": m["precision"], "f1": m["f1"], "accuracy": m["accuracy"],
            "auc": roc_auc(probs, labels), "tpr_at_fpr02": tpr_at_fpr(probs, labels, 0.02),
            "piksel_T": pix_t, "piksel_esik": float(arrays["thresholds"][ti]), "belge_esik": doc_thr}


def informative_grid(key: str, pred: pd.DataFrame, arrays: dict, cal: dict[str, float]) -> pd.DataFrame:
    """Test kümesinde (piksel T, belge eşiği, piksel eşiği) ızgarası — yalnızca bilgi amaçlı."""
    labels = pred["label"].to_numpy().astype(bool)
    temps = arrays.get("pixel_temperatures", np.array([cal["pixel_temperature"]]))
    probs_raw = sigmoid(np.array([pool(w, POOLINGS["top10"]) for w in arrays["windows"]]) / cal["doc_temperature"])
    grid = arrays["thresholds"][::5]  # 0.05 adım
    rows = []
    for temp in np.asarray(temps, float):
        side = pixel_side(pred, arrays, float(temp), cal["pixel_temperature"])
        if side is None:
            continue
        regions, max_pixel = side
        probs = np.where(max_pixel < CLEAN_PIXEL_LIMIT, DOC_WEIGHT * probs_raw + (1 - DOC_WEIGHT) * max_pixel,
                         probs_raw)
        for doc_thr in grid:
            hybrid = (probs >= doc_thr) & (max_pixel >= CLEAN_PIXEL_LIMIT)
            for pix_thr in grid:
                ti = int(np.argmin(np.abs(arrays["thresholds"] - pix_thr)))
                decided = hybrid | (regions[:, ti] >= cal["min_region_fraction"])
                rows.append({"kume": key, "piksel_T": float(temp), "belge_esik": float(doc_thr),
                             "piksel_esik": float(pix_thr), "tpr": float(decided[labels].mean()),
                             "fpr": float(decided[~labels].mean()), "not": INFORMATIVE_NOTE})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.results)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    cal = read_calibration(summary, args.threshold)
    rows, grids = [], []
    for pred_path in sorted(root.glob("predictions_*.csv")):
        key = pred_path.stem.removeprefix("predictions_")
        arr_path = root / f"arrays_{key}.npz"
        if not arr_path.is_file():
            print(f"{key}: arrays_{key}.npz yok (eski evaluate.py çıktısı), atlanıyor.")
            continue
        arrays = dict(np.load(arr_path, allow_pickle=True))
        pred = pd.read_csv(pred_path)
        for col in ("max_pixel", "label", "ocr_risk", "copymove_risk"):
            if col in pred:
                pred[col] = pd.to_numeric(pred[col], errors="coerce")
        if len(pred) != len(arrays["regions"]):
            raise ValueError(f"{key}: tahmin ve dizi satır sayıları farklı")
        for dim, value in [(None, None)] + VARIANTS:
            cfg = dict(DEFAULT)
            if dim:
                cfg[dim] = value
            result = evaluate_variant(pred, arrays, cfg, cal)
            if not result:
                continue
            rows.append({"kume": key, "degisen": dim or "(varsayılan)", "deger": "" if dim is None else value,
                         **result})
        if args.informative:
            grids.append(informative_grid(key, pred, arrays, cal))
    table = pd.DataFrame(rows)
    out = root / "ablation_decision.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"Belge T = {cal['doc_temperature']:.3f}, belge eşiği = {cal['doc_threshold']:.2f} · "
              f"piksel T = {cal['pixel_temperature']:.3f}, piksel eşiği = {cal['pixel_threshold']:.2f}")
        print(table.round(3).to_string(index=False))
    print(f"Yazıldı: {out}")
    if grids:
        grid = pd.concat(grids, ignore_index=True)
        grid_out = root / "kalibrasyon_bilgi_amacli.csv"
        grid.to_csv(grid_out, index=False)
        target = float(summary.get("target_fpr") or 0.05)
        print(f"\n{INFORMATIVE_NOTE}\nTest kümesinde FPR ≤ {target:.2f} altındaki en yüksek TPR (iyimser üst sınır):")
        best = (grid[grid["fpr"] <= target].sort_values(["tpr", "fpr"], ascending=[False, True])
                .groupby(["kume", "piksel_T"], sort=False).head(1))
        print(best.drop(columns="not").round(3).to_string(index=False))
        print(f"Yazıldı: {grid_out}")


if __name__ == "__main__":
    main()
