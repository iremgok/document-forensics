"""
Bağımsız değerlendirme betiği: AUC, IoU, Piksel-F1, kalibrasyon ve karar eşiği seçimi.

Kurallar (sızıntı önleme):
  * Bölümler ``train.py``'nin kaydettiği ``splits.csv``'den okunur; burada yeniden
    bölme yapılmaz.
  * Sıcaklık ölçekleme ve karar eşiği yalnızca ``val`` üzerinde öğrenilir, ``test`` ve
    ``ood`` kümelerine sabit olarak uygulanır.
  * Her kaynak ayrı raporlanır (ör. sentetik test, AIForge-Doc OOD, SROIE2019 OOD);
    yalnızca orijinal belgelerden oluşan kümelerde (SROIE) AUC tanımsızdır ve
    yanlış alarm oranı (FPR) raporlanır.

Karar eşiği
-----------
Uygulama bir belgeye ``predict.is_forged`` kuralıyla (belge skoru + piksel bölgesi) "sahte" der.
Eşik, val kümesinde bu kuralın **orijinal belgelerdeki yanlış alarm oranı** ``--target-fpr``'ı
(varsayılan %5) aşmayacak şekilde, bu koşulda sahte belgeleri en çok yakalayan değer olarak seçilir
ve sıcaklıkla birlikte ``<checkpoint klasörü>/calibration.json``'a yazılır. Uygulama ve
``predict.py`` bu dosyayı modelin kendi klasöründen okur.

Piksel olasılıkları uygulamadaki gibi sıcaklıkla ölçeklendiği için sıcaklık öğrenildikten sonra
val kümesi o sıcaklıkla bir kez daha işlenir (T = 1 ise tekrar edilmez).

Örnek:
    python evaluate.py --checkpoint checkpoints/run/best.pth --splits checkpoints/run/splits.csv
    python evaluate.py --checkpoint checkpoints/stageE/best.pth --splits checkpoints/stageE/splits.csv --target-fpr 0.02
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predict import decide_forged, fuse_doc_score, largest_region_fraction
from src.dataset import ForgeryDataset, build_transforms
from src.metrics import (PixelMetrics, expected_calibration_error, fit_temperature, image_level_metrics,
                         sigmoid)
from src.models import load_model, sliding_window_predict

logger = logging.getLogger("evaluate")

#: Taranan karar eşikleri (uygulamadaki kaydırıcı aralığı, 0.01 adım).
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sahtecilik modeli değerlendirmesi")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--splits", required=True, help="train.py tarafından yazılan splits.csv")
    p.add_argument("--sets", nargs="+", default=["test", "ood"], help="Değerlendirilecek bölümler")
    p.add_argument("--no-calibration", action="store_true", help="Val üzerinde sıcaklık ve eşik öğrenmeyi atla")
    p.add_argument("--target-fpr", type=float, default=0.05,
                   help="Eşik seçiminde val orijinallerinde izin verilen en yüksek yanlış alarm oranı")
    p.add_argument("--max-images", type=int, default=None,
                   help="Kaynak ve sınıf (orijinal/sahte) başına üst sınır (hızlı deneme)")
    p.add_argument("--out", default="results")
    return p.parse_args()


@torch.no_grad()
def predict_frame(model: torch.nn.Module, frame: pd.DataFrame, config: dict, temperature: float = 1.0,
                  pixel_threshold: float = 0.5) -> pd.DataFrame:
    """Her görüntü için belge logit'ini, karar özetlerini ve piksel metriklerini hesaplar.

    Piksel haritası uygulamadaki gibi ``temperature`` ile ölçeklenir. ``out.attrs['regions']``,
    her görüntü için ``THRESHOLDS`` eşiklerindeki en büyük bölge oranlarıdır (N, len(THRESHOLDS)).
    """
    dataset = ForgeryDataset(frame, build_transforms(False), config.get("max_side", 1600))
    pixel = PixelMetrics(threshold=pixel_threshold)
    logits, max_pixels, regions = [], [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        prob_map, doc_logit = sliding_window_predict(model, sample["image"], tile=config.get("crop_size", 512),
                                                     temperature=temperature)
        pixel.update(prob_map, sample["mask"][0].numpy())
        logits.append(doc_logit)
        max_pixels.append(float(prob_map.max()))
        regions.append([largest_region_fraction(prob_map, t) for t in THRESHOLDS])
        if (i + 1) % 100 == 0:
            logger.info("  %d / %d", i + 1, len(dataset))
    out = frame.reset_index(drop=True).copy()
    out["doc_logit"] = logits
    out["max_pixel"] = max_pixels
    out.attrs["pixel"] = pixel.compute()
    out.attrs["regions"] = np.asarray(regions, dtype=np.float64).reshape(len(out), len(THRESHOLDS))
    return out


def decisions_at(pred: pd.DataFrame, temperature: float, ti: int) -> np.ndarray:
    """``THRESHOLDS[ti]`` eşiğinde her görüntü için ``predict.is_forged`` kararı."""
    threshold = float(THRESHOLDS[ti])
    doc_probs = sigmoid(pred["doc_logit"].to_numpy() / temperature)
    return np.array([decide_forged(fuse_doc_score(float(p), m), m, r, threshold)
                     for p, m, r in zip(doc_probs, pred["max_pixel"], pred.attrs["regions"][:, ti])])


def select_threshold(pred: pd.DataFrame, temperature: float, target_fpr: float) -> dict[str, float]:
    """Val'de FPR ≤ ``target_fpr`` koşulunda TPR'si en yüksek eşik (eşitlikte küçük eşik).

    Hiçbir eşik hedefi tutturamazsa FPR'si en düşük eşik seçilir ve uyarı verilir.
    """
    labels = pred["label"].to_numpy().astype(int)
    rows = []
    for ti, threshold in enumerate(THRESHOLDS):
        decided = decisions_at(pred, temperature, ti)
        rows.append({"threshold": float(threshold), "fpr": float(decided[labels == 0].mean()),
                     "tpr": float(decided[labels == 1].mean())})
    table = pd.DataFrame(rows)
    feasible = table[table["fpr"] <= target_fpr]
    if feasible.empty:
        best = table.sort_values(["fpr", "tpr"], ascending=[True, False]).iloc[0]
        logger.warning("Hiçbir eşik val'de FPR ≤ %.2f sağlamıyor; en düşük FPR'li eşik seçildi "
                       "(eşik=%.2f, FPR=%.3f, TPR=%.3f). Model yeniden eğitilmeli.",
                       target_fpr, best["threshold"], best["fpr"], best["tpr"])
    else:
        best = feasible.sort_values(["tpr", "threshold"], ascending=[False, True]).iloc[0]
    return {"threshold": float(best["threshold"]), "fpr": float(best["fpr"]), "tpr": float(best["tpr"])}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    splits = pd.read_csv(args.splits, keep_default_na=False)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "aiforgedoc" in config.get("sources", []) and (splits["source"] == "sroie2019").any():
        logger.warning("Model AIForge-Doc ile eğitildi; AIForge-Doc SROIE fişlerinden türetildiği için "
                       "örtüşen SROIE dosyaları splits.csv'de zaten çıkarılmıştır.")

    # --- Kalibrasyon: yalnızca val (sıcaklık + karar eşiği) ----------------------------
    temperature, threshold = 1.0, 0.5
    calibration: dict[str, float] = {}
    if not args.no_calibration:
        val = splits[splits["split"] == "val"]
        if args.max_images:
            # Sınıf başına üst sınır: eşik seçimi için iki sınıf da temsil edilmeli.
            val = pd.concat([g.sample(min(len(g), args.max_images), random_state=0) for _, g in val.groupby("label")])
        if val["label"].nunique() == 2:
            logger.info("Sıcaklık ölçekleme val kümesinde öğreniliyor (%d görüntü)...", len(val))
            val_pred = predict_frame(model, val, config)
            temperature = fit_temperature(val_pred["doc_logit"].to_numpy(), val_pred["label"].to_numpy())
            logger.info("Öğrenilen sıcaklık T = %.3f", temperature)
            if abs(temperature - 1.0) > 1e-6:
                logger.info("Eşik seçimi için val, T=%.3f piksel haritalarıyla yeniden işleniyor...", temperature)
                val_pred = predict_frame(model, val, config, temperature)
            chosen = select_threshold(val_pred, temperature, args.target_fpr)
            threshold = chosen["threshold"]
            calibration = {"target_fpr": args.target_fpr, "val_fpr": chosen["fpr"], "val_tpr": chosen["tpr"],
                           "val_images": len(val_pred)}
            logger.info("Seçilen karar eşiği = %.2f (val: FPR=%.3f, TPR=%.3f, hedef FPR ≤ %.2f)",
                        threshold, chosen["fpr"], chosen["tpr"], args.target_fpr)
        else:
            logger.warning("Val kümesinde iki sınıf yok; sıcaklık ve eşik öğrenilemedi (T=1, eşik=0.5).")
        calib_path = Path(args.checkpoint).parent / "calibration.json"
        calib_path.write_text(json.dumps({"temperature": temperature, "optimal_threshold": threshold, **calibration},
                                         indent=2), encoding="utf-8")
        logger.info("Kalibrasyon yazıldı: %s", calib_path)

    # --- Test ve OOD kümeleri ------------------------------------------------------
    ti = int(np.argmin(np.abs(THRESHOLDS - threshold)))
    report: dict[str, dict] = {}
    for split_name in args.sets:
        subset = splits[splits["split"] == split_name]
        for source, frame in subset.groupby("source"):
            if args.max_images:
                frame = pd.concat([g.sample(min(len(g), args.max_images), random_state=0)
                                   for _, g in frame.groupby("label")])
            key = f"{split_name}/{source}"
            logger.info("Değerlendiriliyor: %s (%d görüntü)", key, len(frame))
            pred = predict_frame(model, frame, config, temperature, threshold)
            labels = pred["label"].to_numpy().astype(int)
            decided = decisions_at(pred, temperature, ti)
            pred["forged_decision"] = decided.astype(int)
            metrics = {**pred.attrs["pixel"],
                       **image_level_metrics(pred["doc_logit"].to_numpy(), labels, temperature),
                       # Uygulamanın verdiği karar (is_forged, seçilen eşik): kullanıcının gördüğü başarım.
                       "decision_acc": float((decided == labels).mean()),
                       "decision_tpr": float(decided[labels == 1].mean()) if (labels == 1).any() else float("nan"),
                       "decision_fpr": float(decided[labels == 0].mean()) if (labels == 0).any() else float("nan"),
                       "n_images": len(pred), "n_forged": int(labels.sum())}
            if pred["label"].nunique() == 2:
                metrics["decision_balanced_acc"] = (metrics["decision_tpr"] + 1 - metrics["decision_fpr"]) / 2
                _, table = expected_calibration_error(sigmoid(pred["doc_logit"] / temperature), pred["label"])
                table.to_csv(out_dir / f"reliability_{split_name}_{source}.csv", index=False)
            pred.to_csv(out_dir / f"predictions_{split_name}_{source}.csv", index=False)
            report[key] = metrics
            logger.info("%s → %s", key, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    summary = pd.DataFrame(report).T
    summary.to_csv(out_dir / "summary.csv")
    (out_dir / "summary.json").write_text(json.dumps({"temperature": temperature, "threshold": threshold,
                                                      **calibration, "results": report},
                                                     indent=2, default=float), encoding="utf-8")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
