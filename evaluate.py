"""
Bağımsız değerlendirme betiği: AUC, IoU, Piksel-F1 ve kalibrasyon.

Kurallar (sızıntı önleme):
  * Bölümler ``train.py``'nin kaydettiği ``splits.csv``'den okunur; burada yeniden
    bölme yapılmaz.
  * Sıcaklık ölçekleme (kalibrasyon) yalnızca ``val`` üzerinde öğrenilir, ``test`` ve
    ``ood`` kümelerine sabit olarak uygulanır.
  * Her kaynak ayrı raporlanır (ör. sentetik test, AIForge-Doc OOD, SROIE2019 OOD);
    yalnızca orijinal belgelerden oluşan kümelerde (SROIE) AUC tanımsızdır ve
    yanlış alarm oranı (FPR) raporlanır.

Örnek:
    python evaluate.py --checkpoint checkpoints/run/best.pth --splits checkpoints/run/splits.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset import ForgeryDataset, build_transforms
from src.metrics import (PixelMetrics, expected_calibration_error, fit_temperature, image_level_metrics,
                         sigmoid)
from src.models import load_model, sliding_window_predict

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sahtecilik modeli değerlendirmesi")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--splits", required=True, help="train.py tarafından yazılan splits.csv")
    p.add_argument("--sets", nargs="+", default=["test", "ood"], help="Değerlendirilecek bölümler")
    p.add_argument("--no-calibration", action="store_true", help="Val üzerinde sıcaklık öğrenmeyi atla")
    p.add_argument("--max-images", type=int, default=None, help="Kaynak başına üst sınır (hızlı deneme)")
    p.add_argument("--out", default="results")
    return p.parse_args()


@torch.no_grad()
def predict_frame(model: torch.nn.Module, frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Her görüntü için belge logit'ini ve piksel metriklerini hesaplar."""
    dataset = ForgeryDataset(frame, build_transforms(False), config.get("max_side", 1600))
    pixel = PixelMetrics()
    logits = []
    for i in range(len(dataset)):
        sample = dataset[i]
        prob_map, doc_logit = sliding_window_predict(model, sample["image"], tile=config.get("crop_size", 512))
        pixel.update(prob_map, sample["mask"][0].numpy())
        logits.append(doc_logit)
        if (i + 1) % 100 == 0:
            logger.info("  %d / %d", i + 1, len(dataset))
    out = frame.reset_index(drop=True).copy()
    out["doc_logit"] = logits
    out.attrs["pixel"] = pixel.compute()
    return out


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

    # --- Kalibrasyon: yalnızca val ---------------------------------------------
    temperature = 1.0
    if not args.no_calibration:
        val = splits[splits["split"] == "val"]
        if args.max_images:
            val = val.sample(min(len(val), args.max_images), random_state=0)
        if val["label"].nunique() == 2:
            logger.info("Sıcaklık ölçekleme val kümesinde öğreniliyor (%d görüntü)...", len(val))
            val_pred = predict_frame(model, val, config)
            temperature = fit_temperature(val_pred["doc_logit"].to_numpy(), val_pred["label"].to_numpy())
            logger.info("Öğrenilen sıcaklık T = %.3f", temperature)
    (Path(args.checkpoint).parent / "calibration.json").write_text(
        json.dumps({"temperature": temperature}, indent=2), encoding="utf-8")

    # --- Test ve OOD kümeleri ------------------------------------------------------
    report: dict[str, dict] = {}
    for split_name in args.sets:
        subset = splits[splits["split"] == split_name]
        for source, frame in subset.groupby("source"):
            if args.max_images and len(frame) > args.max_images:
                frame = frame.sample(args.max_images, random_state=0)
            key = f"{split_name}/{source}"
            logger.info("Değerlendiriliyor: %s (%d görüntü)", key, len(frame))
            pred = predict_frame(model, frame, config)
            metrics = {**pred.attrs["pixel"],
                       **image_level_metrics(pred["doc_logit"].to_numpy(), pred["label"].to_numpy(), temperature),
                       "n_images": len(pred), "n_forged": int(pred["label"].sum())}
            if pred["label"].nunique() == 2:
                _, table = expected_calibration_error(sigmoid(pred["doc_logit"] / temperature), pred["label"])
                table.to_csv(out_dir / f"reliability_{split_name}_{source}.csv", index=False)
            pred.to_csv(out_dir / f"predictions_{split_name}_{source}.csv", index=False)
            report[key] = metrics
            logger.info("%s → %s", key, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    summary = pd.DataFrame(report).T
    summary.to_csv(out_dir / "summary.csv")
    (out_dir / "summary.json").write_text(json.dumps({"temperature": temperature, "results": report},
                                                     indent=2, default=float), encoding="utf-8")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
