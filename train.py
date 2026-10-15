"""
Ana eğitim ve validasyon döngüsü.

Akış:
  1. Manifest oluşturulur ve **grup düzeyinde** train/val/test bölümlerine ayrılır
     (``src/dataset.py``). Bölüm listesi ``<run_dir>/splits.csv`` olarak kaydedilir;
     ``evaluate.py`` test kümesini **yalnızca bu dosyadan** okur.
  2. Eğitim yalnızca ``train``, model seçimi/erken durdurma yalnızca ``val`` ile yapılır.
     ``test`` ve ``ood`` satırları bu betikte hiçbir hesaba girmez.
  3. Her epoch sonunda: konsol + (kuruluysa) TensorBoard günlüğü, ``last.pth`` ve
     izlenen metrik iyileştiyse ``best.pth`` kaydı, erken durdurma kontrolü.

Örnek:
    python train.py --sources synthetic --epochs 40 --batch-size 8
    python train.py --sources synthetic aiforgedoc --init checkpoints/base/best.pth --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import ForgeryDataset, assign_splits, build_manifest, build_transforms
from src.losses import ForensicLoss
from src.metrics import PixelMetrics, image_level_metrics
from src.models import DualStreamForensicNet, save_checkpoint, sliding_window_predict

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB+SRM çift akışlı sahtecilik modeli eğitimi")
    p.add_argument("--data-root", default="data")
    p.add_argument("--sources", nargs="+", default=["synthetic"],
                   choices=["synthetic", "doctamper", "aiforgedoc"],
                   help="Eğitim/val'de kullanılacak kaynaklar; diğerleri yalnızca test/OOD olur.")
    p.add_argument("--run-dir", default="checkpoints/run")
    p.add_argument("--init", default=None, help="İnce ayar için başlangıç kontrol noktası (.pth)")
    p.add_argument("--encoder", default="efficientnet-b4")
    p.add_argument("--encoder-weights", default="imagenet", help="'none' ile rastgele başlatma")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--max-side", type=int, default=1600, help="Uzun kenar sınırı (yalnızca küçültme)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8, help="Erken durdurma sabrı (epoch)")
    p.add_argument("--monitor", default="score", choices=["score", "pixel_f1", "image_auc", "loss"],
                   help="score = (pixel_f1 + image_auc) / 2")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Karışık hassasiyeti kapat")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    """Tekrarlanabilirlik için tüm rastgele üreteçleri sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """İzlenen metrik ``patience`` epoch boyunca iyileşmezse eğitimi durdurur."""

    def __init__(self, patience: int, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Değeri kaydeder; yeni en iyi ise ``True`` döndürür."""
        improved = (self.best is None or
                    (value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta))
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


class MetricLogger:
    """Konsol + opsiyonel TensorBoard günlükleyici."""

    def __init__(self, log_dir: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(log_dir))
        except ImportError:
            self.writer = None
            logger.info("TensorBoard kurulu değil; yalnızca konsol günlüğü kullanılacak.")

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        logger.info("Epoch %3d | %s", epoch, " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)
            self.writer.flush()


def train_one_epoch(model: DualStreamForensicNet, loader: DataLoader, criterion: ForensicLoss,
                    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device,
                    use_amp: bool) -> dict[str, float]:
    """Bir epoch eğitim; kayıp bileşenlerinin ortalamasını döndürür."""
    model.train()
    totals: dict[str, float] = {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            mask_logits, cls_logit = model(images)
        losses = criterion(mask_logits, cls_logit, masks)  # Kayıp float32'de hesaplanır.
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {f"train/{k}": v / max(len(loader), 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model: DualStreamForensicNet, dataset: ForgeryDataset, criterion: ForensicLoss,
             crop_size: int) -> dict[str, float]:
    """
    Validasyon: her belge tam çözünürlükte kayan pencere ile işlenir (uygulamadaki
    çıkarımla birebir aynı). Kayıp, tam görüntü olasılık haritası üzerinden hesaplanır.
    """
    model.eval()
    pixel = PixelMetrics()
    doc_logits, labels, losses = [], [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        prob_map, doc_logit = sliding_window_predict(model, sample["image"], tile=crop_size)
        gt = sample["mask"][0].numpy()
        pixel.update(prob_map, gt)
        doc_logits.append(doc_logit)
        labels.append(int(sample["label"]))
        logits = torch.logit(torch.from_numpy(prob_map).clamp(1e-6, 1 - 1e-6))[None, None]
        out = criterion(logits, torch.tensor([doc_logit]), torch.from_numpy(gt)[None, None])
        losses.append(float(out["loss"]))
    metrics = {**pixel.compute(), **image_level_metrics(np.array(doc_logits), np.array(labels))}
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    auc = metrics["image_auc"] if not np.isnan(metrics["image_auc"]) else 0.5
    metrics["score"] = (metrics["pixel_f1"] + auc) / 2
    return {f"val/{k}": v for k, v in metrics.items()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    # --- 1) Sızıntısız bölümleme -------------------------------------------------
    manifest = build_manifest(args.data_root, args.sources, cache_dir=Path(args.data_root) / "splits")
    splits = assign_splits(manifest, seed=args.seed)
    splits.to_csv(run_dir / "splits.csv", index=False)
    train_df, val_df = splits[splits["split"] == "train"], splits[splits["split"] == "val"]
    if train_df.empty or val_df.empty:
        raise RuntimeError("Train veya val kümesi boş. Önce veri üretin: python -m src.generate_synthetic")
    logger.info("Train: %d, Val: %d görüntü (test/ood bu betikte kullanılmaz)", len(train_df), len(val_df))

    # --- 2) Veri yükleyiciler (dönüşümler dinamik, yalnızca bellekte) -------------
    train_ds = ForgeryDataset(train_df, build_transforms(True, args.crop_size), args.max_side)
    val_ds = ForgeryDataset(val_df, build_transforms(False), args.max_side)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              persistent_workers=args.num_workers > 0)

    # --- 3) Model, kayıp, optimizasyon -------------------------------------------
    weights = None if args.encoder_weights.lower() == "none" or args.init else args.encoder_weights
    model = DualStreamForensicNet(encoder_name=args.encoder, encoder_weights=weights).to(device)
    if args.init:
        state = torch.load(args.init, map_location=device, weights_only=False)["model_state"]
        model.load_state_dict(state)
        logger.info("Başlangıç ağırlıkları yüklendi: %s", args.init)
    criterion = ForensicLoss(focal_weight=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mode = "min" if args.monitor == "loss" else "max"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    stopper = EarlyStopping(args.patience, mode)
    tracker = MetricLogger(run_dir / "tensorboard")
    config: dict[str, Any] = {"encoder": args.encoder, "crop_size": args.crop_size, "max_side": args.max_side,
                              "sources": args.sources, "seed": args.seed}
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # --- 4) Eğitim döngüsü ---------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.epochs)
        start = time.time()
        metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        metrics.update(validate(model, val_ds, criterion, args.crop_size))
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics["time_s"] = time.time() - start
        tracker.log(epoch, metrics)

        monitored = metrics[f"val/{args.monitor}"]
        scheduler.step(monitored)
        save_checkpoint(run_dir / "last.pth", model, config, epoch, metrics)
        if stopper.step(monitored):
            save_checkpoint(run_dir / "best.pth", model, config, epoch, metrics)
            logger.info("Yeni en iyi model (val/%s=%.4f) → %s", args.monitor, monitored, run_dir / "best.pth")
        if stopper.should_stop:
            logger.info("Erken durdurma: %d epoch boyunca iyileşme yok.", args.patience)
            break
    logger.info("Eğitim bitti. En iyi val/%s = %.4f", args.monitor, stopper.best)


if __name__ == "__main__":
    main()
