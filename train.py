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
     ``last.pth`` optimizer/scheduler/AMP ölçekleyici durumunu da içerir; ``--resume``
     ile kesilen eğitim (ör. Kaggle oturum sınırı) kaldığı epoch'tan sürer.

Aşamalı eğitim (``--stage``; açıkça verilen argümanlar ön ayarı ezer):
  * ``stageA``: DocTamper üzerinde temel eğitim (lr=3e-4). En iyi model ayrıca
    ``<run_dir>/stageA_best.pth`` olarak saklanır.
  * ``stageB``: ``stageA_best.pth`` yüklenir; Türkçe sentetik + SROIE2019 üzerinde
    lr=1e-5 ile ince ayar. SROIE2019 bu aşamada eğitime girdiği için OOD değildir.
  * ``stageC``: Türkçe transfer öğrenme. ``checkpoints/finetuned_multilingual.pth`` (Stage B)
    yüklenir; **yalnızca** TR/EN sentetik belgelerle (DocTamper ve SROIE hariç) lr=1e-5,
    AdamW (weight decay 1e-4), 12 epoch ince ayar. Model seçimi ve erken durdurma yalnızca
    Türkçe sentetik belgelerden oluşan validasyon kümesiyle (``--val-lang tr``) yapılır;
    böylece DocTamper ağırlıklı genel val skoru seçimi yönlendirmez. Düşük lr, Stage A/B'de
    öğrenilen DocTamper özniteliklerinin unutulmasını sınırlar.
  * ``stageE``: Yanlış alarm düzeltmesi. ``checkpoints/base_doctamper.pth`` (Stage A) yüklenir
    (Stage C/D'nin "alan konumu = sahte" ezberi taşınmaz); düzenleme izli sentetik v4 +
    gerçek belgeler ve onlardan ``src/forge_real.py`` ile üretilen sahteler (``data/8_real_forged``)
    ile dengeli örneklemeyle eğitilir.

Sınıf dengesi: sentetik veri 1 orijinal : 3 sahte olduğundan eğitim varsayılan olarak
``WeightedRandomSampler`` ile orijinal/sahte 50/50 örneklenir (``--no-balance`` ile kapatılır).
Dengesiz örneklemede belge kafası "her şey sahte" önceline kayar.

Tüm aşamalarda eğitim görüntüleri ``--jpeg-aug-p`` (varsayılan 0.5) olasılıkla Q ∈ [50, 100]
JPEG sıkıştırmasından geçer (``src/dataset.py::build_transforms``).

Ablasyon ve iyileştirme seçenekleri:
  * ``--streams rgb | srm | rgb+srm | rgb+srm+dct``: girdi akışları (``src/models.py``). ``--init`` farklı
    akışlı bir ağırlıktan uyumlu yüklenir (ör. RGB+SRM → +DCT: DCT katmanı sıfırla başlar).
  * ``--flip-aug-p``: kayıpsız çevirme / 90° döndürme artırması (varsayılan 0 = önceki davranış).
  * ``--encoder-lr-mult``: kodlayıcı için öğrenme oranı çarpanı (ayrımsal öğrenme oranı; ör. 0.1).
  * ``--jpeg-curriculum N``: ilk N epoch JPEG artırması yalnızca Q ∈ [90, 100], sonra [50, 100]
    (DocTamper/DTD müfredatı).
  * ``--holdout-issuers`` / ``--holdout-doc-types``: bir kurumu / belge tipini tamamen teste ayırır
    (görülmemiş şablona genelleme).
  * ``--monitor image_tpr_at_fpr02``: model seçimi uygulamadaki çalışma noktasına göre (%2 yanlış alarmda TPR).
  * Ana sonuçlar için farklı ``--seed`` değerleriyle (ör. 42, 43, 44) en az 3 tekrar yapılıp
    ortalama ± standart sapma raporlanmalıdır.

Kaggle'da ``/kaggle/working`` varsa varsayılan ``--run-dir`` odur.

Örnek:
    python train.py --sources synthetic --epochs 40 --batch-size 8
    python train.py --sources synthetic aiforgedoc --init checkpoints/base/best.pth --lr 1e-4
    python train.py --stage stageA --samples-per-epoch 20000
    python train.py --stage stageA --data-root /kaggle/input/.../doctamper --doctamper-max 30000  # LMDB doğrudan
    python train.py --stage stageA --resume /kaggle/working/last.pth   # kesintiden sonra
    python train.py --stage stageB
    python train.py --stage stageC                       # Stage B → TR sentetik ince ayar
    python train.py --stage stageC --init /kaggle/input/.../finetuned_multilingual.pth
    python train.py --stage stageE                       # Stage A → izli sentetik v4 + gerçek belgeler/sahteleri
    python train.py --stage stageE --streams rgb --run-dir checkpoints/abl_rgb       # ablasyon: yalnız RGB
    python train.py --stage stageE --streams rgb+srm+dct --flip-aug-p 0.5 --encoder-lr-mult 0.1 \
        --jpeg-curriculum 3 --monitor image_tpr_at_fpr02 --run-dir checkpoints/stageF  # iyileştirilmiş model
    python train.py --stage stageE --holdout-issuers garanti --run-dir checkpoints/holdout_garanti
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from src.dataset import (REAL_SOURCES, SOURCE_DIRS, ForgeryDataset, apply_holdout, assign_splits, build_manifest,
                         build_transforms, close_lmdb_envs, worker_init_fn)
from src.losses import ForensicLoss
from src.metrics import PixelMetrics, image_level_metrics
from src.models import STREAMS, DualStreamForensicNet, load_compatible_state, save_checkpoint, sliding_window_predict

logger = logging.getLogger("train")

KAGGLE_WORKING = Path("/kaggle/working")

#: ``--stage`` ön ayarları; komut satırında açıkça verilen değerler önceliklidir.
#: Göreli ``init`` yolu bulunamazsa ``run_dir`` altında aranır.
STAGE_PRESETS: dict[str, dict[str, Any]] = {
    "stageA": {"sources": ["doctamper"], "lr": 3e-4},
    "stageB": {"sources": ["synthetic", "sroie2019"], "lr": 1e-5, "init": "stageA_best.pth"},
    "stageC": {"sources": ["synthetic"], "lr": 1e-5, "weight_decay": 1e-4, "epochs": 12,
               "init": "checkpoints/finetuned_multilingual.pth", "val_lang": "tr", "run_dir": "checkpoints/stageC"},
    "stageE": {"sources": ["synthetic", "real_forged"], "lr": 1e-4, "weight_decay": 1e-4, "epochs": 15,
               "init": "checkpoints/base_doctamper.pth", "val_lang": "tr", "run_dir": "checkpoints/stageE"},
}
DEFAULTS: dict[str, Any] = {"sources": ["synthetic"], "lr": 3e-4, "init": None, "epochs": 50,
                            "weight_decay": 1e-4, "val_lang": None}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB+SRM çift akışlı sahtecilik modeli eğitimi")
    p.add_argument("--data-root", default="data")
    p.add_argument("--stage", default=None, choices=sorted(STAGE_PRESETS),
                   help="Aşamalı eğitim ön ayarı: stageA = DocTamper temel eğitim, "
                        "stageB = stageA_best.pth'den sentetik + SROIE ince ayar (lr=1e-5), "
                        "stageC = Stage B'den yalnızca TR/EN sentetik ince ayar, TR val ile model seçimi")
    p.add_argument("--sources", nargs="+", default=None,
                   choices=sorted(SOURCE_DIRS),
                   help="Eğitim/val'de kullanılacak kaynaklar; diğerleri yalnızca test/OOD olur. "
                        "Varsayılan: synthetic (veya --stage ön ayarı).")
    p.add_argument("--run-dir", default=None,
                   help="Çıktı klasörü. Varsayılan: Kaggle'da /kaggle/working, aksi hâlde checkpoints/run")
    p.add_argument("--init", default=None, help="İnce ayar için başlangıç kontrol noktası (.pth)")
    p.add_argument("--resume", default=None,
                   help="Kaldığı yerden devam: last.pth yolu (model + optimizer + scheduler + epoch). "
                        "Klasör verilirse içindeki last.pth kullanılır.")
    p.add_argument("--encoder", default="efficientnet-b4")
    p.add_argument("--streams", default="rgb+srm", choices=STREAMS,
                   help="Girdi akışları (ablasyon): rgb, srm, rgb+srm (varsayılan) veya rgb+srm+dct")
    p.add_argument("--encoder-weights", default="imagenet", help="'none' ile rastgele başlatma")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--max-side", type=int, default=1600, help="Uzun kenar sınırı (yalnızca küçültme)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=None, help="Varsayılan: 50 (stageC: 12)")
    p.add_argument("--lr", type=float, default=None, help="Varsayılan: 3e-4 (stageB/stageC: 1e-5)")
    p.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay (varsayılan 1e-4)")
    p.add_argument("--val-lang", default=None, choices=["tr", "en"],
                   help="Validasyonu (model seçimi + erken durdurma) yalnızca bu dildeki sentetik belgelerle yap "
                        "(stageC: tr)")
    p.add_argument("--jpeg-aug-p", type=float, default=0.5,
                   help="Eğitimde Q∈[50,100] JPEG sıkıştırma artırmasının olasılığı (0 = kapalı)")
    p.add_argument("--crop-nonempty-p", type=float, default=0.5,
                   help="Sahte belgede kırpmanın sahte bölgeyi içerme olasılığı (kalanı rastgele kırpma)")
    p.add_argument("--flip-aug-p", type=float, default=0.0,
                   help="Kayıpsız yatay/dikey çevirme ve 90° döndürme artırması olasılığı (0 = kapalı)")
    p.add_argument("--encoder-lr-mult", type=float, default=1.0,
                   help="Kodlayıcı öğrenme oranı çarpanı (ayrımsal LR; ör. 0.1 → kodlayıcı lr/10)")
    p.add_argument("--jpeg-curriculum", type=int, default=0,
                   help="İlk N epoch JPEG artırması Q∈[90,100], sonrasında Q∈[50,100] (0 = kapalı)")
    p.add_argument("--holdout-issuers", nargs="*", default=[],
                   help="Bu kurumların (real:<kurum>/...) tüm belgelerini test kümesine ayır")
    p.add_argument("--holdout-doc-types", nargs="*", default=[],
                   help="Bu belge tiplerinin (doc_type sütunu) tüm belgelerini test kümesine ayır")
    p.add_argument("--no-balance", action="store_true",
                   help="Orijinal/sahte 50/50 dengeli örneklemeyi kapat (veri setindeki oranla örnekle)")
    p.add_argument("--samples-per-epoch", type=int, default=20000,
                   help="Epoch başına rastgele örneklenen eğitim görüntüsü (her epoch yeni örneklem); "
                        "0 = tüm eğitim kümesi")
    p.add_argument("--doctamper-max", type=int, default=0,
                   help="DocTamper TrainingSet'ten seed'li sabit alt küme boyutu (train+val bundan ayrılır); "
                        "0 = tamamı. LMDB doğrudan okunduğu için diske yazma gerekmez.")
    p.add_argument("--val-samples", type=int, default=2000,
                   help="Her epoch kullanılan sabit val alt kümesi boyutu (tam çözünürlük validasyon "
                        "pahalıdır); 0 = tüm val kümesi. evaluate.py etkilenmez.")
    p.add_argument("--patience", type=int, default=8, help="Erken durdurma sabrı (epoch)")
    p.add_argument("--monitor", default="score",
                   choices=["score", "pixel_f1", "image_auc", "image_tpr_at_fpr02", "loss"],
                   help="score = (pixel_f1 + image_auc) / 2; image_tpr_at_fpr02 = %%2 yanlış alarmda yakalama oranı")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--mp-context", default="spawn", choices=["spawn", "forkserver", "fork"],
                   help="DataLoader worker başlatma yöntemi. spawn: LMDB/OpenCV durumu worker'lara miras "
                        "kalmaz (fork kilitlenmelerini önler); worker'lar epoch'lar arası kalıcıdır.")
    p.add_argument("--log-every", type=int, default=50, help="Kaç adımda bir ilerleme günlüğü yazılacağı")
    p.add_argument("--benchmark-loader", type=int, default=0,
                   help="> 0 ise modeli eğitmeden yalnızca bu kadar batch okuyup veri hızını ölçer ve çıkar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Karışık hassasiyeti kapat")
    return apply_stage_presets(p.parse_args())


def apply_stage_presets(args: argparse.Namespace) -> argparse.Namespace:
    """``--stage`` ön ayarlarını ve genel varsayılanları yalnızca boş bırakılan argümanlara uygular."""
    preset = STAGE_PRESETS.get(args.stage, {}) if args.stage else {}
    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, preset.get(key, default))
    if args.run_dir is None:
        args.run_dir = str(KAGGLE_WORKING if KAGGLE_WORKING.is_dir() else Path(preset.get("run_dir", "checkpoints/run")))
    if args.init and not Path(args.init).is_absolute() and not Path(args.init).exists():
        args.init = str(Path(args.run_dir) / args.init)
    if args.resume and Path(args.resume).is_dir():
        args.resume = str(Path(args.resume) / "last.pth")
    return args


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

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best, self.bad_epochs = state["best"], state["bad_epochs"]

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
    """Konsol + ``history.csv`` + opsiyonel TensorBoard günlükleyici.

    ``history.csv`` (her epoch bir satır) ``app.py``'deki model başarımı sekmesinde
    eğitim/validasyon eğrilerini çizmek için kullanılır. Eğitime devam edildiğinde
    (``--resume``) mevcut dosyanın sonuna eklenir.
    """

    def __init__(self, log_dir: Path, history_path: Optional[Path] = None) -> None:
        self.history_path = history_path
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(log_dir))
        except ImportError:
            self.writer = None
            logger.info("TensorBoard kurulu değil; yalnızca konsol günlüğü kullanılacak.")

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        logger.info("Epoch %3d | %s", epoch, " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if self.history_path is not None:
            import pandas as pd

            row = pd.DataFrame([{"epoch": epoch, **metrics}])
            write_header = not self.history_path.exists()
            row.to_csv(self.history_path, mode="a", header=write_header, index=False)
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)
            self.writer.flush()


def train_one_epoch(model: DualStreamForensicNet, loader: DataLoader, criterion: ForensicLoss,
                    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device,
                    use_amp: bool, log_every: int = 50) -> dict[str, float]:
    """Bir epoch eğitim; kayıp bileşenlerinin ortalamasını döndürür.

    ``log_every`` adımda bir hız ve veri bekleme oranı yazılır: ``veri_bekleme`` yüksekse
    (> %50) darboğaz DataLoader'dır (worker sayısı / G/Ç), düşükse GPU'dur.
    """
    model.train()
    totals: dict[str, float] = {}
    start = tick = time.time()
    wait = 0.0
    for step, batch in enumerate(loader, 1):
        wait += time.time() - tick
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
        if log_every and (step % log_every == 0 or step == 1):
            elapsed = time.time() - start
            logger.info("  adım %d/%d | loss=%.4f | %.1f görüntü/s | veri_bekleme=%%%d | kalan≈%.0f dk",
                        step, len(loader), totals["loss"] / step, step * images.size(0) / elapsed,
                        100 * wait / elapsed, elapsed / step * (len(loader) - step) / 60)
        tick = time.time()
    return {f"train/{k}": v / max(len(loader), 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model: DualStreamForensicNet, loader: DataLoader, criterion: ForensicLoss,
             crop_size: int, use_amp: bool = False) -> dict[str, float]:
    """
    Validasyon: her belge tam çözünürlükte kayan pencere ile işlenir (uygulamadaki
    çıkarımla birebir aynı). Kayıp, tam görüntü olasılık haritası üzerinden hesaplanır.
    Görüntüler ``batch_size=1`` DataLoader worker'larında okunur; ana süreç LMDB açmaz.
    """
    model.eval()
    pixel = PixelMetrics()
    doc_logits, labels, losses = [], [], []
    for batch in loader:
        sample = {k: v[0] for k, v in batch.items()}
        with torch.autocast(device_type=next(model.parameters()).device.type, enabled=use_amp):
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
    if np.isnan(metrics["image_tpr_at_fpr02"]):  # val'de tek sınıf: model seçimi piksel F1'e düşer
        metrics["image_tpr_at_fpr02"] = metrics["pixel_f1"]
    return {f"val/{k}": v for k, v in metrics.items()}


def benchmark_loader(loader: DataLoader, n_batches: int) -> None:
    """Modelsiz veri okuma hızı testi: worker başlangıç süresi ve sürekli görüntü/s."""
    logger.info("Veri yükleyici testi: %d batch okunacak (model yok)...", n_batches)
    start = time.time()
    first = None
    for step, batch in enumerate(loader, 1):
        if first is None:
            first = time.time() - start
            logger.info("  ilk batch %.1f s'de geldi (worker başlangıcı dahil) | %s", first,
                        tuple(batch["image"].shape))
        if step % 10 == 0:
            rate = (step - 1) * batch["image"].size(0) / max(time.time() - start - first, 1e-6)
            logger.info("  %d/%d batch | %.1f görüntü/s", step, n_batches, rate)
        if step >= n_batches:
            break
    logger.info("Veri yükleyici testi bitti: %.1f s", time.time() - start)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    logger.info("Aşama: %s | kaynaklar: %s | lr: %g | run_dir: %s | AMP: %s",
                args.stage or "-", args.sources, args.lr, run_dir, use_amp)

    resume_ckpt: Optional[dict[str, Any]] = None
    if args.resume:
        if not Path(args.resume).is_file():
            raise FileNotFoundError(f"--resume kontrol noktası bulunamadı: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer_state" not in resume_ckpt:
            raise ValueError(f"{args.resume} optimizer durumu içermiyor (eski sürüm ya da best.pth); "
                             "devam için last.pth kullanın veya --init ile ince ayar başlatın.")
        saved_stage = resume_ckpt.get("stage")
        if saved_stage != args.stage:
            raise ValueError(f"{args.resume} '{saved_stage}' aşamasına ait, istenen aşama '{args.stage}'. "
                             "Yeni aşamaya geçerken --resume yerine --init kullanın.")
    elif args.init and not Path(args.init).is_file():
        raise FileNotFoundError(f"Başlangıç kontrol noktası bulunamadı: {args.init}"
                                + (" (önce --stage stageA çalıştırın)" if args.stage == "stageB" else "")
                                + (" (Stage B ağırlığını --init ile verin)" if args.stage == "stageC" else ""))

    # --- 1) Sızıntısız bölümleme -------------------------------------------------
    # /kaggle/input salt okunurdur: önbellek yazılamıyorsa run_dir altına yazılır.
    cache_dir = Path(args.data_root) / "splits"
    if not os.access(args.data_root, os.W_OK):
        cache_dir = run_dir / "splits"
    manifest = build_manifest(args.data_root, args.sources, cache_dir=cache_dir,
                              doctamper_max=args.doctamper_max, seed=args.seed)
    if args.holdout_issuers or args.holdout_doc_types:
        manifest = apply_holdout(manifest, args.holdout_issuers, args.holdout_doc_types)
    splits = assign_splits(manifest, seed=args.seed)
    splits.to_csv(run_dir / "splits.csv", index=False)
    train_df, val_df = splits[splits["split"] == "train"], splits[splits["split"] == "val"]
    if args.val_lang:
        # Stage C: model seçimi yalnızca hedef dildeki sentetik val belgeleriyle (val_tr).
        # Bölümleme değişmez; diğer val satırları yalnızca seçime katılmaz (eğitime de girmez).
        # Gerçek belge kaynakları (orijinal + gerçek tabanlı sahte) yanlış alarm ölçümü için val'de kalır.
        lang = val_df["lang"] if "lang" in val_df else pd.Series(index=val_df.index, dtype=object)
        keep = (((val_df["source"] == "synthetic") & (lang == args.val_lang))
                | val_df["source"].isin(REAL_SOURCES))
        logger.info("Validasyon '%s' dilindeki sentetik + gerçek belgelerle: %d / %d görüntü (val_%s).",
                    args.val_lang, int(keep.sum()), len(val_df), args.val_lang)
        val_df = val_df[keep]
        val_df.to_csv(run_dir / f"val_{args.val_lang}.csv", index=False)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Train veya val kümesi boş. Önce veri üretin: python -m src.generate_synthetic")
    logger.info("Train: %d, Val: %d görüntü (test/ood bu betikte kullanılmaz)", len(train_df), len(val_df))

    # --- 2) Veri yükleyiciler (dönüşümler dinamik, yalnızca bellekte) -------------
    if 0 < args.val_samples < len(val_df):
        # Sabit (seed'li) alt küme: epoch'lar arası metrikler karşılaştırılabilir kalır.
        val_df = val_df.sample(n=args.val_samples, random_state=args.seed)
        logger.info("Validasyon her epoch %d görüntülük sabit alt kümeyle yapılacak.", len(val_df))
    align8 = "dct" in args.streams

    def train_transforms(jpeg_quality: tuple[int, int] = (50, 100)):
        return build_transforms(True, args.crop_size, args.jpeg_aug_p, args.crop_nonempty_p,
                                flip_p=args.flip_aug_p, align8=align8, jpeg_quality=jpeg_quality)

    curriculum = args.jpeg_curriculum > 0 and args.jpeg_aug_p > 0
    train_ds = ForgeryDataset(train_df, train_transforms((90, 100) if curriculum else (50, 100)), args.max_side)
    if align8:
        logger.info("DCT akışı: eğitim kırpmaları JPEG 8x8 ızgarasına hizalı.")
    val_ds = ForgeryDataset(val_df, build_transforms(False), args.max_side)
    sampler = None
    labels = train_ds.frame["label"].astype(int)
    if not args.no_balance and labels.nunique() == 2:
        # Her sınıf toplam ağırlığın yarısını alır → epoch'ta orijinal/sahte ≈ 50/50 (yerine koyarak).
        weights = labels.map(0.5 / labels.value_counts()).to_numpy()
        n_samples = args.samples_per_epoch if args.samples_per_epoch > 0 else len(train_ds)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), n_samples, replacement=True)
        logger.info("Dengeli örnekleme: epoch başına %d görüntü, orijinal/sahte ≈ 50/50 (veri setinde %d / %d).",
                    n_samples, int((labels == 0).sum()), int((labels == 1).sum()))
    elif 0 < args.samples_per_epoch < len(train_ds):
        # Her epoch eğitim kümesinden yinelemesiz yeni bir rastgele örneklem çekilir.
        sampler = RandomSampler(train_ds, replacement=False, num_samples=args.samples_per_epoch)
        logger.info("Epoch başına %d / %d eğitim görüntüsü örneklenecek.", args.samples_per_epoch, len(train_ds))
    # Worker'lar başlamadan önce ana süreçte açık LMDB ortamı kalmamalı (fork mirası → kilitlenme).
    close_lmdb_envs()
    workers = {}
    if args.num_workers > 0:
        workers = dict(num_workers=args.num_workers, multiprocessing_context=args.mp_context,
                       worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=4)
    def make_train_loader() -> DataLoader:
        return DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                          drop_last=True, pin_memory=device.type == "cuda", **workers)

    train_loader = make_train_loader()
    val_workers = {**workers, "num_workers": min(args.num_workers, 2), "prefetch_factor": 2} if workers else {}
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **val_workers)
    logger.info("DataLoader: %d worker (%s), batch=%d, %d adım/epoch",
                args.num_workers, args.mp_context if workers else "ana süreç", args.batch_size, len(train_loader))

    if args.benchmark_loader > 0:
        benchmark_loader(train_loader, args.benchmark_loader)
        return

    # --- 3) Model, kayıp, optimizasyon -------------------------------------------
    pretrained = bool(args.init) or resume_ckpt is not None
    weights = None if args.encoder_weights.lower() == "none" or pretrained else args.encoder_weights
    model = DualStreamForensicNet(encoder_name=args.encoder, encoder_weights=weights, streams=args.streams).to(device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
    elif args.init:
        state = torch.load(args.init, map_location=device, weights_only=False)["model_state"]
        # Farklı akışlı ağırlıklar da yüklenebilir (ablasyon, RGB+SRM → +DCT ince ayarı).
        load_compatible_state(model, state)
        logger.info("Başlangıç ağırlıkları yüklendi: %s (akışlar: %s)", args.init, args.streams)
    criterion = ForensicLoss(focal_weight=2.0)
    encoder_params = list(model.net.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    param_groups = [{"params": encoder_params, "lr": args.lr * args.encoder_lr_mult},
                    {"params": other_params, "lr": args.lr}]
    if args.encoder_lr_mult != 1.0:
        logger.info("Ayrımsal öğrenme oranı: kodlayıcı %g, kod çözücü/kafalar/DCT %g",
                    args.lr * args.encoder_lr_mult, args.lr)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    mode = "min" if args.monitor == "loss" else "max"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    stopper = EarlyStopping(args.patience, mode)
    tracker = MetricLogger(run_dir / "tensorboard", history_path=run_dir / "history.csv")
    config: dict[str, Any] = {"encoder": args.encoder, "crop_size": args.crop_size, "max_side": args.max_side,
                              "sources": args.sources, "seed": args.seed, "stage": args.stage,
                              "val_lang": args.val_lang, "jpeg_aug_p": args.jpeg_aug_p,
                              "crop_nonempty_p": args.crop_nonempty_p, "balance": not args.no_balance,
                              "streams": args.streams, "flip_aug_p": args.flip_aug_p,
                              "encoder_lr_mult": args.encoder_lr_mult, "jpeg_curriculum": args.jpeg_curriculum,
                              "holdout_issuers": args.holdout_issuers, "holdout_doc_types": args.holdout_doc_types}
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    start_epoch = 1
    if resume_ckpt is not None:
        if len(resume_ckpt["optimizer_state"]["param_groups"]) != len(optimizer.param_groups):
            # Ayrımsal LR öncesi (tek gruplu) bir last.pth'den devam: optimizer o yapıyla kurulur.
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=3)
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        scaler.load_state_dict(resume_ckpt["scaler_state"])
        stopper.load_state_dict(resume_ckpt["stopper_state"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        logger.info("Devam ediliyor: %s (epoch %d tamamlanmış, en iyi val/%s=%s)",
                    args.resume, start_epoch - 1, args.monitor, stopper.best)
        del resume_ckpt
        if stopper.should_stop:
            logger.info("Bu eğitim erken durdurma ile zaten bitmiş; yapılacak bir şey yok.")
            return

    def training_state() -> dict[str, Any]:
        return {"optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(), "stopper_state": stopper.state_dict(), "stage": args.stage}

    # --- 4) Eğitim döngüsü ---------------------------------------------------------
    if curriculum and start_epoch > args.jpeg_curriculum:
        train_ds.transform = train_transforms((50, 100))  # devam: müfredat zaten tamamlanmış
        train_loader = make_train_loader()
    for epoch in range(start_epoch, args.epochs + 1):
        if curriculum and epoch == args.jpeg_curriculum + 1:
            # Kalıcı worker'lar dönüşümün eski kopyasını tutar; yükleyici yeni dönüşümle yeniden kurulur.
            logger.info("JPEG müfredatı: artırma Q∈[90,100] → Q∈[50,100]")
            train_ds.transform = train_transforms((50, 100))
            del train_loader
            train_loader = make_train_loader()
        logger.info("Epoch %d/%d", epoch, args.epochs)
        start = time.time()
        metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp,
                                  args.log_every)
        logger.info("Validasyon (%d görüntü)...", len(val_ds))
        metrics.update(validate(model, val_loader, criterion, args.crop_size, use_amp))
        metrics["lr"] = optimizer.param_groups[-1]["lr"]  # kod çözücü / kafalar (kodlayıcı: × --encoder-lr-mult)
        metrics["time_s"] = time.time() - start
        tracker.log(epoch, metrics)

        monitored = metrics[f"val/{args.monitor}"]
        scheduler.step(monitored)
        if stopper.step(monitored):
            save_checkpoint(run_dir / "best.pth", model, config, epoch, metrics)
            if args.stage:
                shutil.copyfile(run_dir / "best.pth", run_dir / f"{args.stage}_best.pth")
            logger.info("Yeni en iyi model (val/%s=%.4f) → %s", args.monitor, monitored, run_dir / "best.pth")
        # last.pth en son yazılır: kesinti olursa best.pth ile erken durdurma durumu tutarlı kalır.
        save_checkpoint(run_dir / "last.pth", model, config, epoch, metrics, extra=training_state())
        if stopper.should_stop:
            logger.info("Erken durdurma: %d epoch boyunca iyileşme yok.", args.patience)
            break
    logger.info("Eğitim bitti. En iyi val/%s = %.4f", args.monitor, stopper.best)


if __name__ == "__main__":
    main()
