"""
RGB + SRM çift akışlı sahtecilik segmentasyon modeli.

Mimari
------
::

    girdi (3, H, W) normalize RGB ──┬──────────────────────────────► RGB akışı (3 kanal)
                                    │                                       │
                                    └─► SRMNoiseExtractor (sabit) ─► gürültü akışı (3 kanal)
                                                                            │
                                    birleştirme (concat, 6 kanal) ◄─────────┘
                                                │
                       smp.Unet(efficientnet-b4, in_channels=6)
                         ├── kod çözücü ─► mask_logits (1, H, W)   piksel sahtecilik haritası
                         └── kodlayıcı en derin öznitelik ─► global max-pool ─► cls_logit (1)

* **RGB akışı** tipografi, font, hizalama ve renk tutarsızlıklarını taşır.
* **SRM akışı** içerikten bağımsız yüksek frekanslı gürültü kalıntısını taşır.
  Sahte bölge (düz boyama, farklı sıkıştırma geçmişi, difüzyon dolgusu) bu
  kalıntının istatistiğini bozar.
* İki akış erken birleştirme (early fusion) ile 6 kanallı tek bir kodlayıcıya
  verilir. ImageNet ağırlıklı ilk katman, smp tarafından 6 kanala genişletilir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import IMAGENET_MEAN, IMAGENET_STD

# ---------------------------------------------------------------------------
# SRM (Spatial Rich Model) filtreleri
# ---------------------------------------------------------------------------

#: Fridrich & Kodovsky (2012) SRM ailesinden RGB-N (Zhou ve ark., 2018) ile
#: yaygınlaşan üç 5x5 yüksek geçiren filtre. Hepsinin katsayı toplamı 0'dır; bu
#: nedenle düz (sabit renkli) bölgelerde çıktı 0 olur ve yalnızca komşu pikseller
#: arasındaki tahmin hatası, yani **gürültü kalıntısı** kalır.
SRM_KERNELS: tuple[np.ndarray, ...] = (
    # KB (3. dereceden "square 3x3"): 2. türev yaklaşımı, normalizasyon 1/4
    np.array([[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0,
    # KV ("square 5x5"): daha geniş bağlamlı kalıntı, normalizasyon 1/12
    np.array([[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
    # Yatay 2. dereceden fark: x[i-1] - 2x[i] + x[i+1], normalizasyon 1/2
    np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0,
)


class SRMNoiseExtractor(nn.Module):
    """
    Sabit (eğitilmeyen) SRM gürültü kalıntısı çıkarıcı.

    Hesap adımları:
      1. Normalize girdi piksel ölçeğine geri çevrilir: ``x_255 = (x·σ + μ)·255``.
         (Eşik değeri T piksel biriminde anlamlı olduğu için.)
      2. Her SRM çekirdeği k_i, üç renk kanalının ortalamasına uygulanır:
         ``r_i = k_i * mean_c(x_255)``. Bu, ağırlığı ``k_i / 3`` olan ve tüm giriş
         kanallarına yayılan bir Conv2d ile tek seferde hesaplanır.
      3. TLU (Truncated Linear Unit): ``r_i ← clip(r_i, −T, T) / T``.
         Kenar/metin sınırlarındaki büyük yanıtlar kırpılır; ağın dikkati içerikten
         çok gürültü istatistiğine yönelir. Çıktı [−1, 1] aralığındadır.

    Ağırlıklar ``register_buffer`` ile tutulur → optimizasyona girmez, ``state_dict``'e
    kaydedilir ve cihaz değişikliklerinde modelle birlikte taşınır.
    """

    def __init__(self, threshold: float = 3.0,
                 mean: tuple[float, ...] = IMAGENET_MEAN, std: tuple[float, ...] = IMAGENET_STD) -> None:
        super().__init__()
        kernels = torch.from_numpy(np.stack(SRM_KERNELS))                  # (3, 5, 5)
        weight = kernels.unsqueeze(1).repeat(1, 3, 1, 1) / 3.0             # (3 çıkış, 3 giriş, 5, 5)
        self.register_buffer("weight", weight)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixels = (x * self.std + self.mean) * 255.0
        residual = F.conv2d(pixels, self.weight.to(pixels.dtype), padding=2)
        return torch.clamp(residual, -self.threshold, self.threshold) / self.threshold


# ---------------------------------------------------------------------------
# Çift akışlı ağ
# ---------------------------------------------------------------------------

class DualStreamForensicNet(nn.Module):
    """
    RGB + SRM erken birleştirmeli U-Net; çift kafalı çıktı.

    Returns (forward):
        mask_logits: (B, 1, H, W) — piksel düzeyi sahtecilik logit'i.
        cls_logit:   (B,)        — belge/yama düzeyi sahtecilik logit'i.

    Not: Girdi yüksekliği ve genişliği 32'nin katı olmalıdır (U-Net kodlayıcısının
    5 kez yarıya inmesi). Kayan pencere çıkarımı bunu otomatik sağlar.
    """

    def __init__(self, encoder_name: str = "efficientnet-b4", encoder_weights: Optional[str] = "imagenet",
                 srm_threshold: float = 3.0, cls_dropout: float = 0.2) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.srm = SRMNoiseExtractor(threshold=srm_threshold)
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=6,
            classes=1,
            # Yardımcı sınıflandırma kafası: en derin öznitelik haritası → global
            # max-pool → dropout → Linear. Max-pool, küçük bir sahte bölgenin
            # ortalamada kaybolmasını önler.
            aux_params={"classes": 1, "pooling": "max", "dropout": cls_dropout},
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat([x, self.srm(x)], dim=1)  # (B, 6, H, W)
        mask_logits, cls_logit = self.net(fused)
        return mask_logits, cls_logit.squeeze(1)


# ---------------------------------------------------------------------------
# Kaydetme / yükleme
# ---------------------------------------------------------------------------

def save_checkpoint(path: str | Path, model: DualStreamForensicNet, config: dict[str, Any],
                    epoch: int, metrics: dict[str, float], extra: Optional[dict[str, Any]] = None) -> None:
    """
    Model ağırlıklarını ve yeniden kurulum için gerekli yapılandırmayı kaydeder.

    ``extra`` (ör. optimizer/scheduler/scaler durumları) eğitime kaldığı yerden devam
    etmek için kayda eklenir. Dosya önce geçici ada yazılıp sonra yerine taşınır;
    böylece kayıt sırasında oturum kesilirse önceki kontrol noktası bozulmaz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "config": config, "epoch": epoch, "metrics": metrics,
               **(extra or {})}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_model(path: str | Path, device: torch.device | str = "cpu"
               ) -> tuple[DualStreamForensicNet, dict[str, Any]]:
    """Kontrol noktasından modeli kurar (önceden eğitilmiş ağırlık indirmeden) ve eval moduna alır."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = DualStreamForensicNet(encoder_name=config.get("encoder", "efficientnet-b4"), encoder_weights=None)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), {**config, "epoch": ckpt.get("epoch"), "metrics": ckpt.get("metrics", {})}


# ---------------------------------------------------------------------------
# Kayan pencere çıkarımı
# ---------------------------------------------------------------------------

def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Son pencere kenara hizalanacak şekilde başlangıç konumları."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    return starts + [length - tile]


@torch.no_grad()
def sliding_window_predict(model: DualStreamForensicNet, image: torch.Tensor, tile: int = 512,
                           overlap: float = 0.25, batch_size: int = 4) -> tuple[np.ndarray, float]:
    """
    Tam çözünürlüklü belgeyi örtüşen pencerelerle işler.

    Eğitim yamalar (crop) üzerinde yapıldığı için çıkarım da aynı ölçekte, yeniden
    boyutlandırma yapmadan yapılır; böylece SRM gürültü istatistiği korunur.

    * Piksel haritası: örtüşen bölgelerde olasılıkların ortalaması.
    * Belge logit'i: pencerelerin ``cls_logit`` değerlerinin **maksimumu**. Eğitimde
      yama etiketi "yamada sahte piksel var mı?" olduğu için, belgenin herhangi bir
      penceresinde sahtecilik varsa belge sahtedir (çoklu örnek öğrenme mantığı).

    Args:
        image: (3, H, W) normalize tensör.
    Returns:
        (prob_map (H, W) float32 [0,1], doc_logit float)
    """
    device = next(model.parameters()).device
    _, h, w = image.shape
    ph, pw = max(h, tile), max(w, tile)
    padded = F.pad(image.unsqueeze(0), (0, pw - w, 0, ph - h), mode="reflect" if (pw - w < w and ph - h < h)
                   else "replicate")[0]
    stride = max(32, int(tile * (1 - overlap)))
    coords = [(y, x) for y in _tile_starts(ph, tile, stride) for x in _tile_starts(pw, tile, stride)]

    prob_sum = torch.zeros((ph, pw), device=device)
    count = torch.zeros((ph, pw), device=device)
    doc_logit = -float("inf")
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = torch.stack([padded[:, y:y + tile, x:x + tile] for y, x in chunk]).to(device)
        mask_logits, cls_logit = model(batch)
        probs = torch.sigmoid(mask_logits[:, 0].float())
        for (y, x), p in zip(chunk, probs):
            prob_sum[y:y + tile, x:x + tile] += p
            count[y:y + tile, x:x + tile] += 1
        doc_logit = max(doc_logit, float(cls_logit.max()))
    prob_map = (prob_sum / count.clamp_min(1))[:h, :w]
    return prob_map.cpu().numpy().astype(np.float32), doc_logit
