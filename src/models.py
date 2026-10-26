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

Akış seçimi (``streams``, ablasyon için)
---------------------------------------
* ``rgb``            : yalnızca RGB (3 kanal)
* ``srm``            : yalnızca SRM gürültü kalıntısı (3 kanal)
* ``rgb+srm``        : varsayılan (6 kanal, yukarıdaki mimari)
* ``rgb+srm+dct``    : + **DCT (JPEG) akışı** (CAT-Net / DTD yaklaşımı). Y kanalının 8×8 blok DCT
  katsayıları küçük bir kodlayıcıdan geçirilip kodlayıcının 1/8 çözünürlüklü özniteliğine **eklenir**
  (orta füzyon). Ekleme katmanı sıfırla başlatıldığı için mevcut bir RGB+SRM ağırlığından ince ayar,
  ilk adımda tam olarak aynı çıktıyı verir; DCT katkısı eğitimle öğrenilir.

  DCT izleri yalnızca JPEG blok ızgarası korunduğunda anlamlıdır: eğitimde kırpmalar 8'e hizalanır
  (``src/dataset.py``), çıkarımda pencere başlangıçları 8'in katıdır. Uzun kenarı ``max_side``'dan
  büyük olup küçültülen görüntülerde ve PNG ekran görüntülerinde bu akış katkı vermez (beklenen durum).
"""

from __future__ import annotations

import inspect
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

#: Desteklenen girdi akışı kombinasyonları.
STREAMS: tuple[str, ...] = ("rgb", "srm", "rgb+srm", "rgb+srm+dct")
#: DCT özniteliklerinin eklendiği kodlayıcı seviyesi (indeks 3 = 1/8 çözünürlük = 8x8 JPEG bloğu).
DCT_FUSION_LEVEL = 3

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
# DCT (JPEG) akışı
# ---------------------------------------------------------------------------

def _dct_basis() -> torch.Tensor:
    """8x8 tip-II DCT taban fonksiyonları (64, 1, 8, 8); ``conv2d(stride=8)`` blok DCT'si verir."""
    basis = torch.zeros(64, 1, 8, 8)
    for u in range(8):
        for v in range(8):
            cu = math.sqrt(1 / 8) if u == 0 else math.sqrt(2 / 8)
            cv_ = math.sqrt(1 / 8) if v == 0 else math.sqrt(2 / 8)
            for x in range(8):
                for y in range(8):
                    basis[u * 8 + v, 0, x, y] = (cu * cv_ * math.cos((2 * x + 1) * u * math.pi / 16)
                                                 * math.cos((2 * y + 1) * v * math.pi / 16))
    return basis


class DCTStream(nn.Module):
    """
    JPEG sıkıştırma izi öznitelikleri (1/8 çözünürlük).

    1. Normalize girdi piksel ölçeğine çevrilir ve parlaklık (Y = 0.299R + 0.587G + 0.114B) alınır.
    2. 8x8 bloklarda DCT: ``conv2d(Y − 128, taban, stride=8)`` → (B, 64, H/8, W/8). JPEG nicemlemesi bu
       katsayıların belirli değerlerde kümelenmesine yol açar; farklı sıkıştırma geçmişli (yapıştırılmış /
       yeniden yazılmış) bölgede bu dağılım bozulur (çift sıkıştırma izi).
    3. Katsayılar [−clip, clip]'e kırpılıp ölçeklenir (DC ve büyük AC terimleri içeriği taşır; nicemleme izi
       küçük katsayılardadır) ve iki 3x3 konvolüsyonla ``out_channels`` özniteliğe dönüştürülür.
    """

    def __init__(self, out_channels: int = 64, clip: float = 20.0,
                 mean: tuple[float, ...] = IMAGENET_MEAN, std: tuple[float, ...] = IMAGENET_STD) -> None:
        super().__init__()
        self.register_buffer("basis", _dct_basis())
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.clip = clip
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixels = (x.float() * self.std + self.mean) * 255.0
        y = (pixels * self.luma).sum(dim=1, keepdim=True)
        coef = F.conv2d(y - 128.0, self.basis, stride=8)
        return self.encoder((coef.clamp(-self.clip, self.clip) / self.clip).to(x.dtype))


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
                 srm_threshold: float = 3.0, cls_dropout: float = 0.2, streams: str = "rgb+srm") -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        if streams not in STREAMS:
            raise ValueError(f"Bilinmeyen akış kombinasyonu: {streams} (seçenekler: {', '.join(STREAMS)})")
        self.streams = streams
        self.srm = SRMNoiseExtractor(threshold=srm_threshold)
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3 if streams in ("rgb", "srm") else 6,
            classes=1,
            # Yardımcı sınıflandırma kafası: en derin öznitelik haritası → global
            # max-pool → dropout → Linear. Max-pool, küçük bir sahte bölgenin
            # ortalamada kaybolmasını önler.
            aux_params={"classes": 1, "pooling": "max", "dropout": cls_dropout},
        )
        self.dct: Optional[DCTStream] = None
        if "dct" in streams:
            self.dct = DCTStream(out_channels=64)
            self.dct_proj = nn.Conv2d(64, self.net.encoder.out_channels[DCT_FUSION_LEVEL], kernel_size=1)
            # Sıfır başlatma: eğitim başında DCT katkısı 0 → mevcut RGB+SRM ağırlıklarıyla birebir aynı çıktı.
            nn.init.zeros_(self.dct_proj.weight)
            nn.init.zeros_(self.dct_proj.bias)
            params = inspect.signature(self.net.decoder.forward).parameters.values()
            # smp >= 0.5: decoder(features); daha eski sürümler: decoder(*features)
            self._decoder_varargs = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)

    def _inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.streams == "rgb":
            return x
        if self.streams == "srm":
            return self.srm(x)
        return torch.cat([x, self.srm(x)], dim=1)  # (B, 6, H, W)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._inputs(x)
        if self.dct is None:
            mask_logits, cls_logit = self.net(inputs)
            return mask_logits, cls_logit.squeeze(1)
        features = list(self.net.encoder(inputs))
        features[DCT_FUSION_LEVEL] = features[DCT_FUSION_LEVEL] + self.dct_proj(self.dct(x))
        decoded = self.net.decoder(*features) if self._decoder_varargs else self.net.decoder(features)
        mask_logits = self.net.segmentation_head(decoded)
        cls_logit = self.net.classification_head(features[-1])
        return mask_logits, cls_logit.squeeze(1)


def load_compatible_state(model: DualStreamForensicNet, state: dict[str, torch.Tensor]) -> None:
    """Farklı akış kombinasyonuyla kaydedilmiş bir ağırlığı yükler (ablasyon / DCT ince ayarı için).

    * Eksik anahtarlar (ör. yeni DCT akışı) rastgele/sıfır başlangıçta kalır.
    * Giriş kanal sayısı farklı ilk konvolüsyon (6 → 3) dilimlenir: ``rgb`` için ilk 3, ``srm`` için son
      3 giriş kanalı alınır ve 2 ile çarpılır (kaldırılan kanalların katkısını kabaca telafi eder).
    """
    own = model.state_dict()
    adapted, skipped = {}, []
    for key, value in state.items():
        if key not in own:
            skipped.append(key)
            continue
        target = own[key]
        if value.shape == target.shape:
            adapted[key] = value
        elif (value.dim() == 4 and value.shape[1] == 6 and target.shape[1] == 3
              and value.shape[0] == target.shape[0]):
            part = value[:, :3] if model.streams == "rgb" else value[:, 3:]
            adapted[key] = part * 2.0
        else:
            skipped.append(key)
    missing = [k for k in own if k not in adapted]
    model.load_state_dict(adapted, strict=False)
    if missing or skipped:
        logger.info("Uyumlu yükleme: %d anahtar yeni başlatıldı (%s...), %d anahtar atlandı.",
                    len(missing), ", ".join(missing[:3]), len(skipped))


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
    model = DualStreamForensicNet(encoder_name=config.get("encoder", "efficientnet-b4"), encoder_weights=None,
                                  streams=config.get("streams", "rgb+srm"))
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
                           overlap: float = 0.25, batch_size: int = 4, temperature: float = 1.0,
                           top_k_ratio: float = 0.10, return_windows: bool = False):
    """
    Tam çözünürlüklü belgeyi örtüşen pencerelerle işler.

    Eğitim yamalar (crop) üzerinde yapıldığı için çıkarım da aynı ölçekte, yeniden
    boyutlandırma yapmadan yapılır; böylece SRM gürültü istatistiği korunur.

    * Piksel haritası: örtüşen bölgelerde olasılıkların ortalaması.
    * Belge logit'i: en yüksek ``cls_logit``'e sahip pencerelerin ``top_k_ratio``
      kadarının (en az 1) **ortalaması** (Top-K Average). Eğitimde yama etiketi
      "yamada sahte piksel var mı?" olduğu için belge skoru en şüpheli pencerelerden
      gelir (çoklu örnek öğrenme mantığı); tek bir pencerenin aykırı logit'i yerine
      en şüpheli %10'un ortalaması alınarak gerçek belgelerdeki yanlış alarmlar azaltılır.
      ``k = max(1, ceil(top_k_ratio · pencere_sayısı))``; tek pencereli (≤ tile) belgede
      ve ``top_k_ratio → 0`` için eski maksimum davranışına eşittir.

    Args:
        image: (3, H, W) normalize tensör.
        temperature: Piksel logit'lerine sigmoid'den önce uygulanan sıcaklık (``logit / T``).
            ``doc_logit`` ham döner; kalibrasyonu çağıran taraf uygular.
        top_k_ratio: Belge logit'ine ortalaması alınan en yüksek pencere oranı (0, 1].
        return_windows: True ise pencere ``cls_logit``'leri de döndürülür (havuzlama ablasyonu için).
    Returns:
        (prob_map (H, W) float32 [0,1], doc_logit float) — ``return_windows`` ile üçüncü öğe
        olarak pencere logit'leri (float32 dizi).
    """
    device = next(model.parameters()).device
    _, h, w = image.shape
    ph, pw = max(h, tile), max(w, tile)
    if getattr(model, "dct", None) is not None:
        # DCT akışı: tüm pencere başlangıçları 8'in katı olsun (JPEG blok ızgarası korunur).
        ph, pw = math.ceil(ph / 8) * 8, math.ceil(pw / 8) * 8
    padded = F.pad(image.unsqueeze(0), (0, pw - w, 0, ph - h), mode="reflect" if (pw - w < w and ph - h < h)
                   else "replicate")[0]
    stride = max(32, int(tile * (1 - overlap)))
    coords = [(y, x) for y in _tile_starts(ph, tile, stride) for x in _tile_starts(pw, tile, stride)]

    prob_sum = torch.zeros((ph, pw), device=device)
    count = torch.zeros((ph, pw), device=device)
    window_logits = []
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = torch.stack([padded[:, y:y + tile, x:x + tile] for y, x in chunk]).to(device)
        mask_logits, cls_logit = model(batch)
        probs = torch.sigmoid(mask_logits[:, 0].float() / temperature)
        for (y, x), p in zip(chunk, probs):
            prob_sum[y:y + tile, x:x + tile] += p
            count[y:y + tile, x:x + tile] += 1
        window_logits.append(cls_logit.float())
    window_logits = torch.cat(window_logits)
    k = max(1, math.ceil(top_k_ratio * window_logits.numel()))
    doc_logit = float(window_logits.topk(k).values.mean())
    prob_map = (prob_sum / count.clamp_min(1))[:h, :w].cpu().numpy().astype(np.float32)
    if return_windows:
        return prob_map, doc_logit, window_logits.cpu().numpy().astype(np.float32)
    return prob_map, doc_logit
