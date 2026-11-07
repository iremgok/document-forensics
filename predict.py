"""
Tek belge için sahtecilik çıkarımı ve 3 panelli görselleştirme.

Çalıştırma:
    python predict.py
    python predict.py --mode resize --image belge.jpg --checkpoint checkpoints/finetuned_multilingual_stageB.pth
    python predict.py --image sahte.png --gt-mask maske.png --gradcam   # + hata haritası ve Grad-CAM

Akış:
  1. Belge okunur ve ImageNet normalizasyonu uygulanır.
     * ``--mode sliding`` (varsayılan): ``app.py`` ile aynı ön işlem — uzun kenar
       ``max_side`` ile sınırlanır, yeniden boyutlandırma yapılmadan örtüşen 512'lik
       pencerelerle taranır (SRM gürültü istatistiği korunur).
     * ``--mode resize``: belge 512x512'ye ölçeklenip tek geçişte işlenir.
  2. ``DualStreamForensicNet`` piksel logit'i ve belge logit'i üretir.
  3. Logit'ler sıcaklık (T) ile kalibre edilir: ``p = sigmoid(logit / T)``.
  4. Olasılık haritası orijinal boyuta geri ölçeklenir ve
     (orijinal | jet olasılık haritası | %60 orijinal + %40 ısı haritası)
     figürü ``--output`` yoluna kaydedilir.
  5. (opsiyonel) ``--gt-mask`` verilirse TP/FP/FN **hata haritası** (yeşil/kırmızı/mavi), ``--gradcam``
     ile sınıflandırma kafası için **Grad-CAM** paneli eklenir (bkz. ``src/explain.py``).

Kalibrasyon: önce checkpoint'e özel dosya (``best.pth`` / ``last.pth`` için klasördeki ``calibration.json``,
ör. ``checkpoints/stageE/calibration.json``; diğer adlar için ``<ad>.calibration.json``; ``evaluate.py``
tarafından yazılır), sonra checkpoint klasöründeki ``calibration.json`` → ``checkpoints/calibration.json`` →
``checkpoints/synthetic/calibration.json`` aranır; hiçbiri yoksa varsayılan değerlerle
``checkpoints/calibration.json`` oluşturulur. ``--calibration`` ile açıkça verilebilir.

Şema (``Calibration``): belge skoru ve piksel haritası **ayrı** kalibre edilir — ``doc_temperature`` /
``doc_threshold`` belge koşulu, ``pixel_temperature`` / ``pixel_threshold`` bölge koşulu için; ayrıca
``min_region_fraction``, ``target_fpr``, ``val_fpr``, ``val_tpr``, ``val_images``, ``checkpoint``. Eski şema
(tek ``temperature`` ve ``optimal_threshold`` / ``threshold``) okunmaya devam eder: iki taraf da aynı T ve
eşiği alır (eski davranış).

Bir dosya yalnızca ``evaluate.py``'nin val kümesinde yazdığı ``val_images`` alanını taşıyorsa "kalibre"
sayılır; genel ``checkpoints/calibration.json`` (T=1, eşik 0.60, elle girilmiş) bu yüzden **kalibre
edilmemiş** olarak işaretlenir (``Calibration.calibrated``).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib

matplotlib.use("Agg")  # Ekransız ortamlarda (sunucu/Kaggle) da çalışsın
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, limit_long_side, read_image_rgb, read_mask
from src.explain import cam_overlay, error_counts, error_map, grad_cam
from src.models import DualStreamForensicNet, load_model, sliding_window_predict

logger = logging.getLogger("predict")

INPUT_SIZE = 512
DEFAULT_IMAGES = ("test_document.jpeg", "test_document.jpg", "test_document.png")
IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp")
CALIBRATION_PATHS = (Path("checkpoints/calibration.json"), Path("checkpoints/synthetic/calibration.json"))
#: Kalibrasyon dosyası olmayan modeller için nötr varsayılan (kalibrasyonsuz logit, eşik 0.5). Eski 6.389 / 0.35
#: çifti Stage B'nin bir ara benchmark'ına aitti ve başka modellere uygulandığında yanıltıcıydı; her modelin
#: kalibrasyonu ``evaluate.py`` ile kendi val kümesinde üretilmelidir.
DEFAULT_CALIBRATION = {"temperature": 1.0, "optimal_threshold": 0.5}
CLEAN_PIXEL_LIMIT = 0.05  # Bu değerin altındaki maks. piksel olasılığı "sahte bölge yok" sayılır
DOC_WEIGHT = 0.7          # Temiz haritada belge skorunun ağırlığı (kalanı maks. piksel olasılığı)
MIN_REGION_FRACTION = 0.0005  # Piksel koşulu için asgari bağlantılı bölge alanı (sayfanın %0.05'i)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DualStreamForensicNet ile belge sahtecilik tahmini")
    p.add_argument("--image", default=None,
                   help=f"Girdi belge görseli (varsayılan: {' / '.join(DEFAULT_IMAGES)} içinden ilk bulunan)")
    p.add_argument("--checkpoint", default="checkpoints/stageE/best.pth", help="Model ağırlıkları")
    p.add_argument("--mode", choices=("sliding", "resize"), default="sliding",
                   help="sliding: orijinal çözünürlükte kayan pencere; resize: 512x512 tek geçiş")
    p.add_argument("--output", default="prediction_result.png", help="Kaydedilecek figür")
    p.add_argument("--calibration", default=None,
                   help="Kalibrasyon JSON yolu (verilmezse otomatik aranır / oluşturulur)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Belge eşiği (verilirse kalibrasyon dosyasındakini ezer)")
    p.add_argument("--pixel-threshold", type=float, default=None,
                   help="Piksel (bölge) eşiği (verilirse kalibrasyon dosyasındakini ezer)")
    p.add_argument("--gt-mask", default=None,
                   help="Gerçek sahtecilik maskesi (varsa TP/FP/FN hata haritası paneli eklenir)")
    p.add_argument("--gradcam", action="store_true",
                   help="Sınıflandırma kafası (belge skoru) için Grad-CAM paneli ekle")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Girdi / kalibrasyon
# ---------------------------------------------------------------------------

def resolve_image_path(path: str | Path | None) -> Path:
    """Yol verilmezse varsayılan test görsellerinden ilk bulunanı; verilip yoksa aynı adlı
    farklı uzantılı dosyayı döndürür (ör. .png yerine .jpeg)."""
    if path is None:
        for name in DEFAULT_IMAGES:
            if Path(name).exists():
                return Path(name)
        raise FileNotFoundError(f"Varsayılan görsellerin hiçbiri bulunamadı: {', '.join(DEFAULT_IMAGES)}")
    path = Path(path)
    if path.exists():
        return path
    for suffix in IMAGE_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            logger.warning("%s bulunamadı, %s kullanılıyor", path, candidate)
            return candidate
    raise FileNotFoundError(f"Görsel bulunamadı: {path}")


@dataclass
class Calibration:
    """Belge skoru ve piksel haritası için ayrı kalibrasyon + karar eşikleri (bkz. modül açıklaması)."""

    doc_temperature: float = 1.0
    doc_threshold: float = 0.5
    pixel_temperature: float = 1.0
    pixel_threshold: float = 0.5
    min_region_fraction: float = MIN_REGION_FRACTION
    target_fpr: Optional[float] = None
    val_fpr: Optional[float] = None
    val_tpr: Optional[float] = None
    val_images: Optional[int] = None
    checkpoint: Optional[str] = None
    #: Okunan dosya (dosyaya yazılmaz).
    path: Optional[Path] = field(default=None, compare=False)
    #: Eski tek-T şemasından mı okundu (dosyaya yazılmaz).
    legacy: bool = field(default=False, compare=False)

    @property
    def calibrated(self) -> bool:
        """Val kümesinde ``evaluate.py`` ile öğrenilmiş mi? (Elle girilmiş genel dosya → False.)"""
        return self.val_images is not None and int(self.val_images) > 0

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Optional[Path] = None) -> "Calibration":
        """Yeni ya da eski (``temperature`` + ``optimal_threshold``) şemadan okur."""
        def num(key: str, default: Optional[float]) -> Optional[float]:
            value = data.get(key, default)
            return None if value is None else float(value)

        legacy = "doc_temperature" not in data
        if legacy:
            temperature = num("temperature", DEFAULT_CALIBRATION["temperature"])
            threshold = num("optimal_threshold", data.get("threshold", DEFAULT_CALIBRATION["optimal_threshold"]))
            doc_t, doc_thr, pix_t, pix_thr = temperature, threshold, temperature, threshold
        else:
            doc_t, doc_thr = num("doc_temperature", 1.0), num("doc_threshold", 0.5)
            pix_t, pix_thr = num("pixel_temperature", 1.0), num("pixel_threshold", doc_thr)
        for name, t in (("belge", doc_t), ("piksel", pix_t)):
            if t <= 0:
                raise ValueError(f"Geçersiz {name} sıcaklığı ({t}): {path}")
        images = data.get("val_images")
        return cls(doc_temperature=doc_t, doc_threshold=doc_thr, pixel_temperature=pix_t, pixel_threshold=pix_thr,
                   min_region_fraction=num("min_region_fraction", MIN_REGION_FRACTION),
                   target_fpr=num("target_fpr", None), val_fpr=num("val_fpr", None), val_tpr=num("val_tpr", None),
                   val_images=None if images is None else int(images), checkpoint=data.get("checkpoint"),
                   path=path, legacy=legacy)

    def to_dict(self) -> dict[str, Any]:
        """Yeni şema + eski okuyucular için ``temperature`` / ``optimal_threshold`` (= belge tarafı)."""
        out = {k: v for k, v in asdict(self).items() if k not in ("path", "legacy")}
        out.update(temperature=self.doc_temperature, optimal_threshold=self.doc_threshold)
        return out

    def summary(self) -> str:
        text = (f"belge T={self.doc_temperature:.3f} / eşik {self.doc_threshold:.2f} · "
                f"piksel T={self.pixel_temperature:.3f} / eşik {self.pixel_threshold:.2f}")
        return text + ("" if self.calibrated else " · kalibre edilmemiş")


def calibration_path_for(checkpoint: str | Path) -> Path:
    """``evaluate.py``'nin bir checkpoint için kalibrasyonu yazdığı yol.

    ``best.pth`` / ``last.pth`` → klasördeki ``calibration.json`` (her çalışma kendi klasöründe); diğer
    adlar (ör. ``checkpoints/finetuned_multilingual_stageB.pth``) → ``<ad>.calibration.json``. Böylece
    ``checkpoints/`` kökündeki modeller genel ``checkpoints/calibration.json``'ı ezmez.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.stem in ("best", "last"):
        return checkpoint.parent / "calibration.json"
    return checkpoint.with_name(f"{checkpoint.stem}.calibration.json")


def find_calibration_file(checkpoint: str | Path | None = None) -> Optional[Path]:
    """``load_calibration_info``'nun okuyacağı dosya (yoksa None; hiçbir dosya oluşturmaz)."""
    own: tuple[Path, ...] = ()
    if checkpoint:
        own = tuple(dict.fromkeys((calibration_path_for(checkpoint), Path(checkpoint).parent / "calibration.json")))
    return next((p for p in own + CALIBRATION_PATHS if p.exists()), None)


def load_calibration_info(explicit: str | None = None, checkpoint: str | Path | None = None) -> Calibration:
    """Kalibrasyonu ``Calibration`` olarak okur; dosya yoksa varsayılanlarla oluşturur.

    Arama sırası: ``explicit`` → ``calibration_path_for(checkpoint)`` → checkpoint klasöründeki
    ``calibration.json`` → ``CALIBRATION_PATHS`` (``find_calibration_file``).
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Kalibrasyon dosyası bulunamadı: {path}")
    else:
        path = find_calibration_file(checkpoint)
        if path is None:
            path = CALIBRATION_PATHS[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(DEFAULT_CALIBRATION, indent=2), encoding="utf-8")
            logger.warning("Kalibrasyon dosyası bulunamadı; nötr varsayılanlarla (T=1, eşik=0.5) oluşturuldu: %s. "
                           "Modelin kendi kalibrasyonu için: python evaluate.py --checkpoint ... --splits ...", path)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return Calibration.from_dict(data, path)


def load_calibration(explicit: str | None = None, checkpoint: str | Path | None = None
                     ) -> tuple[float, float, Path]:
    """Geriye dönük uyumluluk: (belge sıcaklığı, belge eşiği, okunan dosya).

    Piksel tarafının ayrı değerleri için ``load_calibration_info`` kullanın.
    """
    cal = load_calibration_info(explicit, checkpoint)
    return cal.doc_temperature, cal.doc_threshold, cal.path


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) → normalize (3, H, W) tensör."""
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    return (tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Çıkarım
# ---------------------------------------------------------------------------

def predict(model: DualStreamForensicNet, image: np.ndarray, config: dict[str, Any], temperature: float,
            device: str, mode: str = "sliding", pixel_temperature: Optional[float] = None) -> tuple[np.ndarray, float]:
    """
    ``temperature`` belge logit'ine, ``pixel_temperature`` (verilmezse ``temperature``; eski tek-T
    davranışı) piksel logit'lerine uygulanır.

    Returns:
        (orijinal boyutta kalibre olasılık haritası float32 [0,1], kalibre belge olasılığı)
    """
    pixel_t = temperature if pixel_temperature is None else pixel_temperature
    prob_map, doc_logit = predict_raw(model, image, config, pixel_t, device, mode)
    return prob_map, float(torch.sigmoid(torch.tensor(doc_logit / temperature)))


@torch.no_grad()
def predict_raw(model: DualStreamForensicNet, image: np.ndarray, config: dict[str, Any], temperature: float,
                device: str, mode: str = "sliding") -> tuple[np.ndarray, float]:
    """``predict`` ile aynı; belge skorunu kalibre edilmemiş ham logit olarak döndürür
    (değerlendirmede farklı sıcaklıkların karşılaştırılabilmesi için). ``temperature`` yalnızca piksel
    haritasına uygulanır (piksel sıcaklığı).

    Returns:
        (orijinal boyutta kalibre olasılık haritası float32 [0,1], ham belge logit'i)
    """
    if mode == "sliding":
        # app.py.run_detector ile aynı: eğitimdeki uzun kenar sınırı, ardından ölçeklemesiz tarama
        small, _ = limit_long_side(image, None, config.get("max_side", 1600))
        prob, doc_logit = sliding_window_predict(model, to_tensor(small), tile=config.get("crop_size", INPUT_SIZE),
                                                 temperature=temperature)
    else:
        resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        mask_logits, cls_logit = model(to_tensor(resized).unsqueeze(0).to(device))
        prob = torch.sigmoid(mask_logits[0, 0].float() / temperature).cpu().numpy()
        doc_logit = float(cls_logit[0])
    h, w = image.shape[:2]
    mask_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask_resized, 0.0, 1.0).astype(np.float32), float(doc_logit)


def fuse_doc_score(doc_prob: float, max_pixel_prob: float) -> float:
    """
    Piksel haritasında sahte bölge izi yoksa belge skorunu baskılar.

    Sınıflandırma kafası, segmentasyon haritası tamamen temiz olduğu hâlde yüksek
    logit üretebilir (ör. temiz bir fişte %85). Maksimum piksel olasılığı
    ``CLEAN_PIXEL_LIMIT``'in altındaysa iki kanıt ağırlıklı ortalamayla birleştirilir:

        doc_prob = 0.7 · doc_prob + 0.3 · max_pixel_prob

    Bu durumda skor en fazla 0.7 + 0.3 · 0.05 = %71.5 olabilir. Piksel kanıtı varsa
    belge skoru değiştirilmez.
    """
    if max_pixel_prob < CLEAN_PIXEL_LIMIT:
        return DOC_WEIGHT * doc_prob + (1.0 - DOC_WEIGHT) * max_pixel_prob
    return doc_prob


def largest_region_fraction(prob_map: np.ndarray, threshold: float) -> float:
    """Eşiği geçen en büyük bağlantılı bölgenin sayfa alanına oranı (bölge yoksa 0).

    Piksel karar koşulu ``largest_region_fraction(...) >= MIN_REGION_FRACTION`` şeklindedir; böylece
    dijital belgelerdeki kenar yumuşatma (anti-aliasing) kaynaklı tekil pikseller yanlış alarm üretmez.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((prob_map >= threshold).astype(np.uint8))
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / prob_map.size if num_labels > 1 else 0.0


def decide_forged(doc_prob: float, max_pixel: float, region_fraction: float, threshold: float,
                  min_region_fraction: float = MIN_REGION_FRACTION) -> bool:
    """``is_forged`` kuralı, önceden hesaplanmış özetlerle (eşik taraması için; bkz. ``evaluate.py``).

    ``threshold`` belge eşiğidir; ``region_fraction`` = ``largest_region_fraction(prob_map, piksel eşiği)``
    ve ``max_pixel`` piksel sıcaklığıyla kalibre haritanın maksimumudur.
    """
    hybrid_condition = doc_prob >= threshold and max_pixel >= CLEAN_PIXEL_LIMIT
    pixel_condition = region_fraction >= min_region_fraction
    return hybrid_condition or pixel_condition


def is_forged(doc_prob: float, prob_map: np.ndarray, threshold: float, pixel_threshold: Optional[float] = None,
              min_region_fraction: float = MIN_REGION_FRACTION) -> bool:
    """
    Sahtecilik kararı — iki koşuldan biri yeterlidir:

    * **Hibrit:** belge skoru belge eşiğini (``threshold`` = ``doc_threshold``) geçer VE piksel haritasında
      en az hafif bir iz vardır (maks. piksel olasılığı ≥ ``CLEAN_PIXEL_LIMIT``). Piksel haritası tamamen
      temizken yalnızca sınıflandırma kafasına dayanan yanlış alarmlar böylece engellenir.
    * **Piksel:** piksel eşiğini (``pixel_threshold``; verilmezse ``threshold`` — eski tek eşik davranışı)
      geçen bağlantılı bir bölgenin alanı sayfanın en az ``min_region_fraction`` kadarıdır (belge skoru
      düşük kalsa bile). Dijital belgelerdeki kenar yumuşatma (anti-aliasing) kaynaklı tekil piksellerin
      yanlış alarm üretmesi böylece engellenir.

    Eşikler, ``evaluate.py`` tarafından val kümesinde hedef yanlış alarm oranına göre birlikte seçilir.
    """
    pixel_threshold = threshold if pixel_threshold is None else pixel_threshold
    return decide_forged(doc_prob, float(prob_map.max()), largest_region_fraction(prob_map, pixel_threshold),
                         threshold, min_region_fraction)


def decide_with(cal: Calibration, doc_prob: float, prob_map: np.ndarray) -> bool:
    """``is_forged``; eşikler ``Calibration``'dan (belge koşulu ``doc_threshold``, bölge ``pixel_threshold``)."""
    return is_forged(doc_prob, prob_map, cal.doc_threshold, cal.pixel_threshold, cal.min_region_fraction)


def select_operating_point(doc_probs: np.ndarray, max_pixel: np.ndarray, regions: np.ndarray, labels: np.ndarray,
                           doc_thresholds: np.ndarray, target_fpr: float,
                           min_region_fraction: float = MIN_REGION_FRACTION) -> dict[str, Any]:
    """(belge eşiği, piksel eşiği) çiftini birlikte seçer: FPR ≤ ``target_fpr`` koşulunda TPR en büyük.

    Args:
        doc_probs: (N,) kalibre (``fuse_doc_score`` uygulanmış) belge olasılıkları.
        max_pixel: (N,) piksel sıcaklığıyla kalibre haritaların maksimumu.
        regions: (N, P) her aday piksel eşiğinde en büyük bağlantılı bölge oranı.
        doc_thresholds: (D,) aday belge eşikleri.
    Returns:
        ``doc_index``, ``pixel_index``, ``fpr``, ``tpr``, ``feasible`` (hedef tutturuldu mu).

    Eşitlikte önce düşük FPR, sonra **yüksek** eşikler (daha muhafazakâr çalışma noktası) seçilir. Hiçbir
    çift hedefi tutturamazsa FPR'si en düşük (onların içinde TPR'si en yüksek) çift döner.
    """
    labels = np.asarray(labels).astype(bool)
    doc_ok = ((np.asarray(doc_probs, float)[None, :] >= np.asarray(doc_thresholds, float)[:, None])
              & (np.asarray(max_pixel, float)[None, :] >= CLEAN_PIXEL_LIMIT))           # (D, N)
    pixel_ok = np.asarray(regions, float).T >= min_region_fraction                       # (P, N)
    decided = doc_ok[:, None, :] | pixel_ok[None, :, :]                                  # (D, P, N)
    shape = decided.shape[:2]
    fpr = decided[..., ~labels].mean(axis=-1) if (~labels).any() else np.zeros(shape)
    tpr = decided[..., labels].mean(axis=-1) if labels.any() else np.zeros(shape)
    d_idx, p_idx = (a.ravel() for a in np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"))
    fpr, tpr = fpr.ravel(), tpr.ravel()
    feasible = fpr <= target_fpr + 1e-12
    if feasible.any():
        # np.lexsort: son anahtar birincil → TPR (büyük), FPR (küçük), belge eşiği (büyük), piksel eşiği (büyük)
        cand = np.flatnonzero(feasible)
        k = cand[np.lexsort((-p_idx[cand], -d_idx[cand], fpr[cand], -tpr[cand]))[0]]
    else:
        k = np.lexsort((-p_idx, -d_idx, -tpr, fpr))[0]
    return {"doc_index": int(d_idx[k]), "pixel_index": int(p_idx[k]), "fpr": float(fpr[k]), "tpr": float(tpr[k]),
            "feasible": bool(feasible.any())}


def make_overlay(image: np.ndarray, mask_resized: np.ndarray) -> np.ndarray:
    """%60 orijinal + %40 jet ısı haritası (RGB uint8)."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)


def explain_cam(model: DualStreamForensicNet, image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Belgenin (orijinal boyutta) Grad-CAM haritası; ön işlem kayan pencere çıkarımıyla aynıdır."""
    small, _ = limit_long_side(image, None, config.get("max_side", 1600))
    cam = grad_cam(model, to_tensor(small), tile=config.get("crop_size", INPUT_SIZE))
    return cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def save_figure(image: np.ndarray, mask_resized: np.ndarray, overlay: np.ndarray, doc_prob: float,
                threshold: float, out_path: Path, extra: Optional[list[tuple[str, np.ndarray]]] = None) -> None:
    """3 temel panel + ``extra`` (başlık, RGB görüntü) panelleri (hata haritası, Grad-CAM)."""
    area = float((mask_resized >= threshold).mean() * 100)
    extra = extra or []
    fig, axes = plt.subplots(1, 3 + len(extra), figsize=(6 * (3 + len(extra)), 6.5))

    axes[0].imshow(image)
    axes[0].set_title("Orijinal Belge")

    im = axes[1].imshow(mask_resized, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Sahtecilik Olasılık Haritası")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    if mask_resized.max() >= threshold:
        axes[2].contour(mask_resized >= threshold, levels=[0.5], colors="white", linewidths=1.0)
    axes[2].set_title(f"Isı Haritası Bindirme (eşik={threshold:.2f})")

    for ax, (title, panel) in zip(axes[3:], extra):
        ax.imshow(panel)
        ax.set_title(title)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Belge sahtecilik olasılığı: {doc_prob:.1%}  ·  Eşik üstü alan: {area:.2f}%", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint}")
    image_path = resolve_image_path(args.image)

    model, config = load_model(checkpoint, args.device)
    logger.info("Model yüklendi: %s (epoch=%s, val/score=%.4f)", checkpoint, config.get("epoch"),
                config.get("metrics", {}).get("val/score", float("nan")))

    cal = load_calibration_info(args.calibration, checkpoint)
    if args.threshold is not None:
        cal.doc_threshold = args.threshold
    if args.pixel_threshold is not None:
        cal.pixel_threshold = args.pixel_threshold
    # Görselleştirme (kontur, hata haritası, alan) piksel eşiğiyle yapılır.
    temperature, threshold = cal.doc_temperature, cal.pixel_threshold
    logger.info("Kalibrasyon: %s (%s)", cal.summary(), cal.path)

    image = read_image_rgb(image_path)
    logger.info("Görsel: %s (%dx%d), mod=%s", image_path, image.shape[1], image.shape[0], args.mode)
    mask_resized, raw_doc_prob = predict(model, image, config, temperature, args.device, args.mode,
                                         pixel_temperature=cal.pixel_temperature)
    doc_prob = fuse_doc_score(raw_doc_prob, float(mask_resized.max()))
    if doc_prob != raw_doc_prob:
        logger.info("Piksel haritası temiz (maks < %.2f): belge skoru %.4f → %.4f", CLEAN_PIXEL_LIMIT,
                    raw_doc_prob, doc_prob)
    overlay = make_overlay(image, mask_resized)

    extra: list[tuple[str, np.ndarray]] = []
    if args.gt_mask:
        gt = read_mask(args.gt_mask, image.shape[:2])
        counts = error_counts(mask_resized, gt, threshold)
        logger.info("Hata haritası (eşik=%.2f): TP=%d, FP=%d, FN=%d piksel", threshold,
                    counts["tp"], counts["fp"], counts["fn"])
        extra.append(("Hata Haritası (yeşil=TP, kırmızı=FP, mavi=FN)", error_map(image, mask_resized, gt, threshold)))
    if args.gradcam:
        if args.mode != "sliding":
            logger.warning("Grad-CAM kayan pencere ön işlemiyle hesaplanır (--mode %s yok sayıldı).", args.mode)
        extra.append(("Grad-CAM (belge skoru)", cam_overlay(image, explain_cam(model, image, config))))

    out_path = Path(args.output)
    save_figure(image, mask_resized, overlay, doc_prob, threshold, out_path, extra)
    logger.info("Belge olasılığı=%.4f, maks piksel=%.4f, eşik üstü alan=%.2f%%",
                doc_prob, mask_resized.max(), (mask_resized >= threshold).mean() * 100)
    logger.info("Karar (belge eşiği=%.2f, piksel eşiği=%.2f): %s", cal.doc_threshold, cal.pixel_threshold,
                "SAHTE" if decide_with(cal, doc_prob, mask_resized) else "ORİJİNAL")
    logger.info("Sonuç kaydedildi: %s", out_path.resolve())


if __name__ == "__main__":
    main()
