"""
Tek belge için sahtecilik çıkarımı ve 3 panelli görselleştirme.

Çalıştırma:
    python predict.py
    python predict.py --mode resize --image belge.jpg --checkpoint checkpoints/finetuned_multilingual.pth

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

Kalibrasyon: ``checkpoints/calibration.json`` → ``checkpoints/synthetic/calibration.json``
sırasıyla aranır; ikisi de yoksa varsayılan değerlerle ``checkpoints/calibration.json``
oluşturulur. Anahtarlar: ``temperature`` ve ``optimal_threshold`` (ya da ``threshold``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")  # Ekransız ortamlarda (sunucu/Kaggle) da çalışsın
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, limit_long_side, read_image_rgb
from src.models import DualStreamForensicNet, load_model, sliding_window_predict

logger = logging.getLogger("predict")

INPUT_SIZE = 512
DEFAULT_IMAGES = ("test_document.jpeg", "test_document.jpg", "test_document.png")
IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp")
CALIBRATION_PATHS = (Path("checkpoints/calibration.json"), Path("checkpoints/synthetic/calibration.json"))
DEFAULT_CALIBRATION = {"temperature": 6.389, "optimal_threshold": 0.35}
CLEAN_PIXEL_LIMIT = 0.05  # Bu değerin altındaki maks. piksel olasılığı "sahte bölge yok" sayılır
DOC_WEIGHT = 0.7          # Temiz haritada belge skorunun ağırlığı (kalanı maks. piksel olasılığı)
MIN_REGION_FRACTION = 0.0005  # Piksel koşulu için asgari bağlantılı bölge alanı (sayfanın %0.05'i)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DualStreamForensicNet ile belge sahtecilik tahmini")
    p.add_argument("--image", default=None,
                   help=f"Girdi belge görseli (varsayılan: {' / '.join(DEFAULT_IMAGES)} içinden ilk bulunan)")
    p.add_argument("--checkpoint", default="checkpoints/finetuned_multilingual.pth", help="Model ağırlıkları")
    p.add_argument("--mode", choices=("sliding", "resize"), default="sliding",
                   help="sliding: orijinal çözünürlükte kayan pencere; resize: 512x512 tek geçiş")
    p.add_argument("--output", default="prediction_result.png", help="Kaydedilecek figür")
    p.add_argument("--calibration", default=None,
                   help="Kalibrasyon JSON yolu (verilmezse otomatik aranır / oluşturulur)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Piksel eşiği (verilirse kalibrasyon dosyasındakini ezer)")
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


def load_calibration(explicit: str | None = None) -> tuple[float, float, Path]:
    """Sıcaklık ve optimal eşiği okur; dosya yoksa varsayılanlarla oluşturur.

    Returns:
        (temperature, threshold, okunan dosya)
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Kalibrasyon dosyası bulunamadı: {path}")
    else:
        path = next((p for p in CALIBRATION_PATHS if p.exists()), None)
        if path is None:
            path = CALIBRATION_PATHS[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(DEFAULT_CALIBRATION, indent=2), encoding="utf-8")
            logger.info("Kalibrasyon dosyası bulunamadı, varsayılanlarla oluşturuldu: %s", path)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    temperature = float(data.get("temperature", DEFAULT_CALIBRATION["temperature"]))
    threshold = float(data.get("optimal_threshold", data.get("threshold", DEFAULT_CALIBRATION["optimal_threshold"])))
    if temperature <= 0:
        raise ValueError(f"Geçersiz sıcaklık değeri ({temperature}): {path}")
    return temperature, threshold, path


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) → normalize (3, H, W) tensör."""
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    return (tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Çıkarım
# ---------------------------------------------------------------------------

def predict(model: DualStreamForensicNet, image: np.ndarray, config: dict[str, Any], temperature: float,
            device: str, mode: str = "sliding") -> tuple[np.ndarray, float]:
    """
    Returns:
        (orijinal boyutta kalibre olasılık haritası float32 [0,1], kalibre belge olasılığı)
    """
    prob_map, doc_logit = predict_raw(model, image, config, temperature, device, mode)
    return prob_map, float(torch.sigmoid(torch.tensor(doc_logit / temperature)))


@torch.no_grad()
def predict_raw(model: DualStreamForensicNet, image: np.ndarray, config: dict[str, Any], temperature: float,
                device: str, mode: str = "sliding") -> tuple[np.ndarray, float]:
    """``predict`` ile aynı; belge skorunu kalibre edilmemiş ham logit olarak döndürür
    (değerlendirmede farklı sıcaklıkların karşılaştırılabilmesi için).

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


def make_overlay(image: np.ndarray, mask_resized: np.ndarray) -> np.ndarray:
    """%60 orijinal + %40 jet ısı haritası (RGB uint8)."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)


def save_figure(image: np.ndarray, mask_resized: np.ndarray, overlay: np.ndarray, doc_prob: float,
                threshold: float, out_path: Path) -> None:
    area = float((mask_resized >= threshold).mean() * 100)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    axes[0].imshow(image)
    axes[0].set_title("Orijinal Belge")

    im = axes[1].imshow(mask_resized, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Sahtecilik Olasılık Haritası")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    if mask_resized.max() >= threshold:
        axes[2].contour(mask_resized >= threshold, levels=[0.5], colors="white", linewidths=1.0)
    axes[2].set_title(f"Isı Haritası Bindirme (eşik={threshold:.2f})")

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

    temperature, threshold, calib_path = load_calibration(args.calibration)
    if args.threshold is not None:
        threshold = args.threshold
    logger.info("Kalibrasyon: T=%.3f, eşik=%.2f (%s)", temperature, threshold, calib_path)

    image = read_image_rgb(image_path)
    logger.info("Görsel: %s (%dx%d), mod=%s", image_path, image.shape[1], image.shape[0], args.mode)
    mask_resized, raw_doc_prob = predict(model, image, config, temperature, args.device, args.mode)
    doc_prob = fuse_doc_score(raw_doc_prob, float(mask_resized.max()))
    if doc_prob != raw_doc_prob:
        logger.info("Piksel haritası temiz (maks < %.2f): belge skoru %.4f → %.4f", CLEAN_PIXEL_LIMIT,
                    raw_doc_prob, doc_prob)
    overlay = make_overlay(image, mask_resized)

    out_path = Path(args.output)
    save_figure(image, mask_resized, overlay, doc_prob, threshold, out_path)
    logger.info("Belge olasılığı=%.4f, maks piksel=%.4f, eşik üstü alan=%.2f%%",
                doc_prob, mask_resized.max(), (mask_resized >= threshold).mean() * 100)
    logger.info("Sonuç kaydedildi: %s", out_path.resolve())


if __name__ == "__main__":
    main()
