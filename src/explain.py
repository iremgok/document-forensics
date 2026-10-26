"""
Açıklanabilirlik ve hata analizi görselleri.

* ``error_map``: tahmin ile gerçek maskenin piksel düzeyinde karşılaştırması
  (yeşil = doğru tespit / TP, kırmızı = yanlış alarm / FP, mavi = kaçırılan / FN).
* ``grad_cam``: **sınıflandırma kafasının** (belge skoru) kararını en çok etkileyen bölgeler.
  Segmentasyon haritası "sahte bölge nerede?" sorusunu yanıtlar; Grad-CAM ise "belge skoru neden
  yüksek?" sorusunu. İkisi çeliştiğinde (harita temiz, skor yüksek) Grad-CAM yanlış alarmın kaynağını
  (logo, tablo çizgisi, tutar alanı konumu...) gösterir.

Grad-CAM (Selvaraju ve ark., 2017): kodlayıcının en derin öznitelik haritası A^k için

    α_k = ort_{i,j} ∂y / ∂A^k_{ij},    L = ReLU(Σ_k α_k A^k)

Belgede kayan pencerelerle hesaplanır; örtüşen bölgelerde ortalama alınır ve tüm belge [0, 1]'e
ölçeklenir (pencereler arası karşılaştırılabilirlik için pencere başına değil, belge başına normalize).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

#: Hata haritası renkleri (RGB).
TP_COLOR, FP_COLOR, FN_COLOR = (0, 200, 0), (230, 0, 0), (0, 90, 255)


def error_map(image: np.ndarray, prob_map: np.ndarray, gt_mask: np.ndarray, threshold: float,
              alpha: float = 0.45) -> np.ndarray:
    """TP/FP/FN renkli hata haritası (RGB uint8). ``gt_mask`` > 0 sahte piksel kabul edilir."""
    if prob_map.shape != image.shape[:2]:
        prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    if gt_mask.shape != image.shape[:2]:
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    pred, gt = prob_map >= threshold, gt_mask > 0
    overlay = image.copy()
    overlay[pred & gt] = TP_COLOR
    overlay[pred & ~gt] = FP_COLOR
    overlay[~pred & gt] = FN_COLOR
    out = cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)
    # İnce bölgeler küçültülmüş görselde kaybolmasın diye gerçek maskenin sınırı beyaz çizilir.
    contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), max(1, image.shape[1] // 800))
    return out


def error_counts(prob_map: np.ndarray, gt_mask: np.ndarray, threshold: float) -> dict[str, int]:
    """Hata haritasındaki piksel sayıları."""
    pred, gt = prob_map >= threshold, gt_mask > 0
    return {"tp": int((pred & gt).sum()), "fp": int((pred & ~gt).sum()), "fn": int((~pred & gt).sum())}


def _window_cam(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Bir pencere yığını için normalize edilmemiş Grad-CAM (B, H, W)."""
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, outputs):
        feat = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        feat.retain_grad()
        store["feat"] = feat

    handle = model.net.encoder.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            _, cls_logit = model(batch)
            feat = store["feat"]
            grad, = torch.autograd.grad(cls_logit.sum(), feat)
    finally:
        handle.remove()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
    return F.interpolate(cam, size=batch.shape[-2:], mode="bilinear", align_corners=False)[:, 0].detach()


def grad_cam(model: torch.nn.Module, image: torch.Tensor, tile: int = 512, overlap: float = 0.25,
             batch_size: int = 2) -> np.ndarray:
    """
    Sınıflandırma kafası için belge düzeyinde Grad-CAM haritası.

    Args:
        image: (3, H, W) normalize tensör (``sliding_window_predict`` ile aynı girdi).
    Returns:
        (H, W) float32 [0, 1].
    """
    from src.models import _tile_starts

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    _, h, w = image.shape
    ph, pw = max(h, tile), max(w, tile)
    padded = F.pad(image.unsqueeze(0), (0, pw - w, 0, ph - h), mode="reflect" if (pw - w < w and ph - h < h)
                   else "replicate")[0]
    stride = max(32, int(tile * (1 - overlap)))
    coords = [(y, x) for y in _tile_starts(ph, tile, stride) for x in _tile_starts(pw, tile, stride)]
    cam_sum = torch.zeros((ph, pw), device=device)
    count = torch.zeros((ph, pw), device=device)
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = torch.stack([padded[:, y:y + tile, x:x + tile] for y, x in chunk]).to(device)
        for (y, x), cam in zip(chunk, _window_cam(model, batch)):
            cam_sum[y:y + tile, x:x + tile] += cam
            count[y:y + tile, x:x + tile] += 1
    model.train(was_training)
    cam = (cam_sum / count.clamp_min(1))[:h, :w]
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.cpu().numpy().astype(np.float32)


def cam_overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Grad-CAM haritasını belgenin üzerine bindirir (RGB uint8, 'jet' renk paleti).

    Saydamlık piksel başına CAM değeriyle ölçeklenir: düşük etkili bölgelerde belge olduğu gibi
    görünür, yalnızca kararı etkileyen bölgeler renklenir.
    """
    if cam.shape != image.shape[:2]:
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    cam = np.clip(cam, 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    weight = (alpha * cam)[..., None]
    return (image * (1 - weight) + heat * weight).astype(np.uint8)
