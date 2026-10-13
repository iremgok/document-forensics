"""
Sahtecilik segmentasyonu için kayıp fonksiyonları.

Belgelerde sahte pikseller genellikle görüntünün %1'inden azdır. Bu aşırı sınıf
dengesizliğinde düz BCE, "her şey orijinal" tahminiyle çok düşük kayba ulaşır.
Çözüm iki tamamlayıcı kayıptır:

* **Dice Loss** — bölge örtüşmesini doğrudan optimize eder; arka plan piksellerinin
  sayısından bağımsızdır (küresel, ölçek-değişmez).
* **Focal Loss** — piksel başına çalışır; kolay (doğru ve emin) örneklerin
  katkısını ``(1 − p_t)^γ`` ile bastırarak zor kenar piksellerine odaklanır.

Toplam segmentasyon kaybı:  ``L_seg = L_dice + 2 · L_focal``
Belge kafası için BCE eklenir:  ``L = L_seg + λ_cls · L_bce(cls)``
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Yumuşak (soft) Dice kaybı.

        p = σ(logit),  t ∈ {0, 1}
        Dice = (2 · Σ p·t + s) / (Σ p + Σ t + s)
        L_dice = 1 − Dice

    Toplamlar **tüm batch üzerinden** alınır. Örnek bazlı ortalama yerine batch
    bazlı hesap, maskesi boş (orijinal) yamalarda kaybın patlamasını önler:
    boş bir yamada Σt = 0 olduğundan tek başına hesaplanan Dice yalnızca yanlış
    pozitifleri cezalandırır ve gradyanı çok gürültülüdür. ``s`` (smooth) payda
    sıfırlanmasını engeller.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        targets = targets.float()
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    İkili Focal Loss (Lin ve ark., 2017).

        p_t = p        (t = 1 ise)      α_t = α        (t = 1 ise)
        p_t = 1 − p    (t = 0 ise)      α_t = 1 − α    (t = 0 ise)
        FL  = −α_t · (1 − p_t)^γ · log(p_t)

    * γ = 2: doğru sınıflandırılan (p_t ≈ 1) piksellerin katkısı ~0'a iner.
    * α: pozitif sınıf ağırlığı. Dice dengesizliği zaten telafi ettiği için
      RetinaNet'teki varsayılan 0.25 kullanılır; bu, focal terimin yanlış
      pozitifleri (orijinal pikselleri sahte sanmayı) da cezalandırmasını sağlar.

    ``log(p_t)`` sayısal kararlılık için ``binary_cross_entropy_with_logits`` ile
    hesaplanır (BCE = −log p_t).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class ForensicLoss(nn.Module):
    """
    Çift kafalı model için birleşik kayıp.

        L = (L_dice + focal_weight · L_focal)  +  cls_weight · BCE(cls_logit, y_yama)

    ``y_yama``, kırpılmış yamanın maskesinde en az bir sahte piksel olup
    olmadığıdır (belge etiketi değil): rastgele kırpma sahte bölgeyi dışarıda
    bırakabilir; bu durumda yamayı "sahte" diye etiketlemek gürültülü öğretir.
    """

    def __init__(self, focal_weight: float = 2.0, cls_weight: float = 0.5,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.focal_weight = focal_weight
        self.cls_weight = cls_weight

    def forward(self, mask_logits: torch.Tensor, cls_logit: torch.Tensor,
                masks: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_labels = (masks.flatten(1).amax(dim=1) > 0.5).float()
        dice = self.dice(mask_logits, masks)
        focal = self.focal(mask_logits, masks)
        cls = F.binary_cross_entropy_with_logits(cls_logit.float(), patch_labels)
        total = dice + self.focal_weight * focal + self.cls_weight * cls
        return {"loss": total, "dice": dice.detach(), "focal": focal.detach(), "cls": cls.detach()}
