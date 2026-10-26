"""
Değerlendirme metrikleri (yalnızca NumPy; train.py ve evaluate.py ortak kullanır).

* Görüntü düzeyi: ROC-AUC (Mann-Whitney U), doğruluk, FPR.
* Piksel düzeyi: F1 ve IoU — hem küresel (tüm piksellerin toplamı) hem de
  sahte görüntüler üzerinde görüntü başına ortalama (DocTamper protokolü).
* Kalibrasyon: ECE (Expected Calibration Error), Brier skoru, sıcaklık ölçekleme.
* Çalışma noktası: belirli yanlış alarm oranında (FPR) yakalama oranı (TPR@FPR).
* Belirsizlik: bootstrap güven aralıkları (belge ve görüntü bazında yeniden örnekleme).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    ROC-AUC'yi Mann-Whitney U istatistiği ile hesaplar.

        AUC = (R₊ − n₊(n₊+1)/2) / (n₊ · n₋)

    R₊: pozitif örneklerin (tüm skorlar içindeki) sıra toplamı. Eşit skorlara
    ortalama sıra verilir. AUC, rastgele seçilen bir sahte belgenin rastgele
    seçilen bir orijinal belgeden daha yüksek skor alma olasılığıdır.
    Tek sınıf varsa tanımsızdır (NaN).
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Ortalama kesinlik (AP, PR eğrisinin altındaki alan; scikit-learn ile aynı adım tanımı).

        AP = Σ_k (R_k − R_{k−1}) · P_k

    Skorlar büyükten küçüğe sıralanır; her pozitifte o ana kadarki kesinlik toplanır. Pozitif
    sınıf nadir olduğunda (sahte pikseller < %1) ROC-AUC'den daha bilgilendiricidir.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    # Eşit skorlar tek eşik olarak ele alınır (yalnızca grup sonlarında kesinlik okunur).
    last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(labels)[last]
    precision = tp / (last + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """Orijinallerde yanlış alarm oranı ``target_fpr``'ı aşmayan en düşük eşikteki yakalama oranı.

    Finansal senaryodaki çalışma noktası: "100 orijinalden en fazla 2'sine alarm verirken sahtelerin
    yüzde kaçını yakalıyoruz?". Eşik, orijinal skorların (1 − target_fpr) kantilinin hemen üstüdür.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    neg, pos = np.sort(scores[labels == 0]), scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allowed = int(np.floor(target_fpr * len(neg)))   # izin verilen yanlış alarm sayısı
    threshold = neg[len(neg) - allowed - 1] if allowed < len(neg) else -np.inf
    return float((pos > threshold).mean())


def bootstrap_ci(metric_fn: Callable[..., float], arrays: Sequence[np.ndarray], n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0, strata: Optional[np.ndarray] = None) -> tuple[float, float]:
    """
    Metriğin yüzdelik (percentile) bootstrap güven aralığı: ``(alt, üst)``, varsayılan %95.

    ``arrays`` aynı uzunluktaki dizilerdir (ör. skorlar ve etiketler); her yinelemede satırlar
    yerine koyarak yeniden seçilip ``metric_fn(*arrays)`` hesaplanır. ``strata`` (ör. etiketler)
    verilirse yeniden örnekleme her katmanın içinde yapılır; küçük kümelerde bir sınıfın tamamen
    kaybolup AUC'nin tanımsız kalmasını önler. NaN çıkan yinelemeler atlanır.

    Piksel metrikleri için satırlar **görüntülerdir** (görüntü başına TP/FP/FN); pikselleri tek tek
    yeniden örneklemek aynı görüntüdeki piksellerin bağımlılığını yok sayar ve aralığı yapay olarak daraltır.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)] if strata is not None else [np.arange(n)]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        value = metric_fn(*(a[idx] for a in arrays))
        if value == value:  # NaN değil
            stats.append(value)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _f1_iou(tp: float, fp: float, fn: float) -> tuple[float, float]:
    """F1 = 2TP / (2TP + FP + FN),  IoU = TP / (TP + FP + FN)."""
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 1.0
    return f1, iou


def pixel_counts_f1(counts: np.ndarray) -> float:
    """Görüntü başına (tp, fp, fn) satırlarından küresel piksel F1 (bootstrap için)."""
    tp, fp, fn = np.asarray(counts, float).sum(axis=0)
    return _f1_iou(tp, fp, fn)[0]


@dataclass
class PixelMetrics:
    """Piksel metriklerini görüntü görüntü biriktirir.

    * Küresel F1 / IoU / precision / recall: tüm görüntülerin TP-FP-FN toplamından.
      İkili segmentasyonda **Dice = F1**'dir.
    * Görüntü başına F1 / IoU: yalnızca sahte görüntülerin ortalaması (DocTamper protokolü).
    * ``counts``: görüntü başına (tp, fp, fn) — bootstrap güven aralığı için.
    * ``sample_pixels > 0`` ise her görüntüden en fazla bu kadar piksel (olasılık, etiket) rastgele
      örneklenir; eşikten bağımsız piksel ROC-AUC ve AP bunlardan hesaplanır. Sahte pikseller
      nadir olduğu için örnekleme sınıf başına yapılır: her sınıftan ``sample_pixels // 2``.
    """

    threshold: float = 0.5
    sample_pixels: int = 0
    seed: int = 0
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    per_image_f1: list[float] = field(default_factory=list)
    per_image_iou: list[float] = field(default_factory=list)
    counts: list[tuple[float, float, float]] = field(default_factory=list)
    _scores: list[np.ndarray] = field(default_factory=list, repr=False)
    _labels: list[np.ndarray] = field(default_factory=list, repr=False)

    def update(self, prob_map: np.ndarray, gt_mask: np.ndarray) -> None:
        pred = prob_map >= self.threshold
        gt = gt_mask > 0.5
        tp = float(np.logical_and(pred, gt).sum())
        fp = float(np.logical_and(pred, ~gt).sum())
        fn = float(np.logical_and(~pred, gt).sum())
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
        self.counts.append((tp, fp, fn))
        if gt.any():  # Görüntü başına metrikler yalnızca sahte görüntülerde anlamlıdır.
            f1, iou = _f1_iou(tp, fp, fn)
            self.per_image_f1.append(f1)
            self.per_image_iou.append(iou)
        if self.sample_pixels > 0:
            rng = np.random.default_rng(self.seed + len(self.counts))
            flat_p, flat_g = prob_map.ravel(), gt.ravel()
            for cls in (True, False):
                idx = np.flatnonzero(flat_g == cls)
                if len(idx):
                    take = rng.choice(idx, size=min(len(idx), self.sample_pixels // 2), replace=False)
                    self._scores.append(flat_p[take].astype(np.float32))
                    self._labels.append(np.full(len(take), cls, dtype=np.uint8))

    def compute(self) -> dict[str, float]:
        f1, iou = _f1_iou(self.tp, self.fp, self.fn)
        out = {
            "pixel_f1": f1,
            "pixel_iou": iou,
            "pixel_precision": self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else float("nan"),
            "pixel_recall": self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else float("nan"),
            "pixel_f1_per_image": float(np.mean(self.per_image_f1)) if self.per_image_f1 else float("nan"),
            "pixel_iou_per_image": float(np.mean(self.per_image_iou)) if self.per_image_iou else float("nan"),
        }
        if self._scores:
            scores, labels = np.concatenate(self._scores), np.concatenate(self._labels)
            # Sınıf başına eşit örnekleme AUC'yi değiştirmez (sıralama olasılığı); AP ise sınıf
            # oranına bağlıdır, bu yüzden gerçek oranla yeniden ağırlıklandırmadan "dengeli AP" olarak raporlanır.
            out["pixel_auc"] = roc_auc(scores, labels)
            out["pixel_ap_balanced"] = average_precision(scores, labels)
        return out

    def f1_ci(self, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
        """Küresel piksel F1'in görüntü bazlı bootstrap güven aralığı."""
        if not self.counts:
            return float("nan"), float("nan")
        return bootstrap_ci(pixel_counts_f1, [np.asarray(self.counts)], n_boot=n_boot, seed=seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
                               ) -> tuple[float, pd.DataFrame]:
    """
    ECE ve güvenilirlik (reliability) tablosu.

    Olasılıklar ``n_bins`` eşit genişlikli kutuya ayrılır. Her kutuda ortalama
    tahmin (güven) ile gerçek sahte oranı (doğruluk) karşılaştırılır:

        ECE = Σ_b (|B_b| / N) · |ort(p)_b − ort(y)_b|

    Mükemmel kalibre bir modelde "%80 sahte" denilen belgelerin %80'i gerçekten sahtedir.
    """
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        sel = bins == b
        if not sel.any():
            continue
        conf, acc = probs[sel].mean(), labels[sel].mean()
        ece += sel.mean() * abs(conf - acc)
        rows.append({"bin": f"{b / n_bins:.2f}-{(b + 1) / n_bins:.2f}", "count": int(sel.sum()),
                     "mean_confidence": conf, "fraction_forged": acc})
    return float(ece), pd.DataFrame(rows)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier = ort((p − y)²); hem ayırt edicilik hem kalibrasyonu ölçer (düşük = iyi)."""
    return float(np.mean((np.asarray(probs, float) - np.asarray(labels, float)) ** 2))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Sıcaklık ölçekleme (Guo ve ark., 2017): ``p = σ(z / T)``.

    T, **validasyon** kümesinde negatif log-olabilirliği (NLL) en aza indiren değer
    olarak log-uzayda ızgara aramasıyla bulunur. T > 1 aşırı özgüveni yumuşatır.
    Sıralamayı değiştirmediği için AUC etkilenmez; yalnızca kalibrasyon iyileşir.
    Test kümesi bu adımda asla kullanılmaz.
    """
    logits, labels = np.asarray(logits, float), np.asarray(labels, float)
    best_t, best_nll = 1.0, float("inf")
    for t in np.exp(np.linspace(np.log(0.05), np.log(20.0), 400)):
        p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def image_level_metrics(doc_logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0,
                        threshold: float = 0.5) -> dict[str, float]:
    """Belge düzeyi metriklerin tamamı (AUC, doğruluk, FPR/TPR, ECE, Brier)."""
    probs = sigmoid(np.asarray(doc_logits, float) / temperature)
    labels = np.asarray(labels, int)
    pred = probs >= threshold
    ece, _ = expected_calibration_error(probs, labels)
    neg, pos = labels == 0, labels == 1
    return {
        "image_auc": roc_auc(probs, labels),
        "image_ap": average_precision(probs, labels),
        "image_acc": float((pred == labels).mean()),
        "image_tpr": float(pred[pos].mean()) if pos.any() else float("nan"),
        "image_fpr": float(pred[neg].mean()) if neg.any() else float("nan"),
        "image_tpr_at_fpr01": tpr_at_fpr(probs, labels, 0.01),
        "image_tpr_at_fpr02": tpr_at_fpr(probs, labels, 0.02),
        "image_tpr_at_fpr05": tpr_at_fpr(probs, labels, 0.05),
        "ece": ece,
        "brier": brier_score(probs, labels),
    }


def binary_decision_metrics(decided: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """İkili karar (``is_forged``) için karmaşıklık matrisi ve türetilmiş metrikler."""
    decided, labels = np.asarray(decided, bool), np.asarray(labels, int).astype(bool)
    tp, fp = int((decided & labels).sum()), int((decided & ~labels).sum())
    fn, tn = int((~decided & labels).sum()), int((~decided & ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"),
        "accuracy": (tp + tn) / max(len(labels), 1),
    }
