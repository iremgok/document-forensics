"""
Değerlendirme metrikleri (yalnızca NumPy; train.py ve evaluate.py ortak kullanır).

* Görüntü düzeyi: ROC-AUC (Mann-Whitney U), doğruluk, FPR.
* Piksel düzeyi: F1 ve IoU — hem küresel (tüm piksellerin toplamı) hem de
  sahte görüntüler üzerinde görüntü başına ortalama (DocTamper protokolü).
* Kalibrasyon: ECE (Expected Calibration Error), Brier skoru, sıcaklık ölçekleme.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def _f1_iou(tp: float, fp: float, fn: float) -> tuple[float, float]:
    """F1 = 2TP / (2TP + FP + FN),  IoU = TP / (TP + FP + FN)."""
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 1.0
    return f1, iou


@dataclass
class PixelMetrics:
    """Piksel metriklerini görüntü görüntü biriktirir."""

    threshold: float = 0.5
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    per_image_f1: list[float] = field(default_factory=list)
    per_image_iou: list[float] = field(default_factory=list)

    def update(self, prob_map: np.ndarray, gt_mask: np.ndarray) -> None:
        pred = prob_map >= self.threshold
        gt = gt_mask > 0.5
        tp = float(np.logical_and(pred, gt).sum())
        fp = float(np.logical_and(pred, ~gt).sum())
        fn = float(np.logical_and(~pred, gt).sum())
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
        if gt.any():  # Görüntü başına metrikler yalnızca sahte görüntülerde anlamlıdır.
            f1, iou = _f1_iou(tp, fp, fn)
            self.per_image_f1.append(f1)
            self.per_image_iou.append(iou)

    def compute(self) -> dict[str, float]:
        f1, iou = _f1_iou(self.tp, self.fp, self.fn)
        return {
            "pixel_f1": f1,
            "pixel_iou": iou,
            "pixel_f1_per_image": float(np.mean(self.per_image_f1)) if self.per_image_f1 else float("nan"),
            "pixel_iou_per_image": float(np.mean(self.per_image_iou)) if self.per_image_iou else float("nan"),
        }


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
        "image_acc": float((pred == labels).mean()),
        "image_tpr": float(pred[pos].mean()) if pos.any() else float("nan"),
        "image_fpr": float(pred[neg].mean()) if neg.any() else float("nan"),
        "ece": ece,
        "brier": brier_score(probs, labels),
    }
