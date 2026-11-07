"""
Bağımsız değerlendirme betiği: AUC, IoU, Piksel-F1, kalibrasyon ve karar eşiği seçimi.

Kurallar (sızıntı önleme):
  * Bölümler ``train.py``'nin kaydettiği ``splits.csv``'den okunur; burada yeniden
    bölme yapılmaz.
  * Sıcaklık ölçekleme ve karar eşiği yalnızca ``val`` üzerinde öğrenilir, ``test`` ve
    ``ood`` kümelerine sabit olarak uygulanır.
  * Her kaynak ayrı raporlanır (ör. sentetik test, AIForge-Doc OOD, SROIE2019 OOD);
    yalnızca orijinal belgelerden oluşan kümelerde (SROIE) AUC tanımsızdır ve
    yanlış alarm oranı (FPR) raporlanır.

Kalibrasyon ve karar eşikleri (ayrı belge / piksel kalibrasyonu)
---------------------------------------------------------------
Uygulama bir belgeye ``predict.is_forged`` kuralıyla (belge koşulu + piksel bölgesi koşulu) "sahte" der.
Belge skoru ile piksel haritası farklı ölçeklerde aşırı özgüvenlidir; tek bir sıcaklık ikisine birden
uygulanınca (eski şema: T≈4.2) piksel eşiği 0.77 ham olasılıkta ≈0.994'e denk gelir. Bu yüzden val'de:

1. **Belge sıcaklığı** ``doc_temperature``: belge logit'lerinin NLL'i ile (``fit_temperature``).
2. **Piksel sıcaklığı** ``pixel_temperature`` (``--pixel-temperature``): ``fit`` (varsayılan) → val'deki
   sahte belgelerden (maskesi olanlar) düzgün rastgele örneklenen piksellerin ham logit'leri üzerinden NLL;
   ``1`` → sabit 1; ``doc`` → belge sıcaklığıyla aynı (eski davranış).
3. ``(doc_threshold, pixel_threshold)`` çifti ızgara aramasıyla **birlikte**: val orijinallerinde kuralın
   yanlış alarm oranı ``--target-fpr``'ı (varsayılan %5) aşmayacak şekilde TPR'yi en büyükleyen çift
   (``predict.select_operating_point``).

Val'de piksel maskesi olan sahte yoksa piksel tarafı T=1 ve tek eşiğe (belge eşiği = piksel eşiği; eski
davranış) düşer ve uyarı verilir. Sonuç ``predict.calibration_path_for(checkpoint)`` yoluna (``best.pth``
için klasördeki ``calibration.json``) yeni şemayla yazılır; val kümesi yoksa **hiçbir dosya yazılmaz**.

``--from-calibration``: hiçbir şey öğrenmeden modelin mevcut kalibrasyon dosyasını (yeni ya da eski şema)
uygular; dosya val'de öğrenilmemişse sonuçlar "kalibre edilmemiş" olarak işaretlenir (``summary.json`` →
``calibrated``). Val kümesi yerelde olmayan modelleri aynı test bölmesinde karşılaştırmak içindir.

Test kümelerinde model tek geçişte birden çok piksel sıcaklığıyla çalıştırılır (kalibre piksel T'si,
belge T'si ve 1); ``arrays_*.npz`` bunların bölge oranlarını taşır ve ``ablation_decision.py`` "tek T
(eski)" ile "ayrı kalibrasyon"u yeniden çıkarım yapmadan karşılaştırır.

Sahtecilik türüne göre kırılım
------------------------------
Her test/OOD kümesi için ``breakdown_<bölüm>_<kaynak>.csv`` yazılır: sahtecilik türü (dosya adındaki
``__NN_<tür>`` ekinden; ``digit_change`` metadata'daki düzenleme kaydına göre 1/2+ rakam diye ayrılır),
kurum (``real:<kurum>/...``), belge tipi, edinim ve iletim kanalı başına tespit oranı (TPR), yanlış
alarm (FPR) ve AUC. Ayrıntılar, görüntü yolundaki veri seti klasörünün ``--data-root`` altındaki
``metadata.csv``'sinden okunur (Kaggle'da üretilmiş tahminler yerelde de zenginleştirilebilir).

``--breakdown-only`` modeli çalıştırmadan ``--out`` altındaki mevcut ``predictions_*.csv`` dosyalarından
kırılımı üretir (sıcaklık ``summary.json``'dan okunur).

Ek metrikler ve çıktılar
------------------------
* Piksel: precision, recall, F1 (= Dice), IoU; örneklenmiş piksellerden eşikten bağımsız ROC-AUC ve AP.
* Belge: AUC, AP ve **TPR@FPR** (%1, %2, %5 yanlış alarmda yakalama oranı).
* Karar (``is_forged``): TP/FP/FN/TN, precision, recall, specificity, F1, accuracy.
* **%95 bootstrap güven aralıkları** (``--bootstrap``, varsayılan 1000): belge AUC, TPR@FPR=%2, karar
  TPR/FPR/F1 ve piksel F1 (görüntü bazında yeniden örnekleme). ``*_ci_lo`` / ``*_ci_hi`` sütunları.
* ``arrays_<bölüm>_<kaynak>.npz``: görüntü başına eşik taramalı bölge oranları ve pencere logit'leri;
  ``ablation_decision.py`` karar kuralı ve havuzlama ablasyonlarını modeli yeniden çalıştırmadan bunlardan yapar.
* ``--error-maps N``: her küme için en kötü N yanlış alarm, N kaçırılan ve N doğru tespit için
  TP/FP/FN hata haritaları (``error_maps/<bölüm>_<kaynak>/``).
* ``--copymove``: belge içi kopyala-taşı kanıt modülü (``src/copymove.py``) her test görüntüsünde çalışır;
  ``copymove_risk`` sütunu ve ``ablation_decision.py``'de "copy-move modülü açık/kapalı" satırı için.

Dış modeller (baseline)
-----------------------
``--external-maps <klasör>`` modeli yüklemek yerine TruFor/DTD gibi bir modelin önceden kaydedilmiş
olasılık haritalarını okur (``<image_key>.npy`` ya da 8-bit ``.png``; anahtarlar
``tools/export_eval_images.py`` ile dışa aktarılan dosya adlarıdır). Belge skoru varsa
``--external-scores`` CSV'sinden (``key,score``), yoksa haritanın en yüksek %0,1 piksel ortalamasından
alınır. Böylece tüm modeller **aynı metrik koduyla** ölçülür. Dış modellerde sıcaklık 1, eşik
``--threshold`` (varsayılan 0.5) kullanılır.

Örnek:
    python evaluate.py --checkpoint checkpoints/run/best.pth --splits checkpoints/run/splits.csv
    python evaluate.py --checkpoint checkpoints/stageE/best.pth --splits checkpoints/stageE/splits.csv --target-fpr 0.02
    python evaluate.py --breakdown-only --out results           # Kaggle sonuçlarından yerelde kırılım
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch

from predict import (MIN_REGION_FRACTION, Calibration, calibration_path_for, decide_forged, fuse_doc_score,
                     largest_region_fraction, load_calibration_info, select_operating_point)
from src.dataset import ForgeryDataset, build_transforms, image_key, limit_long_side, read_image_rgb, read_mask
from src.explain import error_map
from src.metrics import (PixelMetrics, binary_decision_metrics, bootstrap_ci, expected_calibration_error,
                         fit_temperature, image_level_metrics, roc_auc, sigmoid, tpr_at_fpr)
from src.models import load_model, sliding_window_maps

logger = logging.getLogger("evaluate")

#: Taranan karar eşikleri (uygulamadaki kaydırıcı aralığı, 0.01 adım).
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)

#: ``generate_synthetic`` / ``forge_real`` dosya adı eki: ``..._02_digit_change.png`` → ``digit_change``.
_TAMPER_RE = re.compile(r"__\d{2}_([A-Za-z_]+)\.\w+$")
#: Kırılım boyutları (prediction tablosunda varsa): sütun → rapordaki ad.
BREAKDOWN_DIMS = {"tamper_type": "sahtecilik_turu", "issuer": "kurum", "doc_type": "belge_tipi",
                  "acquisition": "edinim", "channel": "kanal"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sahtecilik modeli değerlendirmesi")
    p.add_argument("--checkpoint", default=None, help="Model ağırlıkları (--breakdown-only dışında zorunlu)")
    p.add_argument("--splits", default=None, help="train.py tarafından yazılan splits.csv (--breakdown-only dışında zorunlu)")
    p.add_argument("--data-root", default="data", help="Kırılım ayrıntıları için metadata.csv'lerin aranacağı kök")
    p.add_argument("--breakdown-only", action="store_true",
                   help="Modeli çalıştırma; --out altındaki predictions_*.csv dosyalarından kırılım üret")
    p.add_argument("--sets", nargs="+", default=["test", "ood"], help="Değerlendirilecek bölümler")
    p.add_argument("--no-calibration", action="store_true", help="Val üzerinde sıcaklık ve eşik öğrenmeyi atla")
    p.add_argument("--target-fpr", type=float, default=0.05,
                   help="Eşik seçiminde val orijinallerinde izin verilen en yüksek yanlış alarm oranı")
    p.add_argument("--max-images", type=int, default=None,
                   help="Kaynak ve sınıf (orijinal/sahte) başına üst sınır (hızlı deneme)")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap yineleme sayısı (0 = güven aralığı yok)")
    p.add_argument("--pixel-samples", type=int, default=4000,
                   help="Piksel AUC/AP için görüntü başına örneklenen piksel sayısı (0 = kapalı)")
    p.add_argument("--error-maps", type=int, default=0,
                   help="Her küme için kaydedilecek hata haritası sayısı (FP, FN ve TP'den N'er tane)")
    p.add_argument("--external-maps", default=None,
                   help="Dış modelin olasılık haritaları klasörü (<image_key>.npy|.png); model yüklenmez")
    p.add_argument("--external-scores", default=None, help="Dış modelin belge skorları CSV'si (key,score)")
    p.add_argument("--ocr", action="store_true",
                   help="Test/OOD görüntülerinde OCR kural motorunu da çalıştır (ocr_risk sütunu; OCR ablasyonu için)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sabit belge sıcaklığı (val'de öğrenmek yerine). Uygulamadaki kararları yeniden üretmek için "
                        "--from-calibration kullanın.")
    p.add_argument("--pixel-temperature", default="fit",
                   help="Piksel sıcaklığı: 'fit' (val sahtelerinde piksel NLL), 'doc' (belge T'siyle aynı; eski "
                        "davranış) ya da sayı (ör. 1)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Kalibrasyon yapılmadığında (dış model / --no-calibration) kullanılan belge eşiği")
    p.add_argument("--pixel-threshold", type=float, default=None,
                   help="Kalibrasyon yapılmadığında piksel (bölge) eşiği (varsayılan: --threshold)")
    p.add_argument("--from-calibration", action="store_true",
                   help="Öğrenme yapma; modelin mevcut calibration.json dosyasını (yeni ya da eski şema) uygula")
    p.add_argument("--calibration-file", default=None,
                   help="--from-calibration için açık dosya yolu (varsayılan: predict.load_calibration_info araması)")
    p.add_argument("--copymove", action="store_true",
                   help="Test/OOD görüntülerinde kopyala-taşı kanıt modülünü de çalıştır (copymove_risk sütunu)")
    p.add_argument("--out", default="results")
    return p.parse_args()


class Predictor:
    """Bir görüntü için (olasılık haritası, belge logit'i, pencere logit'leri) üretir.

    İki kaynak desteklenir: eğitilmiş ``DualStreamForensicNet`` (kayan pencere, uygulamayla aynı) ya da
    dış bir modelin diske kaydedilmiş haritaları (baseline karşılaştırması).
    """

    def __init__(self, model: Optional[torch.nn.Module] = None, config: Optional[dict] = None,
                 maps_dir: Optional[str] = None, scores_csv: Optional[str] = None) -> None:
        self.model, self.config = model, config or {}
        self.maps_dir = Path(maps_dir) if maps_dir else None
        self.scores: dict[str, float] = {}
        if scores_csv:
            table = pd.read_csv(scores_csv)
            self.scores = dict(zip(table["key"].astype(str), table["score"].astype(float)))

    @property
    def max_side(self) -> int:
        # Dış model haritaları orijinal çözünürlüktedir; maske de küçültülmeden karşılaştırılır.
        return self.config.get("max_side", 1600) if self.model is not None else 10**9

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, path: str, temperatures: tuple[float, ...] = (1.0,),
                 logit_map: bool = False) -> tuple[list[np.ndarray], float, np.ndarray, Optional[np.ndarray]]:
        """(her piksel sıcaklığı için olasılık haritası, belge logit'i, pencere logit'leri, ham logit haritası)."""
        if self.model is not None:
            return sliding_window_maps(self.model, image, tile=self.config.get("crop_size", 512),
                                       temperatures=tuple(temperatures), return_logit_map=logit_map)
        prob_map, doc_logit = self._external(image, path)
        logits = np.log(np.clip(prob_map, 1e-6, 1 - 1e-6) / np.clip(1 - prob_map, 1e-6, 1)) if logit_map else None
        # Dış modelin haritası sıcaklıktan bağımsızdır: her sıcaklık için aynı harita.
        return [prob_map] * len(temperatures), doc_logit, np.array([doc_logit], dtype=np.float32), logits

    def _external(self, image: torch.Tensor, path: str) -> tuple[np.ndarray, float]:
        key = image_key(path)
        npy, png = self.maps_dir / f"{key}.npy", self.maps_dir / f"{key}.png"
        if npy.is_file():
            prob_map = np.load(npy).astype(np.float32)
        elif png.is_file():
            prob_map = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        else:
            raise FileNotFoundError(f"Dış model haritası yok: {npy} / {png}")
        h, w = image.shape[-2:]
        if prob_map.shape != (h, w):
            prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_map = np.clip(prob_map, 0.0, 1.0)
        if key in self.scores:
            score = self.scores[key]
        else:
            flat = np.sort(prob_map.ravel())
            score = float(flat[-max(1, flat.size // 1000):].mean())  # en yüksek %0,1 piksel
        score = min(max(score, 1e-6), 1 - 1e-6)
        return prob_map, float(np.log(score / (1 - score)))


def ocr_risks(frame: pd.DataFrame) -> pd.DataFrame:
    """Her görüntü için OCR kural motoru riski ve kanıt bayrağı (``src/ocr_checker.py``)."""
    from src.ocr_checker import FAIL, check_document_text, run_ocr

    gpu = torch.cuda.is_available()
    rows = []
    for i, path in enumerate(frame["image_path"]):
        report = check_document_text(run_ocr(read_image_rgb(path), gpu=gpu))
        rows.append({"ocr_risk": report.risk, "ocr_evidence": report.has_evidence,
                     "ocr_fail": sum(f.status == FAIL for f in report.findings)})
        if (i + 1) % 50 == 0:
            logger.info("  OCR %d / %d", i + 1, len(frame))
    return pd.DataFrame(rows, index=frame.index)


def predict_frame(predictor: Predictor, frame: pd.DataFrame, temperatures: float | tuple[float, ...] = 1.0,
                  pixel_threshold: float = 0.5, pixel_samples: int = 0, nll_samples: int = 0,
                  copymove: bool = False) -> pd.DataFrame:
    """Her görüntü için belge logit'ini, karar özetlerini ve piksel metriklerini hesaplar.

    Piksel haritası uygulamadaki gibi ``temperatures[0]`` (piksel sıcaklığı) ile ölçeklenir; ek sıcaklıklar
    yalnızca karşılaştırma için bölge oranlarını hesaplamakta kullanılır. ``out.attrs``:

    * ``regions``: ana sıcaklıkta ``THRESHOLDS`` eşiklerindeki en büyük bölge oranları (N, len(THRESHOLDS)),
    * ``regions_by_t`` / ``max_pixel_by_t`` / ``pixel_temperatures``: tüm sıcaklıklar için (N, K, ...),
    * ``windows``: görüntü başına pencere ``cls_logit``'leri (havuzlama ablasyonu),
    * ``pixel`` / ``pixel_obj``: piksel metrikleri ve görüntü başına TP/FP/FN taşıyan nesne,
    * ``pixel_nll`` (``nll_samples > 0``): maskeli sahte görüntülerden düzgün örneklenmiş (ham logit, etiket)
      çiftleri — piksel sıcaklığı uydurmak için.
    """
    temps = tuple(temperatures) if isinstance(temperatures, (tuple, list)) else (float(temperatures),)
    dataset = ForgeryDataset(frame, build_transforms(False), predictor.max_side)
    pixel = PixelMetrics(threshold=pixel_threshold, sample_pixels=pixel_samples)
    rng = np.random.default_rng(0)
    logits, max_pixels, regions, windows, cm_risks = [], [], [], [], []
    nll_logits, nll_labels = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        row = frame.iloc[i]
        gt = sample["mask"][0].numpy()
        want_logits = nll_samples > 0 and int(row["label"]) == 1 and gt.any()
        maps, doc_logit, window_logits, logit_map = predictor(sample["image"], row["image_path"], temps, want_logits)
        pixel.update(maps[0], gt)
        logits.append(doc_logit)
        max_pixels.append([float(m.max()) for m in maps])
        regions.append([[largest_region_fraction(m, t) for t in THRESHOLDS] for m in maps])
        windows.append(window_logits)
        if want_logits and logit_map is not None:
            take = rng.choice(logit_map.size, size=min(nll_samples, logit_map.size), replace=False)
            nll_logits.append(logit_map.ravel()[take])
            nll_labels.append((gt.ravel()[take] > 0.5).astype(np.uint8))
        if copymove:
            from src.copymove import detect_copy_move

            image, _ = limit_long_side(read_image_rgb(row["image_path"]), None, predictor.max_side)
            cm_risks.append(detect_copy_move(image).risk)
        if (i + 1) % 100 == 0:
            logger.info("  %d / %d", i + 1, len(dataset))
    out = frame.reset_index(drop=True).copy()
    by_t_max = np.asarray(max_pixels, dtype=np.float64).reshape(len(out), len(temps))
    by_t_regions = np.asarray(regions, dtype=np.float64).reshape(len(out), len(temps), len(THRESHOLDS))
    out["doc_logit"] = logits
    out["max_pixel"] = by_t_max[:, 0]
    if copymove:
        out["copymove_risk"] = cm_risks
    counts = np.asarray(pixel.counts, dtype=np.float64).reshape(len(out), 3)
    out["px_tp"], out["px_fp"], out["px_fn"] = counts[:, 0], counts[:, 1], counts[:, 2]
    out.attrs["pixel"] = pixel.compute()
    out.attrs["pixel_obj"] = pixel
    out.attrs["regions"] = by_t_regions[:, 0]
    out.attrs["regions_by_t"] = by_t_regions
    out.attrs["max_pixel_by_t"] = by_t_max
    out.attrs["pixel_temperatures"] = np.asarray(temps, dtype=np.float64)
    out.attrs["windows"] = windows
    if nll_logits:
        out.attrs["pixel_nll"] = (np.concatenate(nll_logits), np.concatenate(nll_labels))
    return out


def confidence_intervals(pred: pd.DataFrame, decided: np.ndarray, temperature: float, n_boot: int
                         ) -> dict[str, float]:
    """Ana metriklerin %95 bootstrap güven aralıkları (sınıf katmanlı yeniden örnekleme)."""
    if n_boot <= 0:
        return {}
    labels = pred["label"].to_numpy().astype(int)
    probs = sigmoid(pred["doc_logit"].to_numpy(float) / temperature)
    out: dict[str, float] = {}

    def add(name: str, fn, arrays) -> None:
        lo, hi = bootstrap_ci(fn, arrays, n_boot=n_boot, strata=labels)
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = lo, hi

    if len(np.unique(labels)) == 2:
        add("image_auc", roc_auc, [probs, labels])
        add("image_tpr_at_fpr02", lambda s, l: tpr_at_fpr(s, l, 0.02), [probs, labels])
        add("decision_f1", lambda d, l: binary_decision_metrics(d, l)["f1"], [decided, labels])
    if (labels == 1).any():
        add("decision_tpr", lambda d, l: float(d[l == 1].mean()) if (l == 1).any() else float("nan"),
            [decided, labels])
    if (labels == 0).any():
        add("decision_fpr", lambda d, l: float(d[l == 0].mean()) if (l == 0).any() else float("nan"),
            [decided, labels])
    lo, hi = pred.attrs["pixel_obj"].f1_ci(n_boot)
    out["pixel_f1_ci_lo"], out["pixel_f1_ci_hi"] = lo, hi
    return out


def save_arrays(path: Path, pred: pd.DataFrame) -> None:
    """Ablasyon için görüntü başına eşik taramalı bölge oranları ve pencere logit'leri.

    ``regions`` ana (kalibre) piksel sıcaklığındadır (eski çıktılarla aynı anahtar); ``regions_by_t`` /
    ``max_pixel_by_t`` ``pixel_temperatures`` sırasıyla tüm karşılaştırma sıcaklıklarını taşır.
    """
    windows = np.empty(len(pred), dtype=object)
    windows[:] = pred.attrs["windows"]
    extra = {k: pred.attrs[k] for k in ("regions_by_t", "max_pixel_by_t", "pixel_temperatures") if k in pred.attrs}
    np.savez_compressed(path, thresholds=THRESHOLDS, regions=pred.attrs["regions"], windows=windows,
                        image_path=pred["image_path"].astype(str).to_numpy(), **extra)


def save_error_maps(predictor: Predictor, pred: pd.DataFrame, decided: np.ndarray, cal: Calibration,
                    out_dir: Path, n: int) -> None:
    """En belirgin hatalar ve doğrular için TP/FP/FN hata haritaları (yeşil/kırmızı/mavi; piksel eşiğinde)."""
    labels = pred["label"].to_numpy().astype(int)
    probs = sigmoid(pred["doc_logit"].to_numpy(float) / cal.doc_temperature)
    threshold = cal.pixel_threshold
    groups = {
        "FP_yanlis_alarm": np.flatnonzero((labels == 0) & decided),
        "FN_kacirilan": np.flatnonzero((labels == 1) & ~decided),
        "TP_dogru_tespit": np.flatnonzero((labels == 1) & decided),
    }
    order = {"FP_yanlis_alarm": -probs, "FN_kacirilan": probs, "TP_dogru_tespit": -probs}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in groups.items():
        for rank, i in enumerate(idx[np.argsort(order[name][idx])][:n]):
            row = pred.iloc[i]
            image = read_image_rgb(row["image_path"])
            gt = read_mask(row["mask_path"], image.shape[:2])
            dataset = ForgeryDataset(pred.iloc[[i]], build_transforms(False), predictor.max_side)
            sample = dataset[0]
            prob_map = predictor(sample["image"], row["image_path"], (cal.pixel_temperature,))[0][0]
            if prob_map.shape != image.shape[:2]:  # max_side küçültmesi: görselleştirmede orijinal boyut
                prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            vis = error_map(image, prob_map, gt, threshold)
            name_out = f"{name}_{rank:02d}_{image_key(row['image_path'])}_p{probs[i]:.2f}.jpg"
            cv2.imwrite(str(out_dir / name_out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def threshold_index(threshold: float) -> int:
    return int(np.argmin(np.abs(THRESHOLDS - threshold)))


def decisions_at(pred: pd.DataFrame, doc_temperature: float, doc_ti: int, pixel_ti: Optional[int] = None,
                 min_region_fraction: float = MIN_REGION_FRACTION) -> np.ndarray:
    """Belge eşiği ``THRESHOLDS[doc_ti]`` ve piksel eşiği ``THRESHOLDS[pixel_ti]`` (verilmezse aynı; eski tek
    eşik) için her görüntüde ``predict.is_forged`` kararı. Bölge oranları ``pred.attrs["regions"]``'dan
    (ana piksel sıcaklığı) okunur."""
    pixel_ti = doc_ti if pixel_ti is None else pixel_ti
    threshold = float(THRESHOLDS[doc_ti])
    doc_probs = sigmoid(pred["doc_logit"].to_numpy() / doc_temperature)
    return np.array([decide_forged(fuse_doc_score(float(p), m), m, r, threshold, min_region_fraction)
                     for p, m, r in zip(doc_probs, pred["max_pixel"], pred.attrs["regions"][:, pixel_ti])])


def fused_doc_probs(pred: pd.DataFrame, doc_temperature: float) -> np.ndarray:
    """``fuse_doc_score`` uygulanmış kalibre belge olasılıkları (vektörel)."""
    return np.array([fuse_doc_score(float(p), float(m)) for p, m in
                     zip(sigmoid(pred["doc_logit"].to_numpy(float) / doc_temperature), pred["max_pixel"])])


def select_thresholds(pred: pd.DataFrame, doc_temperature: float, target_fpr: float, joint: bool = True
                      ) -> dict[str, float]:
    """Val'de (belge eşiği, piksel eşiği) seçimi: FPR ≤ ``target_fpr`` koşulunda TPR en büyük.

    ``joint=False`` → eski tek eşik (belge eşiği = piksel eşiği; ızgaranın köşegeni). Hiçbir aday hedefi
    tutturamazsa FPR'si en düşük aday seçilir ve uyarı verilir.
    """
    labels = pred["label"].to_numpy().astype(int)
    probs, regions = fused_doc_probs(pred, doc_temperature), pred.attrs["regions"]
    if joint:
        best = select_operating_point(probs, pred["max_pixel"].to_numpy(float), regions, labels, THRESHOLDS,
                                      target_fpr)
    else:
        # Köşegen: her eşik için 1x1 ızgara; en iyi aday aynı sıralama kuralıyla seçilir.
        rows = [dict(select_operating_point(probs, pred["max_pixel"].to_numpy(float), regions[:, [ti]], labels,
                                            THRESHOLDS[[ti]], target_fpr), ti=ti) for ti in range(len(THRESHOLDS))]
        feasible = [r for r in rows if r["feasible"]]
        pool = feasible or rows
        key = (lambda r: (-r["tpr"], r["fpr"], -r["ti"])) if feasible else (lambda r: (r["fpr"], -r["tpr"], -r["ti"]))
        pick = min(pool, key=key)
        best = {"doc_index": pick["ti"], "pixel_index": pick["ti"], "fpr": pick["fpr"], "tpr": pick["tpr"],
                "feasible": bool(feasible)}
    doc_thr, pix_thr = float(THRESHOLDS[best["doc_index"]]), float(THRESHOLDS[best["pixel_index"]])
    if not best["feasible"]:
        logger.warning("Hiçbir eşik çifti val'de FPR ≤ %.2f sağlamıyor; en düşük FPR'li çift seçildi "
                       "(belge eşiği=%.2f, piksel eşiği=%.2f, FPR=%.3f, TPR=%.3f). Model yeniden eğitilmeli.",
                       target_fpr, doc_thr, pix_thr, best["fpr"], best["tpr"])
    return {"doc_threshold": doc_thr, "pixel_threshold": pix_thr, "fpr": best["fpr"], "tpr": best["tpr"]}


def has_pixel_masks(frame: pd.DataFrame) -> bool:
    """Val'de piksel maskesi olan (boş olmayan ``mask_path``) sahte görüntü var mı?"""
    if "mask_path" not in frame:
        return False
    forged = frame[frame["label"].astype(int) == 1]
    return bool((forged["mask_path"].fillna("").astype(str).str.strip() != "").any())


def resolve_pixel_temperature(option: str, doc_temperature: float, val_pred: pd.DataFrame, masks: bool) -> float:
    """``--pixel-temperature`` seçeneğini sayıya çevirir (``fit``: val sahtelerinde piksel NLL)."""
    option = str(option).strip().lower()
    if option == "doc":
        return doc_temperature
    if option != "fit":
        value = float(option)
        if value <= 0:
            raise ValueError(f"Geçersiz piksel sıcaklığı: {value}")
        return value
    nll = val_pred.attrs.get("pixel_nll")
    if not masks or nll is None or not nll[1].any():
        logger.warning("Val'de piksel maskesi olan sahte yok; piksel sıcaklığı 1'de sabit kaldı.")
        return 1.0
    return fit_temperature(nll[0], nll[1])


# ---------------------------------------------------------------------------
# Sahtecilik türüne göre kırılım
# ---------------------------------------------------------------------------

def _changed_digits(edits_json: str) -> int:
    """``forge_real`` düzenleme kaydındaki değiştirilen rakam sayısı (eski/yeni metin karşılaştırması)."""
    try:
        edits = json.loads(edits_json) if edits_json else []
    except (TypeError, ValueError):
        return 0
    return sum(sum(a != b for a, b in zip(e.get("old", ""), e.get("new", "")) if a.isdigit() or b.isdigit())
               for e in edits)


def enrich_predictions(pred: pd.DataFrame, data_root: str | Path) -> pd.DataFrame:
    """Tahmin tablosuna kırılım sütunlarını ekler: sahtecilik türü, kurum ve (varsa) metadata ayrıntıları.

    Metadata, görüntü yolundaki veri seti klasörünün adıyla ``data_root`` altında aranır; böylece
    Kaggle yollarıyla yazılmış tahminler yerel veri setleriyle eşleşir.
    """
    out = pred.copy()
    paths = out["image_path"].astype(str).str.replace("\\", "/", regex=False)
    out["_dataset"] = paths.str.split("/").str[-3]
    out["_name"] = paths.str.split("/").str[-1]
    out["tamper_type"] = [m.group(1) if (m := _TAMPER_RE.search(p)) else ("orig" if int(l) == 0 else "bilinmiyor")
                          for p, l in zip(paths, out["label"])]
    group = out["group_id"].astype(str) if "group_id" in out else pd.Series("", index=out.index)
    out["issuer"] = np.where(group.str.startswith("real:"), group.str[5:].str.split("/").str[0], None)

    extra_cols = ("doc_type", "acquisition", "channel", "edits")
    frames = []
    for dataset in out["_dataset"].dropna().unique():
        meta_path = Path(data_root) / dataset / "metadata.csv"
        if not meta_path.is_file():
            continue
        meta = pd.read_csv(meta_path, keep_default_na=False)
        meta["_name"] = meta["image"].astype(str).str.split("/").str[-1]
        meta["_dataset"] = dataset
        frames.append(meta[["_dataset", "_name", *[c for c in extra_cols if c in meta]]])
    if frames:
        meta = pd.concat(frames, ignore_index=True).drop_duplicates(["_dataset", "_name"])
        out = out.drop(columns=[c for c in extra_cols if c in out]).merge(meta, on=["_dataset", "_name"], how="left")
        if "edits" in out:
            # Gerçek belgelerde rakam değişikliği tek/çok rakam diye ayrılır (tek rakam en zor durum).
            digits = out["edits"].fillna("").map(_changed_digits)
            is_digit = out["tamper_type"] == "digit_change"
            out.loc[is_digit, "tamper_type"] = np.where(digits[is_digit] <= 1, "digit_change_1", "digit_change_2+")
    real = out["issuer"].notna()
    if "doc_type" in out:
        out.loc[real, "doc_type"] = None  # Gerçek belgelerde doc_type = kurum; ayrı boyut olarak raporlanır.
    return out.drop(columns=["_dataset", "_name", "edits"], errors="ignore")


def breakdown(pred: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """Kırılım tablosu: her boyut değeri için örnek sayıları, tespit oranı, yanlış alarm ve AUC.

    * ``sahtecilik_turu``: her tür, kümedeki **tüm** orijinallere karşı (AUC) ve kendi tespit oranıyla;
      ``orig`` satırı kümenin yanlış alarm oranıdır.
    * Diğer boyutlar (kurum, kanal ...): değer içindeki orijinal ve sahtelerle hesaplanır.

    Karar sütunu ``forged_decision`` (seçilen eşikte ``predict.is_forged``) kullanılır.
    """
    labels = pred["label"].astype(int).to_numpy()
    probs = sigmoid(pred["doc_logit"].to_numpy(float) / temperature)
    decided = pred["forged_decision"].astype(int).to_numpy()
    orig = labels == 0
    rows = []
    for col, dim in BREAKDOWN_DIMS.items():
        if col not in pred or pred[col].isna().all():
            continue
        values = pred[col].fillna("").astype(str).to_numpy()
        for value in sorted(set(values) - {""}):
            sel = values == value
            if col == "tamper_type":
                if value == "orig":
                    neg, pos = sel & orig, np.zeros_like(sel)
                else:
                    neg, pos = orig, sel & ~orig
            else:
                neg, pos = sel & orig, sel & ~orig
            both = neg | pos
            rows.append({
                "boyut": dim, "deger": value, "n_orijinal": int(neg.sum()), "n_sahte": int(pos.sum()),
                "tespit_orani": float(decided[pos].mean()) if pos.any() else np.nan,
                "yanlis_alarm": float(decided[neg].mean()) if neg.any() and (col != "tamper_type" or value == "orig")
                else np.nan,
                "auc": roc_auc(probs[both], labels[both]) if pos.any() and neg.any() else np.nan,
                "ort_belge_olasiligi": float(probs[pos].mean()) if pos.any() else float(probs[neg].mean()),
            })
    return pd.DataFrame(rows)


def log_breakdown(key: str, table: pd.DataFrame) -> None:
    if table.empty:
        return
    logger.info("Kırılım — %s:\n%s", key, table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def breakdown_from_files(out_dir: Path, data_root: str) -> None:
    """``--breakdown-only``: mevcut ``predictions_*.csv`` dosyalarından kırılım üretir."""
    summary_path = out_dir / "summary.json"
    meta = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    temperature = float(meta.get("doc_temperature", meta.get("temperature", 1.0)))
    files = sorted(out_dir.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"{out_dir} altında predictions_*.csv yok; önce evaluate.py'yi modelle çalıştırın.")
    for path in files:
        pred = pd.read_csv(path, keep_default_na=False)
        if "forged_decision" not in pred:
            logger.warning("%s: forged_decision sütunu yok (eski evaluate.py çıktısı), atlanıyor.", path.name)
            continue
        pred = pred.replace("", np.nan)
        table = breakdown(enrich_predictions(pred, data_root), temperature)
        key = path.stem.removeprefix("predictions_")
        table.to_csv(out_dir / f"breakdown_{key}.csv", index=False)
        log_breakdown(key, table)
    logger.info("Kırılımlar yazıldı: %s (T=%.3f)", out_dir, temperature)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.breakdown_only:
        breakdown_from_files(Path(args.out), args.data_root)
        return
    if not args.splits or not (args.checkpoint or args.external_maps):
        raise SystemExit("--splits ile birlikte --checkpoint ya da --external-maps zorunludur "
                         "(yalnızca kırılım için --breakdown-only kullanın).")
    splits = pd.read_csv(args.splits, keep_default_na=False)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.external_maps:
        predictor = Predictor(maps_dir=args.external_maps, scores_csv=args.external_scores)
        config: dict = {}
        args.no_calibration = True  # Dış modelin val haritaları genelde yoktur; eşik --threshold ile verilir.
        logger.info("Dış model haritaları: %s (eşik %.2f, T=1)", args.external_maps, args.threshold)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, config = load_model(args.checkpoint, device)
        predictor = Predictor(model, config)

    if "aiforgedoc" in config.get("sources", []) and (splits["source"] == "sroie2019").any():
        logger.warning("Model AIForge-Doc ile eğitildi; AIForge-Doc SROIE fişlerinden türetildiği için "
                       "örtüşen SROIE dosyaları splits.csv'de zaten çıkarılmıştır.")

    # --- Kalibrasyon: yalnızca val (belge/piksel sıcaklığı + eşik çifti) ----------------
    pixel_t_default = args.temperature if args.temperature is not None else 1.0
    cal = Calibration(doc_temperature=args.temperature if args.temperature is not None else 1.0,
                      doc_threshold=args.threshold,
                      pixel_temperature=pixel_t_default if str(args.pixel_temperature).lower() in ("fit", "doc")
                      else float(args.pixel_temperature),
                      pixel_threshold=args.pixel_threshold if args.pixel_threshold is not None else args.threshold)
    calibration_source = "komut satırı (öğrenilmedi)"
    if args.from_calibration:
        if not args.checkpoint and not args.calibration_file:
            raise SystemExit("--from-calibration için --checkpoint ya da --calibration-file gerekir.")
        cal = load_calibration_info(args.calibration_file, args.checkpoint)
        calibration_source = f"{cal.path.as_posix()} ({'eski tek-T şema' if cal.legacy else 'ayrı kalibrasyon'})"
        logger.info("Mevcut kalibrasyon uygulanıyor: %s — %s", calibration_source, cal.summary())
        if not cal.calibrated:
            logger.warning("Bu kalibrasyon val'de öğrenilmemiş (elle girilmiş/varsayılan): sonuçlar 'kalibre "
                           "edilmemiş' olarak işaretlenecek.")
    elif not args.no_calibration:
        val = splits[splits["split"] == "val"]
        if args.max_images:
            # Sınıf başına üst sınır: eşik seçimi için iki sınıf da temsil edilmeli.
            val = pd.concat([g.sample(min(len(g), args.max_images), random_state=0) for _, g in val.groupby("label")])
        if val["label"].nunique() == 2:
            masks = has_pixel_masks(val)
            logger.info("Sıcaklık ölçekleme val kümesinde öğreniliyor (%d görüntü)...", len(val))
            val_pred = predict_frame(predictor, val, 1.0, nll_samples=args.pixel_samples or 2000)
            if args.temperature is None:
                cal.doc_temperature = fit_temperature(val_pred["doc_logit"].to_numpy(), val_pred["label"].to_numpy())
                logger.info("Öğrenilen belge sıcaklığı T_doc = %.3f", cal.doc_temperature)
            else:
                logger.info("Sabit belge sıcaklığı T_doc = %.3f (--temperature)", cal.doc_temperature)
            cal.pixel_temperature = resolve_pixel_temperature(args.pixel_temperature, cal.doc_temperature, val_pred,
                                                              masks)
            logger.info("Piksel sıcaklığı T_pix = %.3f (--pixel-temperature %s)", cal.pixel_temperature,
                        args.pixel_temperature)
            if abs(cal.pixel_temperature - 1.0) > 1e-6:
                logger.info("Eşik seçimi için val, T_pix=%.3f piksel haritalarıyla yeniden işleniyor...",
                            cal.pixel_temperature)
                val_pred = predict_frame(predictor, val, cal.pixel_temperature)
            if not masks:
                logger.warning("Val'de piksel maskesi olan sahte yok: piksel tarafı T=1 ve tek eşik "
                               "(belge eşiği = piksel eşiği; eski davranış) kullanılıyor.")
            chosen = select_thresholds(val_pred, cal.doc_temperature, args.target_fpr, joint=masks)
            cal.doc_threshold, cal.pixel_threshold = chosen["doc_threshold"], chosen["pixel_threshold"]
            cal.target_fpr, cal.val_fpr, cal.val_tpr = args.target_fpr, chosen["fpr"], chosen["tpr"]
            cal.val_images = len(val_pred)
            cal.checkpoint = Path(args.checkpoint).as_posix() if args.checkpoint else None
            logger.info("Seçilen eşikler: belge=%.2f, piksel=%.2f (val: FPR=%.3f, TPR=%.3f, hedef FPR ≤ %.2f)",
                        cal.doc_threshold, cal.pixel_threshold, chosen["fpr"], chosen["tpr"], args.target_fpr)
            if args.checkpoint:
                calib_path = calibration_path_for(args.checkpoint)
                calib_path.write_text(json.dumps(cal.to_dict(), indent=2), encoding="utf-8")
                cal.path = calib_path
                calibration_source = f"{calib_path.as_posix()} (val'de öğrenildi)"
                logger.info("Kalibrasyon yazıldı: %s", calib_path)
        else:
            # Val yoksa mevcut kalibrasyon dosyası ezilmez (eski sürüm T=1/0.5 yazıyordu).
            logger.warning("Val kümesinde iki sınıf yok; sıcaklık ve eşik öğrenilemedi, kalibrasyon dosyası "
                           "YAZILMADI. Mevcut kalibrasyonu uygulamak için --from-calibration kullanın.")

    # --- Test ve OOD kümeleri ------------------------------------------------------
    doc_ti, pix_ti = threshold_index(cal.doc_threshold), threshold_index(cal.pixel_threshold)
    # Ana harita kalibre piksel sıcaklığında; belge T'si ve 1 yalnızca "tek T (eski)" ablasyonu için.
    temps = tuple(dict.fromkeys([cal.pixel_temperature, cal.doc_temperature, 1.0])) if predictor.model is not None \
        else (1.0,)
    report: dict[str, dict] = {}
    for split_name in args.sets:
        subset = splits[splits["split"] == split_name]
        for source, frame in subset.groupby("source"):
            if args.max_images:
                frame = pd.concat([g.sample(min(len(g), args.max_images), random_state=0)
                                   for _, g in frame.groupby("label")])
            key = f"{split_name}/{source}"
            logger.info("Değerlendiriliyor: %s (%d görüntü)", key, len(frame))
            pred = predict_frame(predictor, frame, temps, cal.pixel_threshold, args.pixel_samples,
                                 copymove=args.copymove)
            if args.ocr:
                logger.info("OCR kural motoru çalışıyor (%d görüntü)...", len(pred))
                attrs = pred.attrs
                pred = pd.concat([pred, ocr_risks(pred)], axis=1)
                pred.attrs = attrs
            labels = pred["label"].to_numpy().astype(int)
            decided = decisions_at(pred, cal.doc_temperature, doc_ti, pix_ti, cal.min_region_fraction)
            pred["forged_decision"] = decided.astype(int)
            confusion = binary_decision_metrics(decided, labels)
            metrics = {**pred.attrs["pixel"],
                       **image_level_metrics(pred["doc_logit"].to_numpy(), labels, cal.doc_temperature),
                       # Uygulamanın verdiği karar (is_forged, seçilen eşikler): kullanıcının gördüğü başarım.
                       "decision_acc": confusion["accuracy"],
                       "decision_tpr": float(decided[labels == 1].mean()) if (labels == 1).any() else float("nan"),
                       "decision_fpr": float(decided[labels == 0].mean()) if (labels == 0).any() else float("nan"),
                       "decision_precision": confusion["precision"], "decision_f1": confusion["f1"],
                       **{f"decision_{k}": confusion[k] for k in ("tp", "fp", "fn", "tn")},
                       "n_images": len(pred), "n_forged": int(labels.sum())}
            metrics.update(confidence_intervals(pred, decided, cal.doc_temperature, args.bootstrap))
            if pred["label"].nunique() == 2:
                metrics["decision_balanced_acc"] = (metrics["decision_tpr"] + 1 - metrics["decision_fpr"]) / 2
                _, table = expected_calibration_error(sigmoid(pred["doc_logit"] / cal.doc_temperature), pred["label"])
                table.to_csv(out_dir / f"reliability_{split_name}_{source}.csv", index=False)
            save_arrays(out_dir / f"arrays_{split_name}_{source}.npz", pred)
            if args.error_maps > 0:
                save_error_maps(predictor, pred, decided, cal, out_dir / "error_maps" / f"{split_name}_{source}",
                                args.error_maps)
            pred = enrich_predictions(pred, args.data_root)
            pred.to_csv(out_dir / f"predictions_{split_name}_{source}.csv", index=False)
            table = breakdown(pred, cal.doc_temperature)
            table.to_csv(out_dir / f"breakdown_{split_name}_{source}.csv", index=False)
            log_breakdown(key, table)
            report[key] = metrics
            logger.info("%s → %s", key, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    summary = pd.DataFrame(report).T
    summary.to_csv(out_dir / "summary.csv")
    source = {"checkpoint": Path(args.checkpoint).as_posix() if args.checkpoint else None,
              "external_maps": args.external_maps, "splits": Path(args.splits).as_posix(),
              "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    calib_fields = {k: v for k, v in cal.to_dict().items() if k not in ("checkpoint", "optimal_threshold")}
    # "temperature" / "threshold": eski okuyucular (dashboard, ablation_decision) için belge tarafı.
    (out_dir / "summary.json").write_text(json.dumps({
        **source, **calib_fields, "threshold": cal.doc_threshold, "calibrated": cal.calibrated,
        "calibration_source": calibration_source, "results": report}, indent=2, default=float), encoding="utf-8")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
