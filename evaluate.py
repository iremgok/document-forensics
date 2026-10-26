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

Karar eşiği
-----------
Uygulama bir belgeye ``predict.is_forged`` kuralıyla (belge skoru + piksel bölgesi) "sahte" der.
Eşik, val kümesinde bu kuralın **orijinal belgelerdeki yanlış alarm oranı** ``--target-fpr``'ı
(varsayılan %5) aşmayacak şekilde, bu koşulda sahte belgeleri en çok yakalayan değer olarak seçilir
ve sıcaklıkla birlikte ``<checkpoint klasörü>/calibration.json``'a yazılır. Uygulama ve
``predict.py`` bu dosyayı modelin kendi klasöründen okur.

Piksel olasılıkları uygulamadaki gibi sıcaklıkla ölçeklendiği için sıcaklık öğrenildikten sonra
val kümesi o sıcaklıkla bir kez daha işlenir (T = 1 ise tekrar edilmez).

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

from predict import decide_forged, fuse_doc_score, largest_region_fraction
from src.dataset import ForgeryDataset, build_transforms, image_key, read_image_rgb, read_mask
from src.explain import error_map
from src.metrics import (PixelMetrics, binary_decision_metrics, bootstrap_ci, expected_calibration_error,
                         fit_temperature, image_level_metrics, roc_auc, sigmoid, tpr_at_fpr)
from src.models import load_model, sliding_window_predict

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
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Kalibrasyon yapılmadığında (dış model / --no-calibration) kullanılan karar eşiği")
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
    def __call__(self, image: torch.Tensor, path: str, temperature: float = 1.0
                 ) -> tuple[np.ndarray, float, np.ndarray]:
        if self.model is not None:
            return sliding_window_predict(self.model, image, tile=self.config.get("crop_size", 512),
                                          temperature=temperature, return_windows=True)
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
        doc_logit = float(np.log(score / (1 - score)))
        return prob_map, doc_logit, np.array([doc_logit], dtype=np.float32)


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


def predict_frame(predictor: Predictor, frame: pd.DataFrame, temperature: float = 1.0,
                  pixel_threshold: float = 0.5, pixel_samples: int = 0) -> pd.DataFrame:
    """Her görüntü için belge logit'ini, karar özetlerini ve piksel metriklerini hesaplar.

    Piksel haritası uygulamadaki gibi ``temperature`` ile ölçeklenir. ``out.attrs``:

    * ``regions``: her görüntü için ``THRESHOLDS`` eşiklerindeki en büyük bölge oranları (N, len(THRESHOLDS)),
    * ``windows``: görüntü başına pencere ``cls_logit``'leri (havuzlama ablasyonu),
    * ``pixel`` / ``pixel_obj``: piksel metrikleri ve görüntü başına TP/FP/FN taşıyan nesne.
    """
    dataset = ForgeryDataset(frame, build_transforms(False), predictor.max_side)
    pixel = PixelMetrics(threshold=pixel_threshold, sample_pixels=pixel_samples)
    logits, max_pixels, regions, windows = [], [], [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        prob_map, doc_logit, window_logits = predictor(sample["image"], frame.iloc[i]["image_path"], temperature)
        pixel.update(prob_map, sample["mask"][0].numpy())
        logits.append(doc_logit)
        max_pixels.append(float(prob_map.max()))
        regions.append([largest_region_fraction(prob_map, t) for t in THRESHOLDS])
        windows.append(window_logits)
        if (i + 1) % 100 == 0:
            logger.info("  %d / %d", i + 1, len(dataset))
    out = frame.reset_index(drop=True).copy()
    out["doc_logit"] = logits
    out["max_pixel"] = max_pixels
    counts = np.asarray(pixel.counts, dtype=np.float64).reshape(len(out), 3)
    out["px_tp"], out["px_fp"], out["px_fn"] = counts[:, 0], counts[:, 1], counts[:, 2]
    out.attrs["pixel"] = pixel.compute()
    out.attrs["pixel_obj"] = pixel
    out.attrs["regions"] = np.asarray(regions, dtype=np.float64).reshape(len(out), len(THRESHOLDS))
    out.attrs["windows"] = windows
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
    """Ablasyon için görüntü başına eşik taramalı bölge oranları ve pencere logit'leri."""
    windows = np.empty(len(pred), dtype=object)
    windows[:] = pred.attrs["windows"]
    np.savez_compressed(path, thresholds=THRESHOLDS, regions=pred.attrs["regions"], windows=windows,
                        image_path=pred["image_path"].astype(str).to_numpy())


def save_error_maps(predictor: Predictor, pred: pd.DataFrame, decided: np.ndarray, temperature: float,
                    threshold: float, out_dir: Path, n: int) -> None:
    """En belirgin hatalar ve doğrular için TP/FP/FN hata haritaları (yeşil/kırmızı/mavi)."""
    labels = pred["label"].to_numpy().astype(int)
    probs = sigmoid(pred["doc_logit"].to_numpy(float) / temperature)
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
            prob_map, _, _ = predictor(sample["image"], row["image_path"], temperature)
            if prob_map.shape != image.shape[:2]:  # max_side küçültmesi: görselleştirmede orijinal boyut
                prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            vis = error_map(image, prob_map, gt, threshold)
            name_out = f"{name}_{rank:02d}_{image_key(row['image_path'])}_p{probs[i]:.2f}.jpg"
            cv2.imwrite(str(out_dir / name_out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def decisions_at(pred: pd.DataFrame, temperature: float, ti: int) -> np.ndarray:
    """``THRESHOLDS[ti]`` eşiğinde her görüntü için ``predict.is_forged`` kararı."""
    threshold = float(THRESHOLDS[ti])
    doc_probs = sigmoid(pred["doc_logit"].to_numpy() / temperature)
    return np.array([decide_forged(fuse_doc_score(float(p), m), m, r, threshold)
                     for p, m, r in zip(doc_probs, pred["max_pixel"], pred.attrs["regions"][:, ti])])


def select_threshold(pred: pd.DataFrame, temperature: float, target_fpr: float) -> dict[str, float]:
    """Val'de FPR ≤ ``target_fpr`` koşulunda TPR'si en yüksek eşik (eşitlikte küçük eşik).

    Hiçbir eşik hedefi tutturamazsa FPR'si en düşük eşik seçilir ve uyarı verilir.
    """
    labels = pred["label"].to_numpy().astype(int)
    rows = []
    for ti, threshold in enumerate(THRESHOLDS):
        decided = decisions_at(pred, temperature, ti)
        rows.append({"threshold": float(threshold), "fpr": float(decided[labels == 0].mean()),
                     "tpr": float(decided[labels == 1].mean())})
    table = pd.DataFrame(rows)
    feasible = table[table["fpr"] <= target_fpr]
    if feasible.empty:
        best = table.sort_values(["fpr", "tpr"], ascending=[True, False]).iloc[0]
        logger.warning("Hiçbir eşik val'de FPR ≤ %.2f sağlamıyor; en düşük FPR'li eşik seçildi "
                       "(eşik=%.2f, FPR=%.3f, TPR=%.3f). Model yeniden eğitilmeli.",
                       target_fpr, best["threshold"], best["fpr"], best["tpr"])
    else:
        best = feasible.sort_values(["tpr", "threshold"], ascending=[False, True]).iloc[0]
    return {"threshold": float(best["threshold"]), "fpr": float(best["fpr"]), "tpr": float(best["tpr"])}


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
    temperature = json.loads(summary_path.read_text(encoding="utf-8")).get("temperature", 1.0) \
        if summary_path.is_file() else 1.0
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

    # --- Kalibrasyon: yalnızca val (sıcaklık + karar eşiği) ----------------------------
    temperature, threshold = 1.0, args.threshold
    calibration: dict[str, float] = {}
    if not args.no_calibration:
        val = splits[splits["split"] == "val"]
        if args.max_images:
            # Sınıf başına üst sınır: eşik seçimi için iki sınıf da temsil edilmeli.
            val = pd.concat([g.sample(min(len(g), args.max_images), random_state=0) for _, g in val.groupby("label")])
        if val["label"].nunique() == 2:
            logger.info("Sıcaklık ölçekleme val kümesinde öğreniliyor (%d görüntü)...", len(val))
            val_pred = predict_frame(predictor, val)
            temperature = fit_temperature(val_pred["doc_logit"].to_numpy(), val_pred["label"].to_numpy())
            logger.info("Öğrenilen sıcaklık T = %.3f", temperature)
            if abs(temperature - 1.0) > 1e-6:
                logger.info("Eşik seçimi için val, T=%.3f piksel haritalarıyla yeniden işleniyor...", temperature)
                val_pred = predict_frame(predictor, val, temperature)
            chosen = select_threshold(val_pred, temperature, args.target_fpr)
            threshold = chosen["threshold"]
            calibration = {"target_fpr": args.target_fpr, "val_fpr": chosen["fpr"], "val_tpr": chosen["tpr"],
                           "val_images": len(val_pred)}
            logger.info("Seçilen karar eşiği = %.2f (val: FPR=%.3f, TPR=%.3f, hedef FPR ≤ %.2f)",
                        threshold, chosen["fpr"], chosen["tpr"], args.target_fpr)
        else:
            logger.warning("Val kümesinde iki sınıf yok; sıcaklık ve eşik öğrenilemedi (T=1, eşik=0.5).")
        calib_path = Path(args.checkpoint).parent / "calibration.json"
        calib_path.write_text(json.dumps({"temperature": temperature, "optimal_threshold": threshold, **calibration},
                                         indent=2), encoding="utf-8")
        logger.info("Kalibrasyon yazıldı: %s", calib_path)

    # --- Test ve OOD kümeleri ------------------------------------------------------
    ti = int(np.argmin(np.abs(THRESHOLDS - threshold)))
    report: dict[str, dict] = {}
    for split_name in args.sets:
        subset = splits[splits["split"] == split_name]
        for source, frame in subset.groupby("source"):
            if args.max_images:
                frame = pd.concat([g.sample(min(len(g), args.max_images), random_state=0)
                                   for _, g in frame.groupby("label")])
            key = f"{split_name}/{source}"
            logger.info("Değerlendiriliyor: %s (%d görüntü)", key, len(frame))
            pred = predict_frame(predictor, frame, temperature, threshold, args.pixel_samples)
            if args.ocr:
                logger.info("OCR kural motoru çalışıyor (%d görüntü)...", len(pred))
                attrs = pred.attrs
                pred = pd.concat([pred, ocr_risks(pred)], axis=1)
                pred.attrs = attrs
            labels = pred["label"].to_numpy().astype(int)
            decided = decisions_at(pred, temperature, ti)
            pred["forged_decision"] = decided.astype(int)
            confusion = binary_decision_metrics(decided, labels)
            metrics = {**pred.attrs["pixel"],
                       **image_level_metrics(pred["doc_logit"].to_numpy(), labels, temperature),
                       # Uygulamanın verdiği karar (is_forged, seçilen eşik): kullanıcının gördüğü başarım.
                       "decision_acc": confusion["accuracy"],
                       "decision_tpr": float(decided[labels == 1].mean()) if (labels == 1).any() else float("nan"),
                       "decision_fpr": float(decided[labels == 0].mean()) if (labels == 0).any() else float("nan"),
                       "decision_precision": confusion["precision"], "decision_f1": confusion["f1"],
                       **{f"decision_{k}": confusion[k] for k in ("tp", "fp", "fn", "tn")},
                       "n_images": len(pred), "n_forged": int(labels.sum())}
            metrics.update(confidence_intervals(pred, decided, temperature, args.bootstrap))
            if pred["label"].nunique() == 2:
                metrics["decision_balanced_acc"] = (metrics["decision_tpr"] + 1 - metrics["decision_fpr"]) / 2
                _, table = expected_calibration_error(sigmoid(pred["doc_logit"] / temperature), pred["label"])
                table.to_csv(out_dir / f"reliability_{split_name}_{source}.csv", index=False)
            save_arrays(out_dir / f"arrays_{split_name}_{source}.npz", pred)
            if args.error_maps > 0:
                save_error_maps(predictor, pred, decided, temperature, threshold,
                                out_dir / "error_maps" / f"{split_name}_{source}", args.error_maps)
            pred = enrich_predictions(pred, args.data_root)
            pred.to_csv(out_dir / f"predictions_{split_name}_{source}.csv", index=False)
            table = breakdown(pred, temperature)
            table.to_csv(out_dir / f"breakdown_{split_name}_{source}.csv", index=False)
            log_breakdown(key, table)
            report[key] = metrics
            logger.info("%s → %s", key, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    summary = pd.DataFrame(report).T
    summary.to_csv(out_dir / "summary.csv")
    (out_dir / "summary.json").write_text(json.dumps({"temperature": temperature, "threshold": threshold,
                                                      **calibration, "results": report},
                                                     indent=2, default=float), encoding="utf-8")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
