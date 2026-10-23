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

import numpy as np
import pandas as pd
import torch

from predict import decide_forged, fuse_doc_score, largest_region_fraction
from src.dataset import ForgeryDataset, build_transforms
from src.metrics import (PixelMetrics, expected_calibration_error, fit_temperature, image_level_metrics,
                         roc_auc, sigmoid)
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
    p.add_argument("--out", default="results")
    return p.parse_args()


@torch.no_grad()
def predict_frame(model: torch.nn.Module, frame: pd.DataFrame, config: dict, temperature: float = 1.0,
                  pixel_threshold: float = 0.5) -> pd.DataFrame:
    """Her görüntü için belge logit'ini, karar özetlerini ve piksel metriklerini hesaplar.

    Piksel haritası uygulamadaki gibi ``temperature`` ile ölçeklenir. ``out.attrs['regions']``,
    her görüntü için ``THRESHOLDS`` eşiklerindeki en büyük bölge oranlarıdır (N, len(THRESHOLDS)).
    """
    dataset = ForgeryDataset(frame, build_transforms(False), config.get("max_side", 1600))
    pixel = PixelMetrics(threshold=pixel_threshold)
    logits, max_pixels, regions = [], [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        prob_map, doc_logit = sliding_window_predict(model, sample["image"], tile=config.get("crop_size", 512),
                                                     temperature=temperature)
        pixel.update(prob_map, sample["mask"][0].numpy())
        logits.append(doc_logit)
        max_pixels.append(float(prob_map.max()))
        regions.append([largest_region_fraction(prob_map, t) for t in THRESHOLDS])
        if (i + 1) % 100 == 0:
            logger.info("  %d / %d", i + 1, len(dataset))
    out = frame.reset_index(drop=True).copy()
    out["doc_logit"] = logits
    out["max_pixel"] = max_pixels
    out.attrs["pixel"] = pixel.compute()
    out.attrs["regions"] = np.asarray(regions, dtype=np.float64).reshape(len(out), len(THRESHOLDS))
    return out


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
    if not args.checkpoint or not args.splits:
        raise SystemExit("--checkpoint ve --splits zorunludur (yalnızca kırılım için --breakdown-only kullanın).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    splits = pd.read_csv(args.splits, keep_default_na=False)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "aiforgedoc" in config.get("sources", []) and (splits["source"] == "sroie2019").any():
        logger.warning("Model AIForge-Doc ile eğitildi; AIForge-Doc SROIE fişlerinden türetildiği için "
                       "örtüşen SROIE dosyaları splits.csv'de zaten çıkarılmıştır.")

    # --- Kalibrasyon: yalnızca val (sıcaklık + karar eşiği) ----------------------------
    temperature, threshold = 1.0, 0.5
    calibration: dict[str, float] = {}
    if not args.no_calibration:
        val = splits[splits["split"] == "val"]
        if args.max_images:
            # Sınıf başına üst sınır: eşik seçimi için iki sınıf da temsil edilmeli.
            val = pd.concat([g.sample(min(len(g), args.max_images), random_state=0) for _, g in val.groupby("label")])
        if val["label"].nunique() == 2:
            logger.info("Sıcaklık ölçekleme val kümesinde öğreniliyor (%d görüntü)...", len(val))
            val_pred = predict_frame(model, val, config)
            temperature = fit_temperature(val_pred["doc_logit"].to_numpy(), val_pred["label"].to_numpy())
            logger.info("Öğrenilen sıcaklık T = %.3f", temperature)
            if abs(temperature - 1.0) > 1e-6:
                logger.info("Eşik seçimi için val, T=%.3f piksel haritalarıyla yeniden işleniyor...", temperature)
                val_pred = predict_frame(model, val, config, temperature)
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
            pred = predict_frame(model, frame, config, temperature, threshold)
            labels = pred["label"].to_numpy().astype(int)
            decided = decisions_at(pred, temperature, ti)
            pred["forged_decision"] = decided.astype(int)
            metrics = {**pred.attrs["pixel"],
                       **image_level_metrics(pred["doc_logit"].to_numpy(), labels, temperature),
                       # Uygulamanın verdiği karar (is_forged, seçilen eşik): kullanıcının gördüğü başarım.
                       "decision_acc": float((decided == labels).mean()),
                       "decision_tpr": float(decided[labels == 1].mean()) if (labels == 1).any() else float("nan"),
                       "decision_fpr": float(decided[labels == 0].mean()) if (labels == 0).any() else float("nan"),
                       "n_images": len(pred), "n_forged": int(labels.sum())}
            if pred["label"].nunique() == 2:
                metrics["decision_balanced_acc"] = (metrics["decision_tpr"] + 1 - metrics["decision_fpr"]) / 2
                _, table = expected_calibration_error(sigmoid(pred["doc_logit"] / temperature), pred["label"])
                table.to_csv(out_dir / f"reliability_{split_name}_{source}.csv", index=False)
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
