"""
OCR + kural motorunun kendi başına değerlendirilmesi (görüntü modelinden bağımsız).

Ölçülenler:

1. **Alan çıkarma doğruluğu** — elle etiketlenmiş belgelerde IBAN, VKN, TCKN, matrah, KDV ve toplamın
   doğru okunup okunmadığı (alan başına recall / precision).
2. **Kural bazında karar tablosu** — ``label`` sütunu varsa her kural (iban, vkn, tckn, total_math,
   vat_rate) için orijinal belgelerde yanlış alarm (FAIL) oranı ve sahte belgelerde tetiklenme oranı.
3. **OCR motoru karşılaştırması** — ``--engine easyocr`` (varsayılan, uygulamadaki) ya da ``paddle``
   (PaddleOCR 2.x API'si; ``pip install paddleocr paddlepaddle``). Süre de raporlanır.

Etiket CSV'si (``--labels``; yollar CSV klasörüne göre, boş hücre = belgede o alan yok)::

    image,label,iban,vkn,tckn,matrah,kdv,total
    gercek/garanti_01.png,0,TR330006100519786457841326,,12345678950,,,
    gercek/trendyol_03.png,0,,3130557669,,900.00,180.00,1080.00
    sahte/trendyol_03_tutar.png,1,,3130557669,,900.00,180.00,1580.00

Tutarlar nokta ondalıkla yazılır; karşılaştırmada ±0.01 tolerans kullanılır. IBAN/VKN/TCKN birden
fazlaysa ``;`` ile ayrılır.

Kullanım:
    python tools/ocr_eval.py --labels data/ocr_labels/labels.csv --out results/ocr_eval
    python tools/ocr_eval.py --labels data/ocr_labels/labels.csv --engine paddle --out results/ocr_eval_paddle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import read_image_rgb  # noqa: E402
from src.ocr_checker import FAIL, check_document_text, group_into_lines, run_ocr  # noqa: E402

LIST_FIELDS = ("iban", "vkn", "tckn")
AMOUNT_FIELDS = ("matrah", "kdv", "total")
RULES = ("iban", "vkn", "tckn", "total_math", "vat_rate")


def paddle_ocr(image: np.ndarray):
    """PaddleOCR 2.x ile satırlar (EasyOCR ile aynı OcrLine biçimi). 3.x API'si farklıdır."""
    global _PADDLE
    if "_PADDLE" not in globals():
        from paddleocr import PaddleOCR

        _PADDLE = PaddleOCR(lang="tr", use_angle_cls=False, show_log=False)
    result = _PADDLE.ocr(image[..., ::-1].copy(), cls=False) or [[]]
    words = [(box, text, conf) for box, (text, conf) in (result[0] or [])]
    return group_into_lines(words)


def split_list(value: str) -> set[str]:
    return {v.strip().replace(" ", "").upper() for v in str(value).split(";") if v.strip()}


def main() -> None:
    p = argparse.ArgumentParser(description="OCR + kural motoru değerlendirmesi")
    p.add_argument("--labels", required=True)
    p.add_argument("--engine", choices=["easyocr", "paddle"], default="easyocr")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--out", default="results/ocr_eval")
    args = p.parse_args()

    labels_path = Path(args.labels)
    gt = pd.read_csv(labels_path, keep_default_na=False, dtype=str)
    rows, timings = [], []
    for i, item in gt.iterrows():
        image = read_image_rgb(labels_path.parent / item["image"])
        start = time.time()
        lines = paddle_ocr(image) if args.engine == "paddle" else run_ocr(image, gpu=args.gpu)
        timings.append(time.time() - start)
        report = check_document_text(lines)
        row = {"image": item["image"], "label": item.get("label", ""), "ocr_s": timings[-1],
               "risk": report.risk}
        for field in LIST_FIELDS:
            truth = split_list(item.get(field, ""))
            found = {str(v).upper() for v in report.fields.get(field, [])}
            row[f"{field}_gt"], row[f"{field}_tp"] = len(truth), len(truth & found)
            row[f"{field}_pred"] = len(found)
        for field in AMOUNT_FIELDS:
            truth = item.get(field, "")
            found = report.fields.get(field)
            row[f"{field}_gt"] = int(truth != "")
            row[f"{field}_pred"] = int(found is not None)
            row[f"{field}_tp"] = int(truth != "" and found is not None and abs(float(truth) - float(found)) <= 0.01)
        for rule in RULES:
            statuses = [f.status for f in report.findings if f.rule == rule]
            row[f"rule_{rule}"] = "fail" if FAIL in statuses else ("pass" if "pass" in statuses
                                                                   else ("warn" if statuses else ""))
        rows.append(row)
        print(f"  {i + 1}/{len(gt)} {item['image']} ({timings[-1]:.1f} s)")
    detail = pd.DataFrame(rows)

    field_rows = []
    for field in LIST_FIELDS + AMOUNT_FIELDS:
        n_gt, n_pred, n_tp = (int(detail[f"{field}_{k}"].sum()) for k in ("gt", "pred", "tp"))
        field_rows.append({"alan": field, "etiketli": n_gt, "okunan": n_pred, "dogru": n_tp,
                           "recall": n_tp / n_gt if n_gt else np.nan,
                           "precision": n_tp / n_pred if n_pred else np.nan})
    fields = pd.DataFrame(field_rows)

    rule_rows = []
    if (detail["label"] != "").all():
        lab = detail["label"].astype(int)
        for rule in RULES:
            col = detail[f"rule_{rule}"]
            evaluated = col.isin(["pass", "fail"])
            rule_rows.append({
                "kural": rule,
                "orijinal_degerlendirilen": int((evaluated & (lab == 0)).sum()),
                "orijinal_yanlis_alarm": int(((col == "fail") & (lab == 0)).sum()),
                "sahte_degerlendirilen": int((evaluated & (lab == 1)).sum()),
                "sahte_tetiklenen": int(((col == "fail") & (lab == 1)).sum()),
            })
        for r in rule_rows:
            r["yanlis_alarm_orani"] = r["orijinal_yanlis_alarm"] / r["orijinal_degerlendirilen"] \
                if r["orijinal_degerlendirilen"] else np.nan
            r["tetiklenme_orani"] = r["sahte_tetiklenen"] / r["sahte_degerlendirilen"] \
                if r["sahte_degerlendirilen"] else np.nan
    rules = pd.DataFrame(rule_rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "detail.csv", index=False)
    fields.to_csv(out / "fields.csv", index=False)
    if not rules.empty:
        rules.to_csv(out / "rules.csv", index=False)
    print(f"\nMotor: {args.engine} | ortalama süre {np.mean(timings):.2f} s/belge")
    print(fields.round(3).to_string(index=False))
    if not rules.empty:
        print(rules.round(3).to_string(index=False))
    print(f"Yazıldı: {out}")


if __name__ == "__main__":
    main()
