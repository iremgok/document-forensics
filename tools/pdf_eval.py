"""
PDF yapı analizörünün (``src/pdf_checker.py``) gerçek PDF'lerde değerlendirilmesi.

Girdi iki biçimde verilebilir:

* ``--dir <klasör>``: klasördeki (alt klasörler dahil) tüm ``*.pdf`` dosyaları. Etiket ``--label`` ile
  (varsayılan 0 = orijinal) ya da yol içindeki ``orijinal`` / ``sahte`` klasör adından okunur.
* ``--metadata <metadata.csv>``: ``src/forge_real.py`` çıktısı; ``source_pdf`` sütunundaki PDF'ler. Sahte
  varyantlar rasterleştirilmiş görüntülerdir (PDF'leri yoktur), bu yüzden buradaki PDF'ler **orijinaldir**.

Her PDF'te ``check_pdf`` çalıştırılır ve şu dosyalar yazılır (``--out`` öneki):

* ``<out>_files.csv``  — dosya başına risk, FAIL/WARN sayısı ve FAIL veren kurallar,
* ``<out>_rules.csv``  — kural başına FAIL / WARN / PASS sayıları; orijinallerde o kuralın FAIL oranı,
* ``<out>_summary.csv`` — orijinallerde **yanlış alarm oranı** (en az bir FAIL alan orijinal PDF oranı),
  sahtelerde (varsa) tespit oranı ve ortalama risk.

Kullanım:
    python tools/pdf_eval.py --metadata data/8_real_forged_v2/metadata.csv --out results/pdf_eval/real_v2
    python tools/pdf_eval.py --dir data/pdf_orijinal --label 0 --out results/pdf_eval/orijinal
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ocr_checker import FAIL, PASS, WARN  # noqa: E402
from src.pdf_checker import check_pdf  # noqa: E402


def collect(args: argparse.Namespace) -> pd.DataFrame:
    """Değerlendirilecek PDF'ler: (path, label)."""
    if args.metadata:
        meta = pd.read_csv(args.metadata, keep_default_na=False)
        if "source_pdf" not in meta:
            raise SystemExit(f"{args.metadata} 'source_pdf' sütunu içermiyor (eski forge_real çıktısı). "
                             "PDF'leri saklayan yeni sürümle üretin: python -m src.forge_real --out data/8_real_forged_v2")
        root = Path(args.metadata).parent
        pdfs = sorted({p for p in meta["source_pdf"] if p})
        return pd.DataFrame({"path": [str(root / p) for p in pdfs], "label": 0})
    files = sorted(Path(args.dir).rglob("*.pdf"))
    labels = []
    for f in files:
        parts = {p.lower() for p in f.parts}
        labels.append(1 if "sahte" in parts else 0 if "orijinal" in parts else args.label)
    return pd.DataFrame({"path": [str(f) for f in files], "label": labels})


def evaluate(files: pd.DataFrame, dpi: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, statuses = [], []
    for path, label in zip(files["path"], files["label"]):
        report = check_pdf(Path(path).read_bytes(), dpi=dpi)
        fails = sorted({f.rule for f in report.findings if f.status == FAIL})
        rows.append({"path": path, "label": int(label), "analyzed": report.analyzed, "risk": report.risk,
                     "n_fail": sum(f.status == FAIL for f in report.findings),
                     "n_warn": sum(f.status == WARN for f in report.findings),
                     "fail_rules": ";".join(fails), "producer": report.info.get("producer", ""),
                     "revizyon": report.info.get("revizyon")})
        # Kural başına dosya düzeyinde en kötü durum (aynı kuraldan birden çok bulgu tek sayılır).
        worst: dict[str, str] = {}
        for f in report.findings:
            order = {FAIL: 0, WARN: 1, PASS: 2}
            if f.rule not in worst or order.get(f.status, 1) < order.get(worst[f.rule], 1):
                worst[f.rule] = f.status
        statuses += [{"path": path, "label": int(label), "rule": r, "status": s} for r, s in worst.items()]
    per_file = pd.DataFrame(rows)
    status = pd.DataFrame(statuses, columns=["path", "label", "rule", "status"])
    rules = []
    n_orig = int((per_file["label"] == 0).sum()) if len(per_file) else 0
    for rule, group in status.groupby("rule"):
        counts = Counter(group["status"])
        orig_fail = int(((group["label"] == 0) & (group["status"] == FAIL)).sum())
        rules.append({"kural": rule, "FAIL": counts.get(FAIL, 0), "WARN": counts.get(WARN, 0),
                      "PASS": counts.get(PASS, 0), "orijinal_fail_orani": orig_fail / n_orig if n_orig else float("nan")})
    summary = []
    for label, name in ((0, "orijinal"), (1, "sahte")):
        sel = per_file[per_file["label"] == label] if len(per_file) else per_file
        if len(sel):
            summary.append({"sinif": name, "n": len(sel),
                            "en_az_bir_fail_orani": float((sel["n_fail"] > 0).mean()),
                            "ort_risk": float(sel["risk"].mean()), "analiz_edilemeyen": int((~sel["analyzed"]).sum())})
    summary_df = pd.DataFrame(summary)
    if len(summary_df):
        summary_df["anlam"] = summary_df["sinif"].map({"orijinal": "yanlış alarm oranı", "sahte": "tespit oranı"})
    return per_file, pd.DataFrame(rules), summary_df


def main() -> None:
    p = argparse.ArgumentParser(description="pdf_checker değerlendirmesi (kural bazında FAIL/WARN/PASS)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="PDF klasörü (alt klasörler dahil)")
    src.add_argument("--metadata", help="forge_real metadata.csv (source_pdf sütunu)")
    p.add_argument("--label", type=int, choices=(0, 1), default=0,
                   help="--dir için varsayılan etiket (yol 'orijinal'/'sahte' klasörü içeriyorsa o kullanılır)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--out", default="results/pdf_eval/pdf_eval")
    args = p.parse_args()

    files = collect(args)
    if files.empty:
        raise SystemExit("Değerlendirilecek PDF bulunamadı.")
    per_file, rules, summary = evaluate(files, args.dpi)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_file.to_csv(f"{out}_files.csv", index=False)
    rules.to_csv(f"{out}_rules.csv", index=False)
    summary.to_csv(f"{out}_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(rules.to_string(index=False))
        print(summary.to_string(index=False))
    print(f"Yazıldı: {out}_files.csv, {out}_rules.csv, {out}_summary.csv")


if __name__ == "__main__":
    main()
