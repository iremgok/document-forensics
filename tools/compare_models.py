"""
Birden fazla modelin (kendi aşamalarınız + TruFor/DTD gibi baseline'lar) ``evaluate.py`` özetlerini tek
karşılaştırma tablosunda birleştirir; güven aralıklarını "değer [alt–üst]" biçiminde yazar.

Kullanım:
    python tools/compare_models.py \\
        --run "Stage A=results/stageA" --run "Stage E=results/stageE" \\
        --run "TruFor=results/baselines/trufor" --run "DTD=results/baselines/dtd" \\
        --metrics image_auc image_tpr_at_fpr02 decision_fpr decision_f1 pixel_f1 pixel_iou \\
        --out results/comparison
Çıktı: ``<out>.csv`` (sayısal) ve ``<out>.md`` (rapora yapıştırılabilir Markdown tablosu).

Aynı test protokolüyle (aynı ``splits`` dosyası) üretilmemiş özetler karşılaştırılmamalıdır; betik
kümelerin görüntü sayıları (``n_images``) farklıysa uyarır.

Her satırda modelin kalibrasyon durumu yazılır (``kalibrasyon`` sütunu): ``summary.json`` → ``calibrated``
(``evaluate.py`` yeni sürüm) ya da checkpoint'in kalibrasyon dosyasından (``src/model_registry.py``). Val'de
öğrenilmemiş kalibrasyonla ölçülen modeller **"kalibre edilmemiş"** olarak işaretlenir; karar metrikleri
(TPR/FPR/F1) bu modellerde doğrulanmış bir çalışma noktasına dayanmaz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_registry import UNCALIBRATED_LABEL, calibration_label  # noqa: E402

DEFAULT_METRICS = ["image_auc", "image_tpr_at_fpr02", "decision_tpr", "decision_fpr", "decision_f1",
                   "pixel_f1", "pixel_iou", "pixel_precision", "pixel_recall"]


def fmt(value: float, lo: float = np.nan, hi: float = np.nan) -> str:
    if value is None or value != value:
        return "—"
    text = f"{value:.3f}"
    if lo == lo and hi == hi:
        text += f" [{lo:.2f}–{hi:.2f}]"
    return text


def main() -> None:
    p = argparse.ArgumentParser(description="evaluate.py özetlerini karşılaştırma tablosunda birleştir")
    p.add_argument("--run", action="append", required=True, help="'Ad=results/klasör' (birden fazla)")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--out", default="results/comparison")
    args = p.parse_args()

    records = []
    for spec in args.run:
        name, folder = spec.split("=", 1)
        summary = json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
        if "calibrated" in summary:
            status = "kalibre" if summary["calibrated"] else UNCALIBRATED_LABEL
        else:
            status = calibration_label(summary.get("checkpoint"))
        for key, metrics in summary["results"].items():
            records.append({"model": name, "kume": key, "kalibrasyon": status,
                            "belge_T": summary.get("doc_temperature", summary.get("temperature")),
                            "belge_esik": summary.get("doc_threshold", summary.get("threshold")),
                            "piksel_T": summary.get("pixel_temperature", summary.get("temperature")),
                            "piksel_esik": summary.get("pixel_threshold", summary.get("threshold")), **metrics})
    table = pd.DataFrame(records)
    sizes = table.groupby("kume")["n_images"].nunique()
    for key in sizes[sizes > 1].index:
        print(f"UYARI: '{key}' kümesinde modeller farklı sayıda görüntüyle ölçülmüş "
              f"({sorted(table.loc[table.kume == key, 'n_images'].unique())}); aynı splits dosyasını kullanın.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    lines = []
    for key, group in table.groupby("kume", sort=False):
        lines += [f"### {key} (n = {int(group['n_images'].iloc[0])})", "",
                  "| Model | Kalibrasyon | " + " | ".join(args.metrics) + " |",
                  "|---|---|" + "---|" * len(args.metrics)]
        for _, row in group.iterrows():
            cells = [fmt(row.get(m, np.nan), row.get(f"{m}_ci_lo", np.nan), row.get(f"{m}_ci_hi", np.nan))
                     for m in args.metrics]
            lines.append(f"| {row['model']} | {row['kalibrasyon']} | " + " | ".join(cells) + " |")
        lines.append("")
    if (table["kalibrasyon"] == UNCALIBRATED_LABEL).any():
        lines.append(f"\"{UNCALIBRATED_LABEL}\": kalibrasyon val kümesinde öğrenilmemiş (genel/elle girilmiş T ve eşik); "
                     "karar metrikleri (TPR/FPR/F1) doğrulanmış bir çalışma noktasına dayanmaz. Eşikten bağımsız "
                     "metrikler (AUC, TPR@FPR) karşılaştırılabilir.")
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Yazıldı: {out.with_suffix('.csv')}, {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
