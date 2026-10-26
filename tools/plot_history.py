"""
Öğrenme eğrileri: ``train.py``'nin yazdığı ``history.csv`` dosyalarından train/val kaybı ve val metrikleri.

Aşırı öğrenme (overfitting) ve yetersiz öğrenmenin (underfitting) raporlanmasının standart yolu:

* train kaybı düşerken val kaybı yükseliyorsa → aşırı öğrenme,
* ikisi de yüksek kalıyorsa → yetersiz öğrenme,
* en iyi epoch (izlenen metriğin en iyi değeri) dikey çizgiyle işaretlenir.

Kullanım:
    python tools/plot_history.py --run "Stage A=checkpoints/stageA" --run "Stage E=checkpoints/stageE" \\
        --monitor val/score --out results/learning_curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PANELS = [("Kayıp", ["train/loss", "val/loss"]),
          ("Piksel F1 (= Dice) / IoU", ["val/pixel_f1", "val/pixel_iou"]),
          ("Belge düzeyi", ["val/image_auc", "val/image_tpr_at_fpr02", "val/image_fpr"])]


def main() -> None:
    p = argparse.ArgumentParser(description="history.csv öğrenme eğrileri")
    p.add_argument("--run", action="append", required=True, help="'Ad=checkpoints/klasör' (history.csv içeren)")
    p.add_argument("--monitor", default="val/score", help="En iyi epoch'u belirleyen metrik")
    p.add_argument("--mode", choices=["max", "min"], default="max")
    p.add_argument("--out", default="results/learning_curves.png")
    args = p.parse_args()

    runs = []
    for spec in args.run:
        name, folder = spec.split("=", 1)
        path = Path(folder) / "history.csv" if Path(folder).is_dir() else Path(folder)
        runs.append((name, pd.read_csv(path)))

    fig, axes = plt.subplots(len(runs), len(PANELS), figsize=(6 * len(PANELS), 4 * len(runs)), squeeze=False)
    for r, (name, hist) in enumerate(runs):
        best = None
        if args.monitor in hist:
            best = int(hist.loc[hist[args.monitor].idxmax() if args.mode == "max"
                                else hist[args.monitor].idxmin(), "epoch"])
        for c, (title, cols) in enumerate(PANELS):
            ax = axes[r][c]
            for col in cols:
                if col in hist:
                    ax.plot(hist["epoch"], hist[col], marker="o", ms=3, label=col)
            if best is not None:
                ax.axvline(best, color="grey", ls="--", lw=1, label=f"en iyi epoch ({best})")
            ax.set_title(f"{name} — {title}")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Yazıldı: {args.out}")


if __name__ == "__main__":
    main()
