"""
Kaggle'da üretilmiş ``predictions_*.csv`` dosyalarından yerel ``splits`` dosyası kurar.

Eğitim Kaggle'da yapıldığında ``splits.csv`` çoğu zaman yerelde yoktur; yerelde ``assign_splits`` yeniden
çalıştırılırsa farklı bir manifest (ör. yalnızca bir kaynak) farklı bölme üretir ve modelin **eğitimde
gördüğü** belgeler "test"e karışır (sızıntı → iyimser sonuç). ``evaluate.py``'ın yazdığı tahmin
dosyaları ise test satırlarını ``group_id`` ve ``split`` ile birlikte taşır; bu betik o satırların Kaggle
yollarını (``/kaggle/working/.../8_real_forged/images/x.png``) yerel ``--data-root`` altındaki aynı veri
seti klasörüne çevirir.

Uygulamadaki kararları birebir yeniden üretmek için modelin kalibrasyonuyla çalıştırın:

    python tools/splits_from_predictions.py --predictions results/predictions_test_*.csv \\
        --out results/protocol/stageE_kaggle_test.csv
    python evaluate.py --checkpoint checkpoints/stageE/best.pth --splits results/protocol/stageE_kaggle_test.csv \\
        --no-calibration --temperature 4.1956 --threshold 0.77 --error-maps 5 --out results/stageE_local

(Val satırları tahmin dosyalarında olmadığı için kalibrasyon yeniden öğrenilemez; ``--no-calibration`` şart.)
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SOURCE_DIRS  # noqa: E402

KNOWN_DIRS = {name for names in SOURCE_DIRS.values() for name in names}


def localize(path: str, data_root: Path) -> str:
    """Yoldaki bilinen veri seti klasöründen sonrasını ``data_root`` altına taşır."""
    if not path:
        return path
    parts = Path(str(path).replace("\\", "/")).parts
    for i, part in enumerate(parts):
        if part in KNOWN_DIRS:
            return str(data_root.joinpath(*parts[i:]))
    raise ValueError(f"Bilinen veri seti klasörü bulunamadı: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="predictions_*.csv → yerel splits dosyası")
    p.add_argument("--predictions", nargs="+", required=True, help="Dosyalar ya da glob desenleri")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = sorted({f for pattern in args.predictions for f in glob.glob(pattern)})
    if not files:
        raise SystemExit("Tahmin dosyası bulunamadı.")
    root = Path(args.data_root)
    frames = []
    for f in files:
        pred = pd.read_csv(f, keep_default_na=False)
        keep = [c for c in ("image_path", "mask_path", "label", "group_id", "source", "split", "lang") if c in pred]
        frame = pred[keep].copy()
        frame["image_path"] = [localize(x, root) for x in frame["image_path"]]
        frame["mask_path"] = [localize(x, root) for x in frame["mask_path"]]
        frames.append(frame)
    splits = pd.concat(frames, ignore_index=True).drop_duplicates("image_path")
    missing = [x for x in splits["image_path"] if not Path(x).is_file()]
    if missing:
        print(f"UYARI: {len(missing)} görüntü yerelde yok (ör. {missing[0]}); bu satırlar çıkarıldı.")
        splits = splits[~splits["image_path"].isin(missing)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.out, index=False)
    print(splits.groupby(["split", "source", "label"]).size().to_string())
    print(f"Yazıldı: {args.out} ({len(splits)} görüntü)")


if __name__ == "__main__":
    main()
