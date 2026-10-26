"""
Değerlendirme görüntülerini dış modeller (TruFor, DTD, CAT-Net) için düz dosya olarak dışa aktarır.

Dış modellerin kendi çıkarım betikleri bir klasördeki görüntü dosyalarını okur; DocTamper ise LMDB
içindedir. Bu betik ``splits.csv``'deki seçili bölümlerin görüntülerini **ham baytlarıyla** (yeniden
kodlamadan; sıkıştırma izleri korunur) ``<out>/images/<image_key>.<uzantı>`` olarak yazar ve
``<out>/list.csv`` (key, image_path, mask_path, label, source, split) dosyasını üretir.

Dış modelin çıktıları ``tools/convert_baseline_outputs.py`` ile ``<image_key>.npy`` haritalarına
çevrilip ``evaluate.py --external-maps`` ile **aynı metrik koduyla** ölçülür.

Kullanım:
    python tools/export_eval_images.py --splits results/protocol/test_v1.csv --out baselines/eval_images
    python tools/export_eval_images.py --splits checkpoints/stageE/splits.csv --sets test --max-per-source 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import LMDB_PREFIX, image_key, read_image_bytes  # noqa: E402


def sniff_extension(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    return ".bin"


def main() -> None:
    p = argparse.ArgumentParser(description="Değerlendirme görüntülerini dış modeller için dışa aktar")
    p.add_argument("--splits", required=True)
    p.add_argument("--sets", nargs="+", default=["test", "ood"])
    p.add_argument("--max-per-source", type=int, default=None, help="Kaynak ve sınıf başına üst sınır")
    p.add_argument("--out", default="baselines/eval_images")
    args = p.parse_args()

    splits = pd.read_csv(args.splits, keep_default_na=False)
    frame = splits[splits["split"].isin(args.sets)]
    if args.max_per_source:
        frame = pd.concat([g.sample(min(len(g), args.max_per_source), random_state=0)
                           for _, g in frame.groupby(["source", "label"])])
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for row in frame.itertuples():
        key = image_key(row.image_path)
        if key in seen:
            raise ValueError(f"Anahtar çakışması: {key} ({row.image_path})")
        seen.add(key)
        data = read_image_bytes(row.image_path)
        ext = sniff_extension(data) if str(row.image_path).startswith(LMDB_PREFIX) else Path(row.image_path).suffix
        (out / "images" / f"{key}{ext.lower()}").write_bytes(data)
        rows.append({"key": key, "file": f"images/{key}{ext.lower()}", "image_path": row.image_path,
                     "mask_path": row.mask_path, "label": row.label, "source": row.source, "split": row.split})
    pd.DataFrame(rows).to_csv(out / "list.csv", index=False)
    print(f"{len(rows)} görüntü yazıldı → {out / 'images'} (liste: {out / 'list.csv'})")


if __name__ == "__main__":
    main()
