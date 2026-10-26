"""
Dış modellerin (TruFor, DTD, CAT-Net) çıktılarını ``evaluate.py --external-maps`` biçimine çevirir.

Girdi: dış modelin her görüntü için yazdığı dosya (adı ``<image_key>`` ile başlamalı; ör.
``tools/export_eval_images.py`` çıktısındaki dosya adlarıyla çalıştırılmışsa kendiliğinden öyledir).

Desteklenen biçimler:

* ``.npz`` (TruFor'un ``test.py`` çıktısı): yerelleştirme haritası ``--map-key`` (varsayılan ``map``),
  belge skoru ``--score-key`` (varsayılan ``score``) anahtarlarından okunur. Anahtar adları sürüme göre
  değişebilir; ``--list-keys`` ile ilk dosyanın anahtarlarını görün.
* ``.npy``: tek bir olasılık haritası.
* ``.png`` / ``.jpg`` / ``.tif``: 8 ya da 16 bit gri olasılık haritası (DTD / CAT-Net görselleştirmeleri).

Harita [0, 1]'e ölçeklenir. ``--invert`` haritayı 1 − p yapar (gerçek = 1 kodlayan modeller için).
Çıktı: ``<out>/<image_key>.npy`` + skor varsa ``<out>/scores.csv`` (key, score).

Kullanım:
    python tools/convert_baseline_outputs.py --inputs ../TruFor/output --out baselines/trufor/maps
    python evaluate.py --external-maps baselines/trufor/maps --external-scores baselines/trufor/maps/scores.csv \\
        --splits results/protocol/test_v1.csv --out results/baselines/trufor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SUFFIXES = (".npz", ".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def to_probability(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(np.asarray(array, dtype=np.float32))
    if array.ndim == 3:  # (C, H, W) ya da (H, W, C): son kanal "sahte" olasılığı kabul edilir
        array = array[-1] if array.shape[0] <= 4 else array[..., -1]
    if array.max() > 1.0 + 1e-6:
        array = array / (65535.0 if array.max() > 255 else 255.0)
    return np.clip(array, 0.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Dış model çıktılarını <image_key>.npy biçimine çevir")
    p.add_argument("--inputs", required=True, help="Dış modelin çıktı klasörü")
    p.add_argument("--out", required=True)
    p.add_argument("--map-key", default="map")
    p.add_argument("--score-key", default="score")
    p.add_argument("--invert", action="store_true", help="Harita 1 = gerçek kodluyorsa çevir")
    p.add_argument("--strip-suffix", default="", help="Dosya adından atılacak ek (ör. '_pred')")
    p.add_argument("--list-keys", action="store_true", help="İlk .npz dosyasının anahtarlarını yazdır ve çık")
    args = p.parse_args()

    files = sorted(f for f in Path(args.inputs).rglob("*") if f.suffix.lower() in SUFFIXES)
    if not files:
        raise SystemExit(f"{args.inputs} altında çıktı dosyası yok")
    if args.list_keys:
        first = next((f for f in files if f.suffix == ".npz"), None)
        print(dict(np.load(first)).keys() if first else "npz dosyası yok")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    for f in files:
        key = f.stem.removesuffix(args.strip_suffix) if args.strip_suffix else f.stem
        if f.suffix == ".npz":
            data = np.load(f)
            if args.map_key not in data:
                raise KeyError(f"{f.name}: '{args.map_key}' anahtarı yok (var olanlar: {list(data.keys())})")
            prob = to_probability(data[args.map_key])
            if args.score_key in data:
                scores.append({"key": key, "score": float(np.squeeze(data[args.score_key]))})
        elif f.suffix == ".npy":
            prob = to_probability(np.load(f))
        else:
            prob = to_probability(cv2.imread(str(f), cv2.IMREAD_UNCHANGED))
        if args.invert:
            prob = 1.0 - prob
        np.save(out / f"{key}.npy", prob.astype(np.float16))
    if scores:
        pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
    print(f"{len(files)} harita → {out}" + (f" (+ {len(scores)} belge skoru: scores.csv)" if scores else ""))


if __name__ == "__main__":
    main()
