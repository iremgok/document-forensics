"""
``forge_real`` metadata'sındaki edinim etiketlerini (``acquisition``) kaynağa göre denetler.

Her kaynak belge (``source_file`` + ``page``) için edinim türü ``src.forge_real.decide_acquisition`` ile
**kaynaktan** yeniden belirlenir (``_acquisition.txt`` > dosya türü > piksel gürültüsü) ve metadata'daki
etiketle karşılaştırılır. Tutarsız satırlar listelenir; ``--out`` ile CSV olarak yazılır.

Eski ``forge_real`` sürümü kararı yalnızca piksel gürültüsünden verdiği için 200 DPI'da PNG'ye çevrilmiş
A4 PDF'lerin bir kısmı ``photo`` olarak etiketlenmişti; bu araç o satırları bulur. Hiçbir dosyayı değiştirmez.

Kullanım:
    python tools/audit_acquisition.py                                   # data/8_real_forged
    python tools/audit_acquisition.py --metadata data/8_real_forged_v2/metadata.csv --out results/audit_v2.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import read_image_rgb  # noqa: E402
from src.forge_real import decide_acquisition, is_digital, load_pages  # noqa: E402


def audit(meta: pd.DataFrame, src: Path, pixel_check: bool = True) -> pd.DataFrame:
    """Kaynak belge başına bir satır: metadata etiketi, kaynaktan karar, gerekçe, tutarlılık."""
    rows = []
    for (source_file, page), group in meta.groupby(["source_file", "page"], sort=True):
        path = src / source_file
        labels = sorted(group["acquisition"].astype(str).unique())
        row = {"source_file": source_file, "page": int(page), "n_images": len(group),
               "metadata_acquisition": "/".join(labels),
               "metadata_source": "/".join(sorted(group.get("acquisition_source", pd.Series([""])).fillna("")
                                                  .astype(str).unique())) if "acquisition_source" in group else ""}
        if not path.is_file():
            rows.append({**row, "kaynak_karari": "", "gerekce": "kaynak dosya yok", "tutarli": None})
            continue
        image = None
        size = ""
        if pixel_check or path.suffix.lower() not in (".pdf", ".png"):
            pages = load_pages(path) if path.suffix.lower() == ".pdf" else [read_image_rgb(path)]
            image = pages[min(int(page), len(pages) - 1)]
            size = f"{image.shape[1]}x{image.shape[0]}"
        decision, reason = decide_acquisition(path, src, image=image)
        rows.append({**row, "boyut": size, "kaynak_karari": decision, "gerekce": reason,
                     "piksel_gurultusu_karari": ("digital" if is_digital(image) else "photo") if image is not None else "",
                     "tutarli": labels == [decision]})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="forge_real edinim etiketi denetimi (salt okunur)")
    p.add_argument("--metadata", default="data/8_real_forged/metadata.csv")
    p.add_argument("--src", default="data/7_real_originals", help="forge_real --src klasörü (kaynak belgeler)")
    p.add_argument("--no-pixel-check", action="store_true",
                   help="PDF/PNG için piksel gürültüsü kararını hesaplama (daha hızlı)")
    p.add_argument("--out", default=None, help="Tutarsız satırların yazılacağı CSV (verilmezse yalnızca ekrana)")
    args = p.parse_args()

    meta = pd.read_csv(args.metadata, keep_default_na=False)
    table = audit(meta, Path(args.src), pixel_check=not args.no_pixel_check)
    bad = table[table["tutarli"] == False]  # noqa: E712 (None = kaynak yok)
    print(f"{len(table)} kaynak belge/sayfa denetlendi; {len(bad)} tutarsız, "
          f"{int(table['tutarli'].isna().sum())} kaynak dosyası bulunamadı.")
    if not bad.empty:
        bad = bad.assign(kurum=bad["source_file"].str.split("/").str[0])
        print(bad.groupby(["kurum", "metadata_acquisition", "kaynak_karari", "gerekce"]).size().to_string())
        with pd.option_context("display.width", 200, "display.max_colwidth", 60, "display.max_rows", 200):
            print(bad.drop(columns="kurum").to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        bad.to_csv(args.out, index=False)
        print(f"Yazıldı: {args.out}")
    sys.exit(1 if len(bad) else 0)


if __name__ == "__main__":
    main()
