"""
Gönüllü (insan eliyle yapılmış) sahtecilik test setini hazırlar: orijinal–sahte çiftlerinden maske üretir.

Atölye akışı (bkz. Find it again! veri seti):
  1. Gönüllülere orijinal belgeler verilir (kişisel veriler maskelenmiş, yazılı onay alınmış).
  2. Her gönüllü Photoshop / GIMP / telefon düzenleyicisi / online PDF editörüyle 1-2 alan değiştirir
     ve sonucu PNG/JPG/PDF olarak kaydeder (isteğe bağlı: WhatsApp'tan gönderip geri indirir).
  3. Bu betik her (orijinal, sahte) çiftinin farkından sahte bölge maskesini çıkarır.
  4. Maskeler ``<out>/review/`` altındaki önizlemelerle **elle kontrol edilmelidir**.

Girdi: çift listesi CSV'si (``--pairs``), sütunlar::

    original,forged,volunteer,tamper_type,doc_type,channel
    orijinaller/garanti_01.pdf,sahteler/ayse/garanti_01.png,ayse,tutar,garanti,png
    orijinaller/garanti_01.pdf,sahteler/mehmet/garanti_01_wa.jpg,mehmet,iban,garanti,whatsapp

(yollar CSV'nin klasörüne göre; ``volunteer``, ``tamper_type``, ``doc_type``, ``channel`` opsiyoneldir)

Maske çıkarımı:
  * PDF'ler ``--dpi`` ile rasterleştirilir (uygulamayla aynı: 200).
  * Boyut farklıysa (ekran görüntüsü / WhatsApp küçültmesi) orijinal, sahtenin boyutuna ölçeklenir;
    ``--align`` ile küçük kaymalar ECC (öteleme) ile hizalanır.
  * Gri tonda |sahte − orijinal| hafif bulanıklaştırılıp ``--diff-threshold`` ile eşiklenir; kapama +
    açma ile gürültü temizlenir, ``--min-area`` pikselden küçük bileşenler atılır, ``--dilate`` kadar genişletilir.
    WhatsApp gibi yeniden sıkıştırılmış çiftlerde eşik yükseltilmelidir (ör. 40).

Çıktı (``src/dataset.py`` → ``human_forged`` kaynağı, **daima test**)::

    data/9_human_forged/images/*   masks/*.png   metadata.csv   review/*.jpg

Her orijinal bir kez (label 0, maske boş), her sahte bir kez (label 1) yazılır; ``group_id`` orijinal
dosyanın adıdır, böylece bir belgenin tüm varyantları aynı grupta kalır.

Kullanım:
    python tools/make_human_masks.py --pairs data/9_human_raw/pairs.csv --out data/9_human_forged
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import read_image_rgb  # noqa: E402


def load(path: Path, dpi: int) -> np.ndarray:
    if path.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
            return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
    return read_image_rgb(path)


def align(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """``moving``'i ``reference``'a ECC öteleme modeliyle hizalar (küçük kaymalar için)."""
    ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mov = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(ref, mov, warp, cv2.MOTION_TRANSLATION,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
    except cv2.error:
        return moving
    return cv2.warpAffine(moving, warp, (reference.shape[1], reference.shape[0]),
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def diff_mask(original: np.ndarray, forged: np.ndarray, threshold: int, min_area: int, dilate: int) -> np.ndarray:
    a = cv2.GaussianBlur(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), (3, 3), 0).astype(np.int16)
    b = cv2.GaussianBlur(cv2.cvtColor(forged, cv2.COLOR_RGB2GRAY), (3, 3), 0).astype(np.int16)
    mask = (np.abs(a - b) > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    if dilate > 0:
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    return keep * 255


def save_image(src: Path, image: np.ndarray, dst_stem: Path) -> str:
    """Raster dosyalar ham baytlarıyla kopyalanır (izler korunur); PDF'ler PNG olarak yazılır."""
    if src.suffix.lower() == ".pdf":
        dst = dst_stem.with_suffix(".png")
        cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tofile(str(dst))
    else:
        dst = dst_stem.with_suffix(src.suffix.lower())
        shutil.copyfile(src, dst)
    return dst.name


def main() -> None:
    p = argparse.ArgumentParser(description="Gönüllü sahtecilik çiftlerinden maske ve test seti üret")
    p.add_argument("--pairs", required=True, help="original,forged[,volunteer,tamper_type,doc_type,channel] CSV")
    p.add_argument("--out", default="data/9_human_forged")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--diff-threshold", type=int, default=25)
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--align", action="store_true", help="Küçük kaymaları ECC ile hizala")
    p.add_argument("--lang", default="tr")
    args = p.parse_args()

    pairs_path = Path(args.pairs)
    pairs = pd.read_csv(pairs_path, keep_default_na=False)
    out = Path(args.out)
    for sub in ("images", "masks", "review"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows, written_originals = [], set()
    for i, pair in pairs.iterrows():
        orig_path, forged_path = pairs_path.parent / pair["original"], pairs_path.parent / pair["forged"]
        group = Path(pair["original"]).stem
        original, forged = load(orig_path, args.dpi), load(forged_path, args.dpi)
        meta = {"group_id": group, "lang": args.lang, "doc_type": pair.get("doc_type", ""),
                "channel": pair.get("channel", ""), "volunteer": pair.get("volunteer", "")}
        if group not in written_originals:
            name = save_image(orig_path, original, out / "images" / f"{group}__00_orig")
            rows.append({"image": f"images/{name}", "mask": "", "label": 0, "tamper_type": "orig", **meta})
            written_originals.add(group)
        reference = original
        if reference.shape[:2] != forged.shape[:2]:
            reference = cv2.resize(reference, (forged.shape[1], forged.shape[0]), interpolation=cv2.INTER_AREA)
        if args.align:
            forged_aligned = align(reference, forged)
        else:
            forged_aligned = forged
        mask = diff_mask(reference, forged_aligned, args.diff_threshold, args.min_area, args.dilate)
        stem = f"{group}__{i + 1:02d}_{pair.get('tamper_type', '') or 'human'}"
        name = save_image(forged_path, forged, out / "images" / stem)
        cv2.imencode(".png", mask)[1].tofile(str(out / "masks" / f"{stem}.png"))
        review = forged.copy()
        review[mask > 0] = (0.5 * review[mask > 0] + [127, 0, 0]).astype(np.uint8)
        cv2.imencode(".jpg", cv2.cvtColor(review, cv2.COLOR_RGB2BGR))[1].tofile(str(out / "review" / f"{stem}.jpg"))
        ratio = float((mask > 0).mean())
        if ratio == 0 or ratio > 0.2:
            print(f"UYARI {stem}: maske alanı %{ratio * 100:.2f} — eşiği/hizalamayı kontrol edin")
        rows.append({"image": f"images/{name}", "mask": f"masks/{stem}.png", "label": 1,
                     "tamper_type": pair.get("tamper_type", "") or "human", **meta})
    fields = ["image", "mask", "group_id", "label", "tamper_type", "lang", "doc_type", "channel", "volunteer"]
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(written_originals)} orijinal + {len(rows) - len(written_originals)} sahte → {out} "
          f"(maskeleri {out / 'review'} altında kontrol edin)")


if __name__ == "__main__":
    main()
