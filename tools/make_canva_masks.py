"""
Canva'da düzenlenmiş PDF'lerden elle sahtecilik test setini hazırlar.

Canva, PDF'i içe aktarırken dekontu yeni bir sayfaya yerleştirip yazıları yeniden dizer; bu yüzden
orijinalle piksel farkı (``make_human_masks.py``) sayfanın her yerinde fark bulur. Bu araç maskeyi metinden
çıkarır: sahte PDF'te bulunup orijinal PDF'te bulunmayan kelimelerin kutuları sahte bölgedir.

Girdi (``--raw``, varsayılan ``data/9_human_raw``)::

    orijinaller/is_02.pdf            bankanın orijinal PDF'i (label 0, kanal banka_orijinal)
    orijinaller/is_02_canva.pdf      opsiyonel: Canva'dan değiştirilmeden indirilmiş kopya (label 0, kanal canva_temiz)
    sahteler/is_02_miktar_tutar.pdf  is_<no>_<yöntem>_<tür>.pdf; yöntem: beyaz | miktar, tür: tutar | iban | tarih | isim

Çıktı ``make_human_masks.py`` ile aynı biçimdedir (``src/dataset.py`` → ``human_forged``, daima test)::

    data/9_human_forged/images/*  masks/*.png  metadata.csv  review/*.jpg

Canva çıktılarındaki beyaz sayfa boşlukları kırpılır; böylece dekont orijinallerle benzer ölçekte değerlendirilir.
Maskeler ``review/`` altındaki önizlemelerle elle kontrol edilmelidir.

Kullanım:
    python tools/make_canva_masks.py --raw data/9_human_raw --out data/9_human_forged
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pymupdf

NAME_RE = re.compile(r"is_(\d+)_(beyaz|miktar|mitar)_(tutar|iban|tarih|isim)\.pdf$")
CHANNELS = {"beyaz": "beyaz_katman", "miktar": "metin_duzenleme", "mitar": "metin_duzenleme"}
FIELDS = ["image", "mask", "group_id", "label", "tamper_type", "lang", "doc_type", "channel", "volunteer"]


def norm(text: str) -> str:
    return re.sub(r"[\s:]+", "", text)


def render(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()


def changed_boxes(original_text: str, words: list) -> list:
    """Orijinal metinde bulunmayan kelimelerin kutuları; Canva'nın boşluk ve iki nokta farkları yok sayılır."""
    squashed = norm(original_text)
    boxes = [w[:4] for w in words if norm(w[4]) and norm(w[4]) not in squashed]
    if boxes:
        return boxes
    # Yeni değer belgede başka yerde de geçiyorsa: kelime sayılarını karşılaştır.
    remaining = Counter(norm(t) for t in original_text.split())
    for word in words:
        token = norm(word[4])
        if token and remaining[token] > 0:
            remaining[token] -= 1
        elif token:
            boxes.append(word[:4])
    return boxes


def crop_rows(image: np.ndarray, pad: int) -> tuple[int, int]:
    rows = np.where((image.min(axis=2) < 245).any(axis=1))[0]
    return max(0, int(rows[0]) - pad), min(image.shape[0], int(rows[-1]) + pad + 1)


def write_png(path: Path, image: np.ndarray) -> None:
    data = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    cv2.imencode(".png", data)[1].tofile(str(path))


def main() -> None:
    p = argparse.ArgumentParser(description="Canva PDF'lerinden metin farkıyla maske ve test seti üret")
    p.add_argument("--raw", default="data/9_human_raw")
    p.add_argument("--out", default="data/9_human_forged")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--volunteer", default="irem")
    p.add_argument("--doc-type", default="is_bankasi")
    args = p.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    for sub in ("images", "masks", "review"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    pad = round(10 * args.dpi / 72)
    rows, counts = [], Counter()
    forged_files = sorted((raw / "sahteler").glob("*.pdf"), key=lambda f: f.name)
    for path in forged_files:
        match = NAME_RE.match(path.name)
        if not match:
            print(f"ATLANDI {path.name}: ad beklenen biçimde değil")
            continue
        group = f"is_{int(match.group(1)):02d}"
        tamper, channel = match.group(3), CHANNELS[match.group(2)]
        meta = {"group_id": group, "lang": "tr", "doc_type": args.doc_type, "volunteer": args.volunteer}

        with pymupdf.open(raw / "orijinaller" / f"{group}.pdf") as doc:
            original_text = doc[0].get_text()
            write_png(out / "images" / f"{group}__00_orig.png", render(doc[0], args.dpi))
        rows.append({"image": f"images/{group}__00_orig.png", "mask": "", "label": 0, "tamper_type": "orig",
                     "channel": "banka_orijinal", **meta})

        with pymupdf.open(path) as doc:
            image = render(doc[0], args.dpi)
            boxes = changed_boxes(original_text, doc[0].get_text("words"))
        mask = np.zeros(image.shape[:2], np.uint8)
        scale = args.dpi / 72
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(mask, (round(x0 * scale) - 3, round(y0 * scale) - 3),
                          (round(x1 * scale) + 3, round(y1 * scale) + 3), 255, -1)
        top, bottom = crop_rows(image, pad)
        image, mask = image[top:bottom], mask[top:bottom]
        stem = f"{group}__01_{tamper}"
        write_png(out / "images" / f"{stem}.png", image)
        write_png(out / "masks" / f"{stem}.png", mask)
        review = image.copy()
        review[mask > 0] = (0.5 * review[mask > 0] + [127, 0, 0]).astype(np.uint8)
        cv2.imencode(".jpg", cv2.cvtColor(review, cv2.COLOR_RGB2BGR))[1].tofile(str(out / "review" / f"{stem}.jpg"))
        ratio = float((mask > 0).mean())
        if ratio == 0 or ratio > 0.05:
            print(f"UYARI {stem}: maske alanı %{ratio * 100:.2f} — elle kontrol edin")
        rows.append({"image": f"images/{stem}.png", "mask": f"masks/{stem}.png", "label": 1, "tamper_type": tamper,
                     "channel": channel, **meta})
        counts[channel] += 1

        clean = raw / "orijinaller" / f"{group}_canva.pdf"
        if clean.exists():
            with pymupdf.open(clean) as doc:
                image = render(doc[0], args.dpi)
            top, bottom = crop_rows(image, pad)
            write_png(out / "images" / f"{group}__02_orig.png", image[top:bottom])
            rows.append({"image": f"images/{group}__02_orig.png", "mask": "", "label": 0, "tamper_type": "orig",
                         "channel": "canva_temiz", **meta})
            counts["canva_temiz"] += 1

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} görüntü → {out}: {dict(counts)} (maskeleri {out / 'review'} altında kontrol edin)")


if __name__ == "__main__":
    main()
