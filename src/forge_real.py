"""
Gerçek orijinal belgelerden OCR rehberli sahte varyant üretici.

Neden?
------
Gerçek dekont/faturalar yalnızca "orijinal" olarak eğitime girerse model sahteciliği değil
**şablon farkını** öğrenir ("Garanti görünümü = orijinal", "sentetik şablon = sahte"). Bu betik
her gerçek belgeden aynı şablon, aynı kanal ve aynı kayıt zinciriyle sahte varyantlar üretir;
orijinal ile sahte arasındaki tek fark değiştirilen alan olur.

Akış (her dosya / PDF sayfası için)
-----------------------------------
1. Görüntü okunur (PDF sayfaları uygulamadaki gibi ``PDF_DPI`` ile rasterleştirilir) ve uzun kenar
   eğitimdeki ``max_side`` sınırına indirilir (sonradan ölçekleme izleri bozmasın diye).
2. EasyOCR kelime kutuları okunur ve sınıflandırılır: ``iban``, ``amount``, ``date``, ``number``,
   ``name``, ``text``.
3. Her varyantta 1-2 işlem uygulanır (maske: 255 = değiştirilen piksel):

   * ``digit_change``   : tutar/tarih/IBAN/numaradaki 1-2 rakam karakter düzeyinde değiştirilir.
   * ``field_rewrite``  : alan silinip yeni değer yazılır (ör. tutar değiştirilir).
   * ``name_swap``      : alıcı/gönderen adı sahte bir adla değiştirilir.
   * ``copy_move``      : belgedeki başka bir rakam glifi kopyalanıp hedef rakamın üzerine konur.
   * ``delete``         : bir alan/satır tamamen silinir (zemin ile doldurma veya inpainting).
   * ``inpaint_rewrite``: üretken YZ benzetimi — bölge ``cv2.inpaint`` ile doldurulur, yeni metin
                          yazılır ve yumuşatılır.

4. Uyumlandırma: taranmış/fotoğraflı belgede yeni yazı çoğunlukla belgedeki bulanıklık ve
   gürültüye getirilir (aksi hâlde "keskin yazı = sahte" kolay ipucu olur); dijital belgede
   ``add_edit_trace`` ile gerçekçi yerel düzenleme izi eklenir (bkz. ``generate_synthetic.EDIT_TRACES``).
5. **Orijinal dahil** gruptaki tüm görüntüler aynı son kanaldan/kaliteyle yeniden kaydedilir;
   böylece "yeniden kaydedilmiş = sahte" kestirmesi oluşmaz.

Girdi klasörü düzeni (alt klasör adı = kurum, ``doc_type`` sütununa yazılır)::

    data/7_real_originals/
        garanti/  dekont_001.pdf  dekont_002.png ...
        ziraat/   ...
        trendyol/ fatura_001.pdf ...

Çıktı (``src/dataset.py`` → ``real_forged`` kaynağı)::

    data/8_real_forged/images/*.png|jpg   masks/*.png   metadata.csv

``group_id`` = ``real:<kurum>/<dosya>`` — ``real_originals`` kaynağıyla aynı biçimdedir; aynı belgenin
orijinali ve sahteleri asla farklı bölümlere (train/val/test) düşmez. ``real_forged`` her belgenin
yeniden kaydedilmiş orijinalini de içerdiği için eğitimde ``real_originals`` ile **birlikte**
kullanılmamalıdır (ham orijinal + yeniden kaydedilmiş sahte → "yeniden kayıt = sahte" kestirmesi).

Kullanım:
    python -m src.forge_real                                  # data/7_real_originals → data/8_real_forged
    python -m src.forge_real --variants 4 --gpu               # sayfa başına 4 sahte, OCR GPU'da
    python -m src.forge_real --limit 5 --out data/_forge_preview --overwrite   # hızlı önizleme
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import ImageFont

from src.dataset import IMAGE_EXTENSIONS, limit_long_side, read_image_rgb
from src.generate_synthetic import (Box, _background_color, _draw_text, _expand, _mark, _person, _save,
                                    add_edit_trace, apply_digital_channel, available_fonts, load_font,
                                    sample_digital_channel)
from src.ocr_checker import _AMOUNT_RE, get_reader

PDF_DPI = 200  # app.py ile aynı rasterleştirme çözünürlüğü
METADATA_FIELDS = ["image", "mask", "group_id", "label", "tamper_type", "lang", "doc_type", "acquisition",
                   "channel", "edit_trace", "source_file", "page", "edits"]

#: İşlemler ve seçilme ağırlıkları.
OP_WEIGHTS: dict[str, float] = {
    "digit_change": 0.25,
    "field_rewrite": 0.20,
    "name_swap": 0.15,
    "copy_move": 0.12,
    "delete": 0.15,
    "inpaint_rewrite": 0.13,
}

_IBAN_RE = re.compile(r"\bTR\s?\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
_HONORIFIC_RE = re.compile(r"^(?:SAYIN|Sayın|Sayin|SN\.?|Sn\.?)\s+")
#: Ad gibi görünen ama alan etiketi olan sözcükler (bunlar ad sayılmaz).
_LABEL_WORDS = {"toplam", "tutar", "tarih", "iban", "açıklama", "aciklama", "alıcı", "alici", "gönderen",
                "gonderen", "fatura", "fiş", "fis", "dekont", "kdv", "matrah", "banka", "şube", "sube", "hesap",
                "numara", "no", "adres", "vergi", "dairesi", "ödeme", "odeme", "işlem", "islem", "referans",
                "müşteri", "musteri", "ünvan", "unvan", "toplamı", "tl", "try", "eft", "fast", "havale"}


@dataclass
class Word:
    """OCR ile okunmuş tek bir metin parçası."""

    text: str
    box: Box
    conf: float
    kind: str

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]


# ---------------------------------------------------------------------------
# Okuma ve OCR
# ---------------------------------------------------------------------------

def load_pages(path: Path, dpi: int = PDF_DPI) -> list[np.ndarray]:
    """Görüntüyü veya PDF sayfalarını RGB uint8 listesi olarak döndürür."""
    if path.suffix.lower() == ".pdf":
        import pymupdf

        pages = []
        with pymupdf.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                pages.append(np.ascontiguousarray(image[:, :, :3]))
        return pages
    return [read_image_rgb(path)]


def is_digital(image: np.ndarray) -> bool:
    """Açık renkli zeminde gürültü yoksa belge dijital doğmuştur (PDF/ekran görüntüsü).

    Taranmış/fotoğraflı belgelerde kâğıt zemininde sensör gürültüsü bulunur; dijitalde zemin
    pikselleri komşularıyla neredeyse aynıdır.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    residual = gray - cv2.blur(gray, (3, 3))
    # Yalnızca 5x5 komşuluğu tamamen açık renkli zemin pikselleri: yazı kenarlarındaki
    # yumuşatma (anti-aliasing) gürültü sanılmasın.
    background = cv2.erode((gray > 200).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if background.mean() < 0.2:
        return False
    return float(np.std(residual[background])) < 0.8


def classify(text: str) -> Optional[str]:
    """OCR metninin alan türü (değiştirilmeye uygun değilse None)."""
    stripped = text.strip()
    if len(stripped) < 2:
        return None
    if _IBAN_RE.search(stripped):
        return "iban"
    if _DATE_RE.search(stripped):
        return "date"
    if _AMOUNT_RE.search(stripped):
        return "amount"
    digits = sum(ch.isdigit() for ch in stripped)
    if digits >= 3 and digits >= 0.5 * len(stripped.replace(" ", "")):
        return "number"
    tokens = stripped.split()
    if (2 <= len(tokens) <= 4 and all(t.isalpha() and len(t) >= 2 for t in tokens)
            and not any(t.lower() in _LABEL_WORDS for t in tokens)
            and all(t[0].isupper() for t in tokens)):
        return "name"
    if sum(ch.isalpha() for ch in stripped) >= 3:
        return "text"
    return None


def split_label(text: str, box: Box, font: ImageFont.FreeTypeFont) -> tuple[str, Box]:
    """"Etiket: değer" biçimindeki OCR parçasından yalnızca değeri ve yaklaşık kutusunu döndürür.

    Örn. ``Tutar: 12.450,00 TL`` → ``12.450,00 TL``, ``SAYIN İrem Gök`` → ``İrem Gök``. Etiket
    değiştirilmez; sahtecilik değerdedir.
    """
    honorific = _HONORIFIC_RE.match(text)
    label, sep, value = text.partition(":")
    if honorific:
        start = honorific.end()
    elif sep and value.strip() and len(label) <= 25:
        start = len(label) + 1 + (len(value) - len(value.lstrip()))
    else:
        return text, box
    if start >= len(text):
        return text, box
    xa, _ = char_span(text, start, box, font)
    return text[start:].strip(), (xa, box[1], box[2], box[3])


def detect_words(image: np.ndarray, reader, measure_font: ImageFont.FreeTypeFont, min_conf: float = 0.4,
                 min_height: int = 8) -> list[Word]:
    """EasyOCR kelime/ifade kutuları → sınıflandırılmış ``Word`` listesi (etiketler ayrılmış)."""
    words = []
    h, w = image.shape[:2]
    for quad, text, conf in reader.readtext(image, detail=1, paragraph=False):
        pts = np.asarray(quad, dtype=float)
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
        box = (max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1)))
        text, box = split_label(text.strip(), box, measure_font)
        kind = classify(text)
        if kind is None or conf < min_conf or box[3] - box[1] < min_height or box[2] - box[0] < 4:
            continue
        words.append(Word(text.strip(), box, float(conf), kind))
    return words


# ---------------------------------------------------------------------------
# Stil tahmini ve çizim yardımcıları
# ---------------------------------------------------------------------------

def ink_color(image: np.ndarray, box: Box) -> tuple[int, int, int]:
    """Kutudaki yazı rengi: en koyu piksellerin medyanı."""
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if crop.size == 0:
        return (30, 30, 30)
    lum = crop.mean(axis=1)
    dark = crop[lum <= np.percentile(lum, 15) + 10]
    return tuple(int(v) for v in np.median(dark, axis=0)) if len(dark) else (30, 30, 30)


def fit_font(text: str, box: Box, font_path: Optional[str]) -> tuple[ImageFont.FreeTypeFont, tuple[int, int]]:
    """Metni kutuya (genişlik ve yükseklik) en iyi oturtan font ve çizim başlangıcı."""
    height = max(6, box[3] - box[1])
    font = load_font(font_path, max(6, int(height * 0.85)))
    length = max(1.0, font.getlength(text))
    size = int(np.clip(font.size * (box[2] - box[0]) / length, height * 0.55, height * 1.1))
    font = load_font(font_path, max(6, size))
    left, top, _, bottom = font.getbbox(text)
    origin = (box[0] - left, int(round((box[1] + box[3]) / 2 - (top + bottom) / 2)))
    return font, origin


def char_span(text: str, index: int, box: Box, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """``text[index]`` karakterinin kutudaki yaklaşık yatay aralığı (orantılı genişlikle)."""
    total = max(1.0, font.getlength(text))
    scale = (box[2] - box[0]) / total
    xa = box[0] + font.getlength(text[:index]) * scale
    xb = box[0] + font.getlength(text[:index + 1]) * scale
    return int(np.floor(xa)), int(np.ceil(xb))


def erase(image: np.ndarray, box: Box, force_inpaint: bool = False) -> np.ndarray:
    """Kutuyu siler: düz zeminde medyan renkle boyar, dokulu zeminde (fotoğraf) inpainting yapar."""
    x0, y0, x1, y1 = _expand(box, 3, image.shape)
    ring = np.concatenate([image[y0:y0 + 2, x0:x1].reshape(-1, 3), image[y1 - 2:y1, x0:x1].reshape(-1, 3)])
    if force_inpaint or float(ring.astype(np.float32).std(axis=0).mean()) > 4.0:
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[box[1]:box[3], box[0]:box[2]] = 255
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    out = image.copy()
    out[box[1]:box[3], box[0]:box[2]] = _background_color(image, box)
    return out


def mutate_digits(text: str, rng: random.Random, count: int) -> tuple[str, list[int]]:
    """Metindeki ``count`` rakamı farklı rakamlarla değiştirir (biçim ve genişlik korunur)."""
    positions = [i for i, ch in enumerate(text) if ch.isdigit()]
    chosen = sorted(rng.sample(positions, min(count, len(positions)))) if positions else []
    chars = list(text)
    for i in chosen:
        options = [d for d in "0123456789" if d != chars[i] and not (i == positions[0] and d == "0")]
        chars[i] = rng.choice(options)
    return "".join(chars), chosen


def harmonize(image: np.ndarray, before: np.ndarray, mask: np.ndarray, digital: bool,
              rng: random.Random) -> tuple[np.ndarray, str]:
    """Düzenlenen bölgeyi belgenin edinim karakterine yaklaştırır.

    * Dijital: ``add_edit_trace`` (yerel JPEG / yeniden örnekleme / gürültü / renk izi).
    * Taranmış/fotoğraf: %70 olasılıkla yeni yazı belgedeki kadar bulanıklaştırılır ve zemin
      gürültüsü kadar gürültü eklenir; kalan %30'da keskin bırakılır (acemi sahtecilik).
    """
    if digital:
        return add_edit_trace(image, mask, rng)
    if rng.random() > 0.7:
        return image, "none"
    region = mask > 0
    gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    noise_std = float(np.std((gray - cv2.blur(gray, (3, 3)))[~region])) if (~region).any() else 1.5
    blurred = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.4, 1.0)).astype(np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    blurred += np_rng.normal(0, max(0.5, min(noise_std, 4.0)), image.shape)
    out = image.copy()
    out[region] = np.clip(blurred, 0, 255).astype(np.uint8)[region]
    return out, "blur_noise"


# ---------------------------------------------------------------------------
# Sahtecilik işlemleri: (image, mask, words, ctx) → (yeni görüntü | None, düzenleme kaydı)
# ---------------------------------------------------------------------------

@dataclass
class Context:
    rng: random.Random
    font_path: Optional[str]


def _pick(words: list[Word], kinds: tuple[str, ...], rng: random.Random) -> Optional[Word]:
    candidates = [w for w in words if w.kind in kinds]
    return rng.choice(candidates) if candidates else None


def _rewrite(image: np.ndarray, mask: np.ndarray, word: Word, new_text: str, ctx: Context,
             generative: bool = False) -> np.ndarray:
    """Alanı silip ``new_text``'i aynı konuma, kutuya oturan font ve yazı rengiyle yazar."""
    font, origin = fit_font(word.text, word.box, ctx.font_path)
    color = ink_color(image, word.box)
    image = erase(image, word.box, force_inpaint=generative)
    image = _draw_text(image, origin, new_text, font, color)
    left, top, right, bottom = font.getbbox(new_text)
    drawn = (origin[0] + left, origin[1] + top, origin[0] + right, origin[1] + bottom)
    full = _expand((min(word.box[0], drawn[0]), min(word.box[1], drawn[1]),
                    max(word.box[2], drawn[2]), max(word.box[3], drawn[3])), 2, image.shape)
    if generative:
        x0, y0, x1, y1 = full
        image[y0:y1, x0:x1] = cv2.GaussianBlur(image[y0:y1, x0:x1], (3, 3), 0.8)
    _mark(mask, full)
    return image


def op_digit_change(image, mask, words, ctx):
    word = _pick(words, ("amount", "date", "iban", "number"), ctx.rng)
    if word is None:
        return None, {}
    new_text, positions = mutate_digits(word.text, ctx.rng, ctx.rng.choice([1, 1, 2]))
    if not positions:
        return None, {}
    font, origin = fit_font(word.text, word.box, ctx.font_path)
    color = ink_color(image, word.box)
    for i in positions:
        xa, xb = char_span(word.text, i, word.box, font)
        cell = _expand((xa, word.box[1], xb, word.box[3]), 1, image.shape)
        image = erase(image, cell)
        image = _draw_text(image, (xa, origin[1]), new_text[i], font, color)
        _mark(mask, _expand(cell, 1, image.shape))
    return image, {"op": "digit_change", "old": word.text, "new": new_text, "box": word.box}


def op_field_rewrite(image, mask, words, ctx, generative: bool = False):
    word = _pick(words, ("amount", "date", "number", "name"), ctx.rng)
    if word is None:
        return None, {}
    if word.kind == "name":
        new_text = _person(ctx.rng, "tr")
        new_text = new_text.upper() if word.text.isupper() else new_text
    else:
        digits = sum(ch.isdigit() for ch in word.text)
        new_text, _ = mutate_digits(word.text, ctx.rng, max(1, min(digits, ctx.rng.randint(2, 4))))
    if new_text == word.text:
        return None, {}
    image = _rewrite(image, mask, word, new_text, ctx, generative)
    return image, {"op": "inpaint_rewrite" if generative else "field_rewrite", "old": word.text, "new": new_text,
                   "box": word.box}


def op_inpaint_rewrite(image, mask, words, ctx):
    return op_field_rewrite(image, mask, words, ctx, generative=True)


def op_name_swap(image, mask, words, ctx):
    word = _pick(words, ("name",), ctx.rng)
    if word is None:
        return None, {}
    new_text = _person(ctx.rng, "tr")
    new_text = new_text.upper() if word.text.isupper() else new_text
    image = _rewrite(image, mask, word, new_text, ctx)
    return image, {"op": "name_swap", "old": word.text, "new": new_text, "box": word.box}


def op_copy_move(image, mask, words, ctx):
    numeric = [w for w in words if w.kind in ("amount", "number", "date") and any(c.isdigit() for c in w.text)]
    if len(numeric) < 2:
        return None, {}
    dst = ctx.rng.choice(numeric)
    sources = [w for w in numeric if w is not dst and abs(w.height - dst.height) <= 0.15 * dst.height]
    if not sources:
        return None, {}
    src = ctx.rng.choice(sources)
    s_font, _ = fit_font(src.text, src.box, ctx.font_path)
    d_font, _ = fit_font(dst.text, dst.box, ctx.font_path)
    si = ctx.rng.choice([i for i, c in enumerate(src.text) if c.isdigit()])
    targets = [i for i, c in enumerate(dst.text) if c.isdigit() and c != src.text[si]]
    if not targets:
        return None, {}
    di = ctx.rng.choice(targets)
    sx0, sx1 = char_span(src.text, si, src.box, s_font)
    dx0, dx1 = char_span(dst.text, di, dst.box, d_font)
    patch = image[src.box[1]:src.box[3], sx0:sx1]
    if patch.size == 0 or dx1 <= dx0:
        return None, {}
    patch = cv2.resize(patch, (dx1 - dx0, dst.box[3] - dst.box[1]), interpolation=cv2.INTER_LINEAR)
    image = image.copy()
    image[dst.box[1]:dst.box[3], dx0:dx1] = patch
    _mark(mask, (dx0, dst.box[1], dx1, dst.box[3]))
    new_text = dst.text[:di] + src.text[si] + dst.text[di + 1:]
    return image, {"op": "copy_move", "old": dst.text, "new": new_text, "box": dst.box}


def op_delete(image, mask, words, ctx):
    narrow = [w for w in words if w.width < 0.5 * image.shape[1]]
    word = _pick(narrow, ("amount", "name", "text", "number", "date"), ctx.rng)
    if word is None:
        return None, {}
    box = _expand(word.box, 2, image.shape)
    image = erase(image, box, force_inpaint=ctx.rng.random() < 0.3)
    _mark(mask, box)
    return image, {"op": "delete", "old": word.text, "new": "", "box": word.box}


OPERATIONS: dict[str, Callable] = {
    "digit_change": op_digit_change,
    "field_rewrite": op_field_rewrite,
    "name_swap": op_name_swap,
    "copy_move": op_copy_move,
    "delete": op_delete,
    "inpaint_rewrite": op_inpaint_rewrite,
}


# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------

def forge_page(image: np.ndarray, words: list[Word], n_variants: int, rng: random.Random,
               fonts: list[tuple[str, str]], digital: bool
               ) -> list[tuple[str, np.ndarray, np.ndarray, str, list[dict]]]:
    """Bir sayfadan ``n_variants`` sahte varyant üretir: (işlem adı, görüntü, maske, iz, düzenlemeler)."""
    variants = []
    attempts = 0
    while len(variants) < n_variants and attempts < n_variants * 6:
        attempts += 1
        ctx = Context(rng, rng.choice(fonts)[0] if fonts else None)
        mask = np.zeros(image.shape[:2], np.uint8)
        current, edits = image, []
        for _ in range(2 if rng.random() < 0.25 else 1):
            name = rng.choices(list(OP_WEIGHTS), weights=list(OP_WEIGHTS.values()))[0]
            result, record = OPERATIONS[name](current, mask, words, ctx)
            if result is not None:
                current, edits = result, edits + [record]
        if not edits or mask.max() == 0:
            continue
        current, trace = harmonize(current, image, mask, digital, rng)
        op_name = edits[0]["op"] if len(edits) == 1 else "multi"
        variants.append((op_name, current, mask, trace, edits))
    return variants


def forge_dataset(src: Path, out: Path, n_variants: int = 3, seed: int = 42, max_side: int = 1600,
                  gpu: bool = False, limit: Optional[int] = None) -> Path:
    """``src`` altındaki tüm belgelerden sahte varyant üretir ve ``metadata.csv`` yolunu döndürür."""
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS | {".pdf"})
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"{src} altında görüntü/PDF bulunamadı.")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    fonts = available_fonts()
    measure_font = load_font(fonts[0][0] if fonts else None, 32)
    reader = get_reader(("tr", "en"), gpu)
    rng = random.Random(seed)
    meta_path = out / "metadata.csv"
    start, n_forged, skipped = time.time(), 0, []
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for fi, path in enumerate(files, 1):
            rel_path = path.relative_to(src)
            rel = rel_path.with_suffix("").as_posix()
            issuer = rel_path.parts[0] if len(rel_path.parts) > 1 else "diger"
            try:
                pages = load_pages(path)
            except Exception as exc:  # Bozuk/şifreli dosya: atla, devam et
                skipped.append(f"{path} ({exc.__class__.__name__}: {exc})")
                continue
            for pi, page in enumerate(pages):
                page, _ = limit_long_side(page, None, max_side)
                digital = is_digital(page)
                words = detect_words(page, reader, measure_font)
                variants = [("orig", page, np.zeros(page.shape[:2], np.uint8), "", [])]
                variants += forge_page(page, words, n_variants, rng, fonts, digital)
                # Gruptaki tüm görüntüler (orijinal dahil) aynı son kanaldan/kaliteyle kaydedilir.
                if digital:
                    channel = sample_digital_channel(rng)
                    quality = channel["quality"]
                else:
                    channel, quality = {"name": "photo"}, rng.randint(80, 95)
                ext = ".jpg" if quality else ".png"
                stem_base = re.sub(r"[^\w\-]+", "_", rel)
                for vi, (op_name, image, mask, trace, edits) in enumerate(variants):
                    if digital:
                        image, mask = apply_digital_channel(image, mask, channel)
                    stem = f"{stem_base}__p{pi}__{vi:02d}_{op_name}"
                    _save(out / "images" / f"{stem}{ext}", image, quality)
                    _save(out / "masks" / f"{stem}.png", mask)
                    writer.writerow({
                        "image": f"images/{stem}{ext}", "mask": f"masks/{stem}.png", "group_id": f"real:{rel}",
                        "label": int(op_name != "orig"), "tamper_type": op_name, "lang": "tr", "doc_type": issuer,
                        "acquisition": "digital" if digital else "photo", "channel": channel["name"],
                        "edit_trace": trace, "source_file": rel_path.as_posix(), "page": pi,
                        "edits": json.dumps(edits, ensure_ascii=False),
                    })
                    n_forged += int(op_name != "orig")
                if len(variants) == 1:
                    skipped.append(f"{path} s.{pi + 1}: değiştirilebilir alan bulunamadı (yalnız orijinal yazıldı)")
            elapsed = time.time() - start
            print(f"  {fi}/{len(files)} dosya ({elapsed / 60:.1f} dk, "
                  f"kalan ~{elapsed / fi * (len(files) - fi) / 60:.1f} dk)", flush=True)
    print(f"{n_forged} sahte varyant üretildi → {meta_path}")
    for message in skipped:
        print(f"  UYARI: {message}")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerçek belgelerden OCR rehberli sahte varyant üretici")
    parser.add_argument("--src", default="data/7_real_originals", help="Gerçek orijinal belgeler (alt klasör = kurum)")
    parser.add_argument("--out", default="data/8_real_forged")
    parser.add_argument("--variants", type=int, default=3, help="Sayfa başına sahte varyant sayısı")
    parser.add_argument("--max-side", type=int, default=1600, help="Uzun kenar sınırı (train.py --max-side ile aynı)")
    parser.add_argument("--gpu", action="store_true", help="EasyOCR'ı GPU'da çalıştır")
    parser.add_argument("--limit", type=int, default=None, help="Yalnızca ilk N dosya (önizleme)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Hedef klasördeki mevcut veri setinin üzerine yaz")
    args = parser.parse_args()
    if (Path(args.out) / "metadata.csv").exists() and not args.overwrite:
        parser.error(f"{args.out} zaten bir veri seti içeriyor; üzerine yazmak için --overwrite verin.")
    forge_dataset(Path(args.src), Path(args.out), args.variants, args.seed, args.max_side, args.gpu, args.limit)


if __name__ == "__main__":
    main()
