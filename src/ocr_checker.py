"""
OCR tabanlı içerik tutarlılık modülü.

Bu modül iki katmandan oluşur:

1. **OCR katmanı** (`run_ocr`): EasyOCR ile belgedeki kelime kutularını okur ve
   aynı satırdaki kutuları birleştirerek satır listesi üretir.
2. **Kural motoru** (`check_document_text`): Satırlardan IBAN, TCKN ve
   Matrah/KDV/Genel Toplam alanlarını Regex ile ayıklar ve matematiksel olarak
   doğrular.

Doğrulama algoritmaları (Mod-97, TCKN) saf Python ile yazılmıştır; bu nedenle
sentetik veri üreticisi (`src/generate_synthetic.py`) de geçerli IBAN/TCKN
üretmek için aynı fonksiyonları kullanır (tek doğruluk kaynağı).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 1) IBAN — ISO 13616 / ISO 7064 Mod-97-10
# ---------------------------------------------------------------------------

#: Ülke koduna göre toplam IBAN uzunlukları (en sık karşılaşılan ülkeler).
IBAN_LENGTHS: dict[str, int] = {
    "TR": 26, "DE": 22, "GB": 22, "FR": 27, "NL": 18, "IT": 27, "ES": 24,
    "BE": 16, "AT": 20, "CH": 21, "IE": 22, "PL": 28, "PT": 25, "SE": 24,
}


def _iban_to_digits(rearranged: str) -> str:
    """Harfleri ISO 13616 kuralına göre sayıya çevirir (A=10, B=11, ..., Z=35)."""
    return "".join(str(int(ch, 36)) for ch in rearranged)


def _mod97(digits: str) -> int:
    """
    Çok uzun bir sayının 97'ye göre kalanını parça parça hesaplar.

    `int(digits) % 97` Python'da da çalışır; ancak burada algoritmanın kendisi
    gösterilmektedir: Sayı soldan 9'ar haneli bloklar hâlinde okunur ve her adımda
    `kalan = (önceki_kalan || blok) mod 97` hesaplanır. Modüler aritmetikte
    (a·10^k + b) mod 97 = ((a mod 97)·10^k + b) mod 97 olduğundan sonuç aynıdır.
    """
    remainder = 0
    for start in range(0, len(digits), 9):
        block = digits[start:start + 9]
        remainder = int(f"{remainder}{block}") % 97
    return remainder


_DIGIT_FIXES = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"})


def normalize_iban(raw: str) -> str:
    """Boşlukları siler, büyük harfe çevirir ve sık OCR karışıklıklarını düzeltir.

    Kontrol haneleri (3-4. karakter) her ülkede rakamdır; TR IBAN'larında ise
    ülke kodundan sonraki her şey rakamdır. Bu konumlarda 'O'→'0', 'I/L'→'1',
    'S'→'5', 'B'→'8', 'Z'→'2' düzeltmesi yapılır (ör. OCR çıktısı ``TRO8`` → ``TR08``).
    Diğer ülkelerde BBAN harf içerebileceği için orada düzeltme yapılmaz.
    """
    iban = re.sub(r"[\s\-]", "", raw).upper()
    end = len(iban) if iban.startswith("TR") else 4
    return iban[:2] + iban[2:end].translate(_DIGIT_FIXES) + iban[end:]


def iban_is_valid(iban: str) -> bool:
    """
    IBAN Mod-97 doğrulaması.

    Algoritma:
      1. İlk 4 karakter (ülke kodu + 2 kontrol hanesi) sona taşınır.
         Örn. ``TR33 0006 1005 ...`` → ``00061005...TR33``
      2. Harfler sayıya çevrilir (T=29, R=27).
      3. Elde edilen büyük sayı 97'ye bölündüğünde kalan **1** ise IBAN geçerlidir.

    Bu yöntem tek hane hatalarının ve yan yana iki hanenin yer değiştirmesinin
    tamamını yakalar; bu yüzden sahte belgede değiştirilen tek bir rakam bile
    tespit edilir.
    """
    iban = normalize_iban(iban)
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{8,30}", iban):
        return False
    expected_len = IBAN_LENGTHS.get(iban[:2])
    if expected_len is not None and len(iban) != expected_len:
        return False
    return _mod97(_iban_to_digits(iban[4:] + iban[:4])) == 1


def compute_iban_check_digits(country: str, bban: str) -> str:
    """Verilen ülke kodu ve BBAN için geçerli 2 haneli kontrol değerini hesaplar.

    Kontrol hanesi = 98 − (BBAN + ülke + "00" sayısının mod 97'si).
    """
    remainder = _mod97(_iban_to_digits(bban + country + "00"))
    return f"{98 - remainder:02d}"


# ---------------------------------------------------------------------------
# 2) TCKN — T.C. Kimlik Numarası algoritması
# ---------------------------------------------------------------------------

def tckn_is_valid(tckn: str) -> bool:
    """
    T.C. Kimlik Numarası doğrulaması (d1..d11 haneler).

    Kurallar:
      * 11 hanelidir ve d1 ≠ 0.
      * d10 = ((d1 + d3 + d5 + d7 + d9) · 7 − (d2 + d4 + d6 + d8)) mod 10
      * d11 = (d1 + d2 + ... + d10) mod 10
    """
    if not re.fullmatch(r"[1-9]\d{10}", tckn):
        return False
    d = [int(c) for c in tckn]
    odd_sum = d[0] + d[2] + d[4] + d[6] + d[8]
    even_sum = d[1] + d[3] + d[5] + d[7]
    if (odd_sum * 7 - even_sum) % 10 != d[9]:
        return False
    return sum(d[:10]) % 10 == d[10]


def complete_tckn(first_nine: str) -> str:
    """İlk 9 haneden geçerli bir 11 haneli TCKN üretir (sentetik veri için)."""
    d = [int(c) for c in first_nine]
    d10 = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    d11 = (sum(d) + d10) % 10
    return f"{first_nine}{d10}{d11}"


# ---------------------------------------------------------------------------
# 2b) VKN — 10 haneli Vergi Kimlik Numarası (GİB algoritması)
# ---------------------------------------------------------------------------

def _vkn_check_digit(first_nine: str) -> int:
    """
    VKN'nin 10. hanesini (kontrol hanesi) hesaplar.

    i = 1..9 için: t = (d_i + 10 − i) mod 10; t = 9 ise katkı 9, aksi hâlde
    katkı = (t · 2^(10−i)) mod 9 (t ≠ 0 iken sonuç 0 çıkarsa 9 alınır).
    Kontrol hanesi = (10 − Σ katkı mod 10) mod 10.
    """
    total = 0
    for i, ch in enumerate(first_nine, start=1):
        t = (int(ch) + 10 - i) % 10
        v = 9 if t == 9 else (t * 2 ** (10 - i)) % 9
        if t != 0 and v == 0:
            v = 9
        total += v
    return (10 - total % 10) % 10


def vkn_is_valid(vkn: str) -> bool:
    """10 haneli VKN doğrulaması. Şahıs şirketlerinde VKN yerine 11 haneli TCKN kullanılır."""
    return bool(re.fullmatch(r"\d{10}", vkn)) and _vkn_check_digit(vkn[:9]) == int(vkn[9])


def complete_vkn(first_nine: str) -> str:
    """İlk 9 haneden geçerli bir VKN üretir (sentetik veri için)."""
    return f"{first_nine}{_vkn_check_digit(first_nine)}"


# ---------------------------------------------------------------------------
# 3) Tutar ayrıştırma (TR: 1.234,56 — EN: 1,234.56)
# ---------------------------------------------------------------------------

#: Ayraçtan sonra tek boşluğa izin verilir: OCR "2.691,61"i sık sık "2. 691,61" ya da "691, 61" diye okur;
#: aksi hâlde yalnızca "691,61" yakalanır ve orijinal belgede "Matrah + KDV ≠ Toplam" yanlış alarmı oluşur.
#: Arkasından ".rakam" gelen eşleşmeler (ör. "20.03.2021" tarihi) tutar sayılmaz.
_AMOUNT_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:(?:[.,] ?|\s)\d{3})*(?:[.,] ?\d{2})(?![\d]|[.,]\d)"
                        r"|(?<![\d.,])\d+(?:[.,] ?\d{2})(?![\d]|[.,]\d)")
_RATE_RE = re.compile(r"%\s*(\d{1,2})|(\d{1,2})\s*%")


def parse_amount(text: str) -> Optional[float]:
    """
    Türkçe veya İngilizce biçimli bir tutarı float'a çevirir.

    Ondalık ayırıcı, **son** ayırıcının ardından tam iki hane gelmesiyle
    belirlenir: ``1.234,56`` → 1234.56 (TR), ``1,234.56`` → 1234.56 (EN).
    Diğer tüm ayırıcılar binlik ayırıcı kabul edilip silinir.
    """
    text = text.strip().replace(" ", "")
    m = re.fullmatch(r"(.*)[.,](\d{2})", text)
    if not m:
        return None
    integer_part = re.sub(r"[.,]", "", m.group(1))
    if not integer_part.isdigit():
        return None
    return float(f"{integer_part}.{m.group(2)}")


def find_amounts(line: str) -> list[float]:
    """Bir satırdaki tüm parasal değerleri soldan sağa döndürür."""
    # Yüzde oranlarını (%20) tutar sanmamak için önce siliyoruz.
    cleaned = _RATE_RE.sub(" ", line)
    values = [parse_amount(m.group(0)) for m in _AMOUNT_RE.finditer(cleaned)]
    return [v for v in values if v is not None]


# ---------------------------------------------------------------------------
# 4) OCR katmanı (EasyOCR)
# ---------------------------------------------------------------------------

@dataclass
class OcrLine:
    """Aynı satırda bulunan OCR kelimelerinin birleşimi."""

    text: str
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) piksel
    confidence: float


@lru_cache(maxsize=2)
def get_reader(languages: tuple[str, ...] = ("tr", "en"), gpu: bool = False) -> Any:
    """EasyOCR okuyucusunu bir kez oluşturur ve önbellekte tutar (model yükleme pahalıdır)."""
    import easyocr  # Ağır bağımlılık: yalnızca OCR gerçekten çağrıldığında yüklenir.

    return easyocr.Reader(list(languages), gpu=gpu, verbose=False)


def group_into_lines(words: Sequence[tuple[Any, str, float]]) -> list[OcrLine]:
    """
    EasyOCR kelime kutularını satırlara gruplar.

    İki kutunun dikey merkezleri arasındaki fark, medyan kutu yüksekliğinin
    yarısından küçükse aynı satırdadır. Satır içi kelimeler x'e göre sıralanır.
    Bu, "Genel Toplam" etiketi ile tutarın ayrı sütunlarda okunduğu faturalarda
    etiket-değer eşleşmesini mümkün kılar.
    """
    if not words:
        return []
    boxes = []
    for quad, text, conf in words:
        pts = np.asarray(quad, dtype=float)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        boxes.append((x0, y0, x1, y1, text, float(conf)))
    median_h = float(np.median([b[3] - b[1] for b in boxes]))
    boxes.sort(key=lambda b: (b[1] + b[3]) / 2)

    lines: list[list[tuple]] = []
    for box in boxes:
        cy = (box[1] + box[3]) / 2
        if lines:
            last = lines[-1]
            last_cy = float(np.mean([(b[1] + b[3]) / 2 for b in last]))
            if abs(cy - last_cy) < 0.5 * median_h:
                last.append(box)
                continue
        lines.append([box])

    result = []
    for members in lines:
        members.sort(key=lambda b: b[0])
        result.append(OcrLine(
            text=" ".join(b[4] for b in members),
            bbox=(int(min(b[0] for b in members)), int(min(b[1] for b in members)),
                  int(max(b[2] for b in members)), int(max(b[3] for b in members))),
            confidence=float(np.mean([b[5] for b in members])),
        ))
    return result


def run_ocr(image_rgb: np.ndarray, languages: tuple[str, ...] = ("tr", "en"),
            gpu: bool = False) -> list[OcrLine]:
    """Bir RGB görüntüyü okuyup satır listesi döndürür."""
    reader = get_reader(languages, gpu)
    words = reader.readtext(image_rgb, detail=1, paragraph=False)
    return group_into_lines(words)


# ---------------------------------------------------------------------------
# 5) Kural motoru
# ---------------------------------------------------------------------------

PASS, FAIL, WARN = "pass", "fail", "warn"

#: Başarısız kuralın sahtecilik riskine katkısı (0-1). Mod-97 ve TCKN algoritmik
#: olarak kesin olduğu için yüksek; OCR okuma hatası olasılığı nedeniyle < 1.
RULE_WEIGHTS: dict[str, float] = {
    "iban": 0.90,
    "tckn": 0.70,
    "vkn": 0.70,
    "total_math": 0.85,
    "vat_rate": 0.60,
}

#: Matematiksel eşitlik toleransı (kuruş yuvarlama farkları için).
AMOUNT_TOLERANCE = 0.02

#: Ortalama OCR güveni bunun altındaki satırlarda kural ihlali FAIL yerine WARN olur:
#: hatalı okunmuş bir rakam, sahte belge kanıtı sayılmamalıdır.
MIN_OCR_CONFIDENCE = 0.60


@dataclass
class Finding:
    """Tek bir kural kontrolünün sonucu."""

    rule: str
    status: str  # pass / fail / warn
    message: str
    bbox: Optional[tuple[int, int, int, int]] = None


@dataclass
class ContentReport:
    """Belgenin içerik tutarlılık raporu."""

    findings: list[Finding] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def has_evidence(self) -> bool:
        """En az bir kural pass/fail sonucuna ulaşabildi mi?"""
        return any(f.status in (PASS, FAIL) for f in self.findings)

    @property
    def risk(self) -> float:
        """
        İçerik risk skoru (0-1), "noisy-OR" birleşimi ile:

            risk = 1 − Π (1 − w_i)   (yalnızca başarısız kurallar için)

        Her başarısız kural bağımsız bir sahtecilik kanıtı sayılır; iki kural
        birden bozulduğunda risk tek kuralınkinden daha yüksek olur.
        """
        survive = 1.0
        for f in self.findings:
            if f.status == FAIL:
                survive *= 1.0 - RULE_WEIGHTS.get(f.rule, 0.5)
        return 1.0 - survive


def _ascii_fold(text: str) -> str:
    """Türkçe karakterleri ASCII'ye indirger ve küçük harfe çevirir (anahtar kelime eşleşmesi için).

    'İ'.lower() Python'da 'i̇' (noktalı) ürettiği için önce elle dönüştürülür.
    """
    text = text.replace("İ", "i").replace("I", "i").replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


#: Satır türleri ve anahtar kelimeleri (ASCII'ye indirgenmiş). **Sıra önemlidir:** ilk eşleşen
#: tür kazanır; bu yüzden "toplam iskonto", "toplam kdv", "kdv dahil toplam" gibi bileşik
#: ifadeler yalın "toplam"dan önce denetlenir. e-Arşiv / e-Fatura etiketleri dahildir:
#:   Mal Hizmet Toplam Tutarı (gross) − Toplam İskonto (discount) = Matrah
#:   Matrah + Σ Hesaplanan KDV = Vergiler Dahil Toplam Tutar (total) [= Ödenecek Tutar]
_LINE_KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("discount", ("iskonto", "indirim", "discount")),
    ("total", ("vergiler dahil", "kdv dahil", "vergi dahil", "genel toplam", "odenecek",
               "grand total", "total due", "amount due", "incl. vat", "incl vat")),
    ("matrah", ("ara toplam", "matrah", "vergiye tabi", "kdv haric", "subtotal", "sub total",
                "net amount", "net tutar")),
    ("gross", ("mal hizmet toplam", "mal/hizmet toplam", "mal ve hizmet toplam")),
    ("kdv_total", ("toplam kdv", "kdv toplam", "total vat", "vat total")),
    ("kdv", ("hesaplanan kdv", "kdv", "vat", "tax")),
    ("total", ("toplam tutar", "toplam", "total")),
)

#: Kısa ve genel anahtarlar: tutarı bir alt satırda aramak (bkz. ``_check_totals``) için fazla
#: belirsizdir (ör. kalem tablosunun "Toplam" sütun başlığı).
_GENERIC_KEYS = {"toplam", "total", "kdv", "vat", "tax"}


def _levenshtein_le(a: str, b: str, limit: int) -> bool:
    """``a`` ile ``b`` arasındaki düzenleme mesafesi ``limit``'ten küçük-eşit mi? (kısa dizeler için)"""
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit


def _contains(text: str, key: str) -> bool:
    """
    ``key`` metinde geçiyor mu? OCR hatalarına toleranslı eşleşme.

    * 5-9 harfli anahtarlarda 1, 10+ harflilerde 2 **düzenleme** (ikame, eksik ya da fazla
      harf) hatasına izin verilir: ``TOIAL`` → total, ``TOPLM`` → toplam,
      ``Vergiler Dahl`` → vergiler dahil.
    * Kısa anahtarlarda (KDV, VAT) tam eşleşme aranır; aksi hâlde yanlış pozitif riski yükselir.
    """
    if key in text:
        return True
    n = len(key)
    if n < 5:
        return False
    limit = 1 if n < 10 else 2
    for size in range(n - limit, n + limit + 1):
        for i in range(len(text) - size + 1):
            window = text[i:i + size]
            if window[0] == key[0] and _levenshtein_le(window, key, limit):
                return True
    return False


def _classify_line(folded: str) -> Optional[tuple[str, str]]:
    """Satırın türünü ve eşleşen anahtarı döndürür: (matrah | gross | discount | kdv | kdv_total | total, anahtar)."""
    for kind, keys in _LINE_KINDS:
        for key in keys:
            if _contains(folded, key):
                return kind, key
    return None


def _status(ok: bool, *lines: OcrLine) -> str:
    """Kural sonucu: geçerse PASS; geçmezse OCR güveni yeterliyse FAIL, değilse WARN."""
    if ok:
        return PASS
    confident = all(line.confidence >= MIN_OCR_CONFIDENCE for line in lines)
    return FAIL if confident else WARN


# --- IBAN ---------------------------------------------------------------------------

#: OCR'ın rakam yerine okuyabileceği harfler; ``normalize_iban`` bunları rakama çevirir.
_DIGITISH = "0-9OILSBZ"
#: TR IBAN adayı: "TR" + 24 hane (aralarda boşluk/tire olabilir). OCR eksik/fazla okumasını
#: yakalamak için 20-28 hane kabul edilir; uzunluk ayrıca denetlenir.
_TR_IBAN_RE = re.compile(rf"(?<![A-Z0-9])T\s?R\s?((?:[{_DIGITISH}][\s\-]?){{20,28}})(?![{_DIGITISH}])")
#: Yabancı IBAN adayı: yalnızca satırda "IBAN" etiketi varsa aranır.
_FOREIGN_IBAN_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2}\s?\d{2}(?:\s?[A-Z0-9]{2,4}){3,8})(?![A-Z0-9])")


def _tr_iban_candidates(text: str) -> list[str]:
    """Metindeki TR IBAN adaylarını normalize edilmiş hâlde döndürür.

    Yanlış pozitifleri önlemek için gövdedeki karakterlerin en az %80'i OCR'dan **gerçek
    rakam** olarak gelmiş olmalıdır: ``TRENDYOL`` ya da ``BE11NTINIZ`` gibi kelimeler,
    harf→rakam düzeltmesiyle IBAN'a dönüştürülemez.
    """
    out = []
    for m in _TR_IBAN_RE.finditer(re.sub(r"\s+", " ", text.upper())):
        body = re.sub(r"[\s\-]", "", m.group(1))
        if sum(c.isdigit() for c in body) / len(body) >= 0.8:
            out.append(normalize_iban("TR" + body))
    return out


def _check_ibans(lines: Sequence[OcrLine], report: ContentReport) -> None:
    """
    IBAN adaylarını Mod-97 ile doğrular.

    * **TR IBAN:** "TR" + rakam ağırlıklı 24 hane. Uzunluk 26 değilse ve satırda "IBAN"
      etiketi varsa WARN (OCR eksik okumuş olabilir); etiket yoksa aday yok sayılır.
    * **Yabancı IBAN:** yalnızca satırda "IBAN" etiketi varsa aranır ve rakam oranı ≥ %50
      olmalıdır. Türkçe metindeki ``IT18ARIYLE`` (itibarıyle) gibi kelimeler böylece elenir.
    * IBAN satır sonunda bölünmüşse (etiketli satırda eksik aday), sonraki satırla birleştirilir.
    """
    seen: set[str] = set()

    def add(iban: str, line: OcrLine, labelled: bool) -> None:
        if iban in seen:
            return
        expected = IBAN_LENGTHS.get(iban[:2])
        if expected is None:
            return
        if len(iban) != expected:
            if labelled:
                seen.add(iban)
                report.findings.append(Finding(
                    "iban", WARN, f"ℹ️ IBAN eksik/fazla okunmuş olabilir ({len(iban)}≠{expected}): {iban}", line.bbox))
            return
        seen.add(iban)
        report.fields.setdefault("iban", []).append(iban)
        if iban_is_valid(iban):
            report.findings.append(Finding("iban", PASS, f"✅ IBAN geçerli (Mod-97): {iban}", line.bbox))
        else:
            report.findings.append(Finding("iban", _status(False, line),
                                           f"⚠️ IBAN Kontrol Hanesi Geçersiz: {iban}", line.bbox))

    for idx, line in enumerate(lines):
        labelled = "iban" in _ascii_fold(line.text)
        candidates = _tr_iban_candidates(line.text)
        if labelled and not any(len(c) == 26 for c in candidates) and idx + 1 < len(lines):
            joined = _tr_iban_candidates(f"{line.text} {lines[idx + 1].text}")
            candidates = [c for c in joined if len(c) == 26] or candidates
        for iban in candidates:
            add(iban, line, labelled)
        if labelled:
            for m in _FOREIGN_IBAN_RE.finditer(line.text.upper()):
                raw = re.sub(r"\s", "", m.group(1))
                if raw.startswith("TR") or sum(c.isdigit() for c in raw) / len(raw) < 0.5:
                    continue
                add(normalize_iban(raw), line, labelled)


# --- TCKN / VKN -----------------------------------------------------------------------

#: Kimlik etiketleri (ASCII'ye indirgenmiş metinde). Etiketin hemen ardındaki numara okunur.
_ID_LABEL_RE = re.compile(
    r"(vkn\s*/\s*tckn|tckn\s*/\s*vkn|vkn|tckn|vergi\s*(?:kimlik)?\s*(?:no|numarasi)|"
    r"t\.?\s?c\.?\s*kimlik\s*(?:no|numarasi)?|kimlik\s*(?:no|numarasi)|tc\s*no)")
#: Etiketten sonra ilk numara: önce boşluksuz 10-11 hane, yoksa tek boşluklarla bölünmüş
#: ("123 456 789 01") ve tam olarak 10-11 haneden oluşan bir grup. Komşu sayılar
#: (ör. "1234567890 5 adet") birleştirilip sahte bir 11 haneli TCKN üretilmez.
_ID_PLAIN_RE = re.compile(r"(?<!\d)(\d{10,11})(?!\d)")
_ID_SPACED_RE = re.compile(r"(?<!\d)(?<!\d )(\d{2,4}(?: \d{2,4}){2,4})(?! ?\d)")


def _first_id_after(text: str) -> Optional[str]:
    """Metindeki ilk 10-11 haneli kimlik numarası adayını döndürür."""
    candidates = [m for m in (_ID_PLAIN_RE.search(text), _ID_SPACED_RE.search(text)) if m]
    for m in sorted(candidates, key=lambda m: m.start()):
        digits = m.group(1).replace(" ", "")
        if len(digits) in (10, 11):
            return digits
    return None


def _check_ids(lines: Sequence[OcrLine], report: ContentReport) -> None:
    """
    Etiketli kimlik numaralarını **hane sayısına göre** doğrular:

    * 10 hane → VKN algoritması (kurumlar),
    * 11 hane → TCKN algoritması (kişiler; şahıs şirketlerinde "VKN" etiketiyle de yazılır).

    Yalnızca bir etiketin (VKN, TCKN, Vergi No, T.C. Kimlik No...) en fazla 40 karakter
    sonrasındaki ilk numara okunur; etiketsiz 10-11 haneli sayılar (sipariş no, ETTN,
    telefon) denetlenmez.
    """
    seen: set[str] = set()
    for line in lines:
        folded = _ascii_fold(line.text)
        for label in _ID_LABEL_RE.finditer(folded):
            number = _first_id_after(folded[label.end():label.end() + 40])
            if number is None or number in seen:
                continue
            seen.add(number)
            rule, valid = ("vkn", vkn_is_valid) if len(number) == 10 else ("tckn", tckn_is_valid)
            name = rule.upper()
            ok = valid(number)
            report.fields.setdefault(rule, []).append(number)
            report.findings.append(Finding(
                rule, _status(ok, line),
                f"✅ {name} algoritmik olarak geçerli: {number}" if ok else f"⚠️ {name} algoritması tutmuyor: {number}",
                line.bbox))


# --- Tutarlar ---------------------------------------------------------------------------

def _check_totals(lines: Sequence[OcrLine], report: ContentReport) -> None:
    """
    Tutar tutarlılığı:

        matrah = Ara Toplam / Matrah        (yoksa: Mal Hizmet Toplamı − İskonto)
        matrah + Σ KDV == Genel Toplam / Vergiler Dahil Toplam / Ödenecek Tutar
        tek KDV oranı varsa: matrah × oran == KDV

    * Her tür için belgede **son** görülen değer kullanılır (nihai toplam en sonda yazılır).
    * Farklı oranlı KDV satırları (%1, %10, %20) toplanır; "Toplam KDV" satırı varsa o kullanılır.
    * Etiket satırında tutar yoksa (OCR etiketi ve tutarı ayrı satıra bölmüşse) hemen sonraki
      **etiketsiz** satırdaki tutar alınır; bu yalnızca özgül etiketlerde yapılır.
    """
    values: dict[str, tuple[float, OcrLine]] = {}
    kdv_by_rate: dict[Optional[float], tuple[float, OcrLine]] = {}
    for idx, line in enumerate(lines):
        match = _classify_line(_ascii_fold(line.text))
        if match is None:
            continue
        kind, key = match
        source = line
        amounts = find_amounts(line.text)
        if not amounts and key not in _GENERIC_KEYS and idx + 1 < len(lines):
            nxt = lines[idx + 1]
            if _classify_line(_ascii_fold(nxt.text)) is None:
                amounts, source = find_amounts(nxt.text), nxt
        if not amounts:
            continue
        if kind == "kdv":
            m = _RATE_RE.search(line.text)
            rate = float(m.group(1) or m.group(2)) / 100.0 if m else None
            kdv_by_rate[rate] = (amounts[-1], source)
        else:
            values[kind] = (amounts[-1], source)

    kdv_value: Optional[float] = None
    kdv_line: Optional[OcrLine] = None
    if "kdv_total" in values:
        kdv_value, kdv_line = values.pop("kdv_total")
    elif kdv_by_rate:
        kdv_value = round(sum(v for v, _ in kdv_by_rate.values()), 2)
        kdv_line = list(kdv_by_rate.values())[-1][1]

    if "matrah" not in values and "gross" in values:
        gross, gross_line = values["gross"]
        discount = values.get("discount", (0.0, gross_line))[0]
        values["matrah"] = (round(gross - discount, 2), gross_line)

    report.fields.update({k: v[0] for k, v in values.items()})
    if kdv_value is not None:
        report.fields["kdv"] = kdv_value
    rates = [r for r in kdv_by_rate if r is not None]
    if len(rates) == 1:
        report.fields["kdv_orani"] = rates[0]

    have = set(values) | ({"kdv"} if kdv_value is not None else set())
    if {"matrah", "kdv", "total"} <= have:
        (matrah, m_line), (total, t_line) = values["matrah"], values["total"]
        diff = abs(matrah + kdv_value - total)
        ok = diff <= AMOUNT_TOLERANCE
        report.findings.append(Finding(
            "total_math", _status(ok, m_line, kdv_line, t_line),
            f"✅ Tutar Matematiksel Olarak Tutarlı: {matrah:,.2f} + {kdv_value:,.2f} = {total:,.2f}" if ok else
            f"⚠️ Matrah + KDV ≠ Genel Toplam: {matrah:,.2f} + {kdv_value:,.2f} = {matrah + kdv_value:,.2f} "
            f"(belgede {total:,.2f}, fark {diff:,.2f})", t_line.bbox))
    elif have & {"matrah", "kdv", "total"}:
        missing = {"matrah", "kdv", "total"} - have
        report.findings.append(Finding(
            "total_math", WARN, f"ℹ️ Tutar kontrolü için eksik alan(lar): {', '.join(sorted(missing))}"))

    if len(rates) == 1 and len(kdv_by_rate) == 1 and "matrah" in values:
        rate = rates[0]
        matrah, m_line = values["matrah"]
        expected = round(matrah * rate, 2)
        ok = abs(expected - kdv_value) <= max(AMOUNT_TOLERANCE, 0.005 * expected)
        report.findings.append(Finding(
            "vat_rate", _status(ok, m_line, kdv_line),
            f"✅ KDV oranı tutarlı (%{rate * 100:.0f})" if ok else
            f"⚠️ KDV oranı tutmuyor: {matrah:,.2f} × %{rate * 100:.0f} = {expected:,.2f}, belgede {kdv_value:,.2f}",
            kdv_line.bbox))


def check_document_text(lines: Sequence[OcrLine]) -> ContentReport:
    """Tüm kuralları çalıştırır ve birleşik raporu döndürür."""
    report = ContentReport()
    _check_ibans(lines, report)
    _check_ids(lines, report)
    _check_totals(lines, report)
    return report


def lines_from_text(text: str) -> list[OcrLine]:
    """Düz metni (ör. testlerde) OCR satırlarına çevirir; kutular boş bırakılır."""
    return [OcrLine(t, (0, 0, 0, 0), 1.0) for t in text.splitlines() if t.strip()]
