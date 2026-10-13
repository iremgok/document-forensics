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
# 3) Tutar ayrıştırma (TR: 1.234,56 — EN: 1,234.56)
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})(?![\d])|(?<![\d.,])\d+(?:[.,]\d{2})(?![\d])")
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
    "total_math": 0.85,
    "vat_rate": 0.60,
}

#: Matematiksel eşitlik toleransı (kuruş yuvarlama farkları için).
AMOUNT_TOLERANCE = 0.02


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


#: Satır sınıflandırma anahtar kelimeleri (ASCII'ye indirgenmiş). Sıra önemlidir:
#: "ara toplam" ve "genel toplam" ifadeleri "toplam"dan önce denetlenir.
_MATRAH_KEYS = ("ara toplam", "matrah", "subtotal", "sub total", "net amount", "net tutar")
_GRAND_KEYS = ("genel toplam", "grand total", "total due", "odenecek", "amount due")
_VAT_KEYS = ("kdv", "vat", "tax")
_TOTAL_KEYS = ("toplam", "total")


def _contains(text: str, key: str) -> bool:
    """
    ``key`` metinde geçiyor mu? 5+ harfli anahtarlarda **1 harf** ikame hatasına izin verilir.

    OCR özellikle serif fontlarda harfleri karıştırır (ör. ``TOTAL`` → ``TOIAL``,
    ``SUBTOTAL`` → ``SUBTOIAL``). Metin üzerinde anahtar uzunluğunda bir pencere
    kaydırılır ve Hamming mesafesi ≤ 1 ise eşleşme kabul edilir. Kısa anahtarlarda
    (KDV, VAT) tam eşleşme aranır; aksi hâlde yanlış pozitif riski yükselir.
    """
    if key in text:
        return True
    if len(key) < 5:
        return False
    n = len(key)
    return any(sum(a != b for a, b in zip(text[i:i + n], key)) <= 1 for i in range(len(text) - n + 1))


def _classify_line(folded: str) -> Optional[str]:
    """Bir satırın matrah / kdv / toplam satırı olup olmadığını belirler."""
    def has(keys: tuple[str, ...]) -> bool:
        return any(_contains(folded, k) for k in keys)

    if has(_MATRAH_KEYS):
        return "matrah"
    if has(_GRAND_KEYS):
        return "total"
    if has(_VAT_KEYS):
        # "KDV dahil toplam" gibi ifadeler toplam satırıdır.
        return "total" if "dahil" in folded or "incl" in folded else "kdv"
    if has(_TOTAL_KEYS):
        return "total"
    return None


#: Kontrol hanelerinde OCR'ın rakam yerine okuyabileceği harflere (O, I, L, S, B, Z) izin
#: verilir; ``normalize_iban`` bunları rakama çevirir.
_IBAN_RE = re.compile(r"\b([A-Z]{2}\s?[0-9OILSBZ]{2}(?:\s?[A-Z0-9]{2,4}){3,8})\b")
_TCKN_RE = re.compile(r"(?<!\d)([1-9]\d{10})(?!\d)")
_TCKN_KEYS = ("tckn", "t.c", "kimlik", "tc no", "tc kimlik")


def _check_ibans(lines: Sequence[OcrLine], report: ContentReport) -> None:
    """Satırlardaki tüm IBAN adaylarını Mod-97 ile doğrular."""
    for line in lines:
        for m in _IBAN_RE.finditer(line.text.upper()):
            iban = normalize_iban(m.group(1))
            if iban[:2] not in IBAN_LENGTHS:
                continue
            report.fields.setdefault("iban", []).append(iban)
            if iban_is_valid(iban):
                report.findings.append(Finding("iban", PASS, f"✅ IBAN geçerli (Mod-97): {iban}", line.bbox))
            elif len(iban) != IBAN_LENGTHS[iban[:2]]:
                report.findings.append(Finding(
                    "iban", WARN,
                    f"⚠️ IBAN uzunluğu beklenenden farklı ({len(iban)}≠{IBAN_LENGTHS[iban[:2]]}); "
                    f"OCR eksik okumuş olabilir: {iban}", line.bbox))
            else:
                report.findings.append(Finding("iban", FAIL, f"⚠️ IBAN Kontrol Hanesi Geçersiz: {iban}", line.bbox))


def _check_tckns(lines: Sequence[OcrLine], report: ContentReport) -> None:
    """TCKN anahtar kelimesi içeren satırlardaki 11 haneli numaraları doğrular."""
    for line in lines:
        folded = _ascii_fold(line.text)
        if not any(k in folded for k in _TCKN_KEYS):
            continue
        digits_only = re.sub(r"(?<=\d)\s(?=\d)", "", line.text)  # "123 456..." → "123456..."
        for m in _TCKN_RE.finditer(digits_only):
            tckn = m.group(1)
            report.fields.setdefault("tckn", []).append(tckn)
            if tckn_is_valid(tckn):
                report.findings.append(Finding("tckn", PASS, f"✅ TCKN algoritmik olarak geçerli: {tckn}", line.bbox))
            else:
                report.findings.append(Finding("tckn", FAIL, f"⚠️ TCKN algoritması tutmuyor: {tckn}", line.bbox))


def _check_totals(lines: Sequence[OcrLine], report: ContentReport) -> None:
    """
    Matrah + KDV == Genel Toplam ve Matrah × KDV oranı == KDV kurallarını denetler.

    Her alan için belgede **son** görülen değer kullanılır (fişlerde ara satırlar
    önce, nihai toplam en sonda yazılır).
    """
    values: dict[str, tuple[float, OcrLine]] = {}
    rate: Optional[float] = None
    for line in lines:
        folded = _ascii_fold(line.text)
        kind = _classify_line(folded)
        if kind is None:
            continue
        amounts = find_amounts(line.text)
        if not amounts:
            continue
        values[kind] = (amounts[-1], line)
        if kind == "kdv":
            m = _RATE_RE.search(line.text)
            if m:
                rate = float(m.group(1) or m.group(2)) / 100.0

    report.fields.update({k: v[0] for k, v in values.items()})
    if rate is not None:
        report.fields["kdv_orani"] = rate

    if {"matrah", "kdv", "total"} <= values.keys():
        matrah, kdv, total = (values[k][0] for k in ("matrah", "kdv", "total"))
        diff = abs(matrah + kdv - total)
        bbox = values["total"][1].bbox
        if diff <= AMOUNT_TOLERANCE:
            report.findings.append(Finding(
                "total_math", PASS,
                f"✅ Tutar Matematiksel Olarak Tutarlı: {matrah:,.2f} + {kdv:,.2f} = {total:,.2f}", bbox))
        else:
            report.findings.append(Finding(
                "total_math", FAIL,
                f"⚠️ Matrah + KDV ≠ Genel Toplam: {matrah:,.2f} + {kdv:,.2f} = {matrah + kdv:,.2f} "
                f"(belgede {total:,.2f}, fark {diff:,.2f})", bbox))
    elif values:
        missing = {"matrah", "kdv", "total"} - values.keys()
        report.findings.append(Finding(
            "total_math", WARN, f"ℹ️ Tutar kontrolü için eksik alan(lar): {', '.join(sorted(missing))}"))

    if rate is not None and {"matrah", "kdv"} <= values.keys():
        matrah, kdv = values["matrah"][0], values["kdv"][0]
        expected = round(matrah * rate, 2)
        bbox = values["kdv"][1].bbox
        if abs(expected - kdv) <= max(AMOUNT_TOLERANCE, 0.005 * expected):
            report.findings.append(Finding("vat_rate", PASS, f"✅ KDV oranı tutarlı (%{rate * 100:.0f})", bbox))
        else:
            report.findings.append(Finding(
                "vat_rate", FAIL,
                f"⚠️ KDV oranı tutmuyor: {matrah:,.2f} × %{rate * 100:.0f} = {expected:,.2f}, belgede {kdv:,.2f}",
                bbox))


def check_document_text(lines: Sequence[OcrLine]) -> ContentReport:
    """Tüm kuralları çalıştırır ve birleşik raporu döndürür."""
    report = ContentReport()
    _check_ibans(lines, report)
    _check_tckns(lines, report)
    _check_totals(lines, report)
    return report


def lines_from_text(text: str) -> list[OcrLine]:
    """Düz metni (ör. testlerde) OCR satırlarına çevirir; kutular boş bırakılır."""
    return [OcrLine(t, (0, 0, 0, 0), 1.0) for t in text.splitlines() if t.strip()]
