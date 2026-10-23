"""
Türkçe / İngilizce sentetik finansal belge ve sahtecilik maskesi üreticisi.

Üretilen belge türleri: banka dekontu, fatura ve perakende fişi (TR: TL, EN: USD/EUR).
Her şablon örneği bir **grup** oluşturur: 1 orijinal + N sahte varyant. Sızıntısız
ayırıcı (``src/dataset.py``) bu grup kimliğini kullanır.

Gerçekçi iz zinciri (tüm varyantlarda aynı)
-------------------------------------------
1. Belge Pillow ile temiz çizilir.
2. "Edinim" simülasyonu: kâğıt tonu, Gauss gürültüsü, hafif bulanıklık, JPEG(q1).
   → Bu, sahtecinin elindeki "orijinal" taranmış/kaydedilmiş belgedir.
3. Sahte varyantlarda düzenleme yapılır (aşağıdaki işlemler).
4. Hem orijinal hem sahte varyant **aynı** son JPEG(q2) kaydından geçer. Böylece
   model "kaç kez sıkıştırıldı" gibi kestirme bir ipucu öğrenemez; yalnızca
   düzenlenen bölgedeki yerel tutarsızlıkları öğrenebilir.

Sahtecilik işlemleri (maske: 255 = sahte piksel)
------------------------------------------------
* ``copy_move``     : Belgedeki bir rakam glifi kopyalanıp başka bir tutarın rakamının üzerine yapıştırılır.
* ``text_rewrite``  : Alan arka plan rengiyle silinip yeni değer (farklı font/boyut kayması ile) yazılır.
* ``amount_tamper`` : Genel toplam değiştirilir → Matrah + KDV ≠ Toplam (OCR kuralı yakalar).
* ``amount_consistent``: Matrah, KDV ve Toplam birlikte, matematiği bozmadan değiştirilir
                      (yalnızca piksel izleriyle yakalanabilir → hibrit sistemin gerekçesi).
* ``iban_tamper``   : IBAN'ın 1-2 rakamı karakter düzeyinde değiştirilir → Mod-97 bozulur.
* ``inpaint_sim``   : Üretken YZ inpainting benzetimi — bölge ``cv2.inpaint`` ile doldurulup
                      yeni metin yazılır ve bölge yumuşatılır (difüzyon çıktılarındaki düşük
                      yüksek-frekans gürültüsünü taklit eder).

Dijital mod (``--digital-ratio``)
--------------------------------
Taranmış kâğıt yerine **dijital doğmuş** belge (e-Arşiv PDF, uygulama ekran görüntüsü) üretir:
kâğıt tonu, gürültü, bulanıklık ve ilk JPEG **uygulanmaz**; beyaz zemin ve keskin tipografi
korunur. Son adımda gerçek iletim kanallarından biri grup içindeki tüm varyantlara aynı
parametrelerle uygulanır (bkz. ``DIGITAL_CHANNELS``): kayıpsız PNG, yüksek kaliteli JPEG,
ekran görüntüsü (ölçekleme) veya WhatsApp (küçültme + JPEG). Böylece model "temiz zemin =
sahte" ya da "JPEG = sahte" gibi kestirme ipuçları öğrenemez.

Dijital belgede beyaz zemin beyazla boyanıp aynı fontla yazılınca sahte alan gerçek metinle
piksel düzeyinde aynı olur (öğrenilemez etiket → model alan konumunu ezberler ve orijinallerde
yanlış alarm verir). Bu yüzden dijital modda her sahte bölgeye ``add_edit_trace`` ile gerçekçi
bir yerel iz eklenir (bkz. ``EDIT_TRACES``); seçilen iz ``metadata.csv``'de ``edit_trace`` sütunundadır.

Kullanım:
    python -m src.generate_synthetic --out data/6_synthetic_digital --num-docs 1500 --digital-ratio 1.0
    python -m src.generate_synthetic --digital-ratio 0.5      # yarı taranmış, yarı dijital
    python -m src.generate_synthetic --out data/6_synthetic_tr_en_v4 --num-images 12000 --digital-ratio 0.5
    python -m src.generate_synthetic                          # 2000 belge × 4 = ~8000 görüntü (Stage C)
    python -m src.generate_synthetic --num-images 10000       # hedef görüntü sayısından belge sayısı
    python -m src.generate_synthetic --out data/4_synthetic_tr_en --num-docs 1000 --variants 3 --overwrite

Varsayılan çıktı ``data/6_synthetic_tr_en_v4`` klasörüdür (düzenleme izli dijital mod);
``src/dataset.py`` bu klasörü ``5_synthetic_tr_en_v3`` ve ``4_synthetic_tr_en``'den önce arar. Hedef klasörde ``metadata.csv`` varsa üretim ``--overwrite``
verilmeden başlamaz (önceki veri setinin yanlışlıkla ezilmesini önler).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.ocr_checker import complete_tckn, compute_iban_check_digits

Box = tuple[int, int, int, int]

# ---------------------------------------------------------------------------
# Font yönetimi
# ---------------------------------------------------------------------------

#: (normal, kalın) font çiftleri. Türkçe karakterleri (ğ, ş, ı, İ) destekleyenler seçildi.
FONT_CANDIDATES: list[tuple[str, str]] = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf"),
    ("C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/verdanab.ttf"),
    ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf"),
    ("C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"),
    ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]


def available_fonts() -> list[tuple[str, str]]:
    """Sistemde bulunan font çiftlerini döndürür (kalın yoksa normal kullanılır)."""
    found = []
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists():
            found.append((regular, bold if Path(bold).exists() else regular))
    return found


def load_font(path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """TTF fontu yükler; yoksa Pillow'un ölçeklenebilir varsayılan fontuna düşer."""
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------------------
# Rastgele içerik üreticileri
# ---------------------------------------------------------------------------

TR_FIRST = ["Ayşe", "Mehmet", "Zeynep", "Mustafa", "Elif", "Ahmet", "Fatma", "Emre", "Şule", "İbrahim",
            "Gülşen", "Çağrı", "Özge", "Barış", "Ece", "Oğuz", "Büşra", "Kerem", "Irmak", "Doğan"]
TR_LAST = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Öztürk", "Aydın", "Özdemir", "Arslan",
           "Doğan", "Kılıç", "Aslan", "Çetin", "Koç", "Kurt", "Özkan", "Şimşek", "Güneş", "Erdoğan"]
EN_FIRST = ["James", "Mary", "Robert", "Linda", "Michael", "Sarah", "David", "Emma", "Daniel", "Olivia",
            "Thomas", "Sophie", "George", "Hannah", "Henry", "Lucy"]
EN_LAST = ["Smith", "Johnson", "Brown", "Taylor", "Wilson", "Davies", "Evans", "Clarke", "Walker", "Wright",
           "Hughes", "Green", "Baker", "Turner", "Hill", "Moore"]
TR_BANKS = ["Anadolu Yıldız Bankası A.Ş.", "Marmara Katılım Bankası", "Ege Ticaret Bankası A.Ş.",
            "Karadeniz Dijital Bank", "Boğaziçi Yatırım Bankası"]
EN_BANKS = ["Northbridge Savings Bank", "Harbor & Crown Bank plc", "Silverline Commercial Bank",
            "Westfield Union Bank", "Bluegate Finance Ltd."]
TR_COMPANIES = ["Güneş Bilişim Ltd. Şti.", "Çınar Gıda San. ve Tic. A.Ş.", "Öz Yapı İnşaat Ltd. Şti.",
                "Akdeniz Kırtasiye", "Işık Elektronik Tic. Ltd. Şti.", "Şimşek Lojistik A.Ş."]
EN_COMPANIES = ["Brightline Software Ltd.", "Oakridge Supplies Inc.", "Northwind Trading Co.",
                "Redstone Electronics LLC", "Greenleaf Catering Ltd.", "Summit Logistics GmbH"]
TR_ITEMS = ["Danışmanlık Hizmeti", "Dizüstü Bilgisayar", "Ofis Kâğıdı A4", "Yazıcı Toneri", "Kablosuz Mouse",
            "Yazılım Lisansı", "Teknik Servis Ücreti", "Çay (1 kg)", "Zeytinyağı 5L", "Şarj Kablosu"]
EN_ITEMS = ["Consulting Service", "Laptop Computer", "A4 Copy Paper", "Printer Toner", "Wireless Mouse",
            "Software License", "Maintenance Fee", "Coffee Beans 1kg", "Olive Oil 5L", "USB-C Cable"]
TR_DESCRIPTIONS = ["Kira ödemesi", "Fatura ödemesi", "Borç iadesi", "Aidat", "Hizmet bedeli", "Maaş ödemesi"]
EN_DESCRIPTIONS = ["Rent payment", "Invoice payment", "Loan repayment", "Membership fee", "Service fee", "Salary"]


def random_tckn(rng: random.Random) -> str:
    """Algoritmik olarak geçerli rastgele TCKN üretir."""
    first_nine = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(8))
    return complete_tckn(first_nine)


def random_iban(rng: random.Random, country: str) -> str:
    """Mod-97 açısından geçerli IBAN üretir ve 4'erli gruplar hâlinde biçimlendirir.

    TR: 5 hane banka kodu + 1 rezerv (0) + 16 hane hesap = 22 haneli BBAN.
    DE: 18 haneli BBAN. GB: 4 harf banka kodu + 14 hane.
    """
    digits = lambda n: "".join(str(rng.randint(0, 9)) for _ in range(n))  # noqa: E731
    if country == "TR":
        bban = digits(5) + "0" + digits(16)
    elif country == "GB":
        bban = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)) + digits(14)
    else:
        bban = digits(18)
    iban = country + compute_iban_check_digits(country, bban) + bban
    return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))


def format_amount(value: float, lang: str, currency: str) -> str:
    """TR: ``1.234,56 TL`` — EN: ``1,234.56 USD``."""
    text = f"{value:,.2f}"
    if lang == "tr":
        text = text.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{text} {currency}"


def random_amount(rng: random.Random, low: float = 50, high: float = 25000) -> float:
    """Log-uniform dağılımlı tutar (küçük tutarlar daha sık)."""
    return round(float(np.exp(rng.uniform(np.log(low), np.log(high)))), 2)


# ---------------------------------------------------------------------------
# Belge çizimi
# ---------------------------------------------------------------------------

@dataclass
class TextField:
    """Belgede düzenlenebilir bir değer alanı."""

    text: str
    origin: tuple[int, int]  # sol-üst (anchor="la")
    font_path: Optional[str]
    size: int
    color: tuple[int, int, int]
    kind: str  # amount / iban / tckn / name / date / text

    def font(self) -> ImageFont.FreeTypeFont:
        return load_font(self.font_path, self.size)

    def box(self) -> Box:
        """Satır yüksekliğini (ascent+descent) kapsayan değer kutusu."""
        font = self.font()
        ascent, descent = font.getmetrics()
        x, y = self.origin
        return x, y, x + int(np.ceil(font.getlength(self.text))), y + ascent + descent

    def char_box(self, index: int) -> Box:
        """``index``'inci karakterin hücresi: x sınırları önek genişliklerinden hesaplanır."""
        font = self.font()
        x0 = self.origin[0] + int(font.getlength(self.text[:index]))
        x1 = self.origin[0] + int(np.ceil(font.getlength(self.text[:index + 1])))
        _, y0, _, y1 = self.box()
        return x0, y0, x1, y1


@dataclass
class Document:
    """Çizilmiş belge ve düzenlenebilir alanları."""

    image: Image.Image
    fields: dict[str, TextField]
    lang: str
    doc_type: str
    currency: str
    values: dict[str, float] = field(default_factory=dict)


class Canvas:
    """Satır imleçli basit bir belge çizici; değer alanlarının konumlarını kaydeder."""

    def __init__(self, width: int, height: int, font_pair: tuple[Optional[str], Optional[str]],
                 size: int, rng: random.Random) -> None:
        self.image = Image.new("RGB", (width, height), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.regular, self.bold = font_pair
        self.size = size
        self.margin = int(width * 0.07)
        self.width = width
        self.y = self.margin
        self.ink = tuple(rng.randint(0, 40) for _ in range(3))
        self.fields: dict[str, TextField] = {}

    def text(self, x: int, text: str, bold: bool = False, size: Optional[int] = None,
             name: Optional[str] = None, kind: str = "text", align: str = "left") -> None:
        """Mevcut satıra metin yazar; ``name`` verilirse alan olarak kaydeder."""
        size = size or self.size
        path = self.bold if bold else self.regular
        font = load_font(path, size)
        if align == "right":
            x = int(x - font.getlength(text))
        self.draw.text((x, self.y), text, font=font, fill=self.ink, anchor="la")
        if name:
            self.fields[name] = TextField(text, (x, self.y), path, size, self.ink, kind)

    def kv(self, label: str, value: str, name: Optional[str] = None, kind: str = "text",
           value_right: bool = False) -> None:
        """"Etiket: değer" satırı. Değer ya sabit bir sütunda ya da sağa yaslı yazılır."""
        self.text(self.margin, label, bold=True)
        if value_right:
            self.text(self.width - self.margin, value, name=name, kind=kind, align="right")
        else:
            self.text(int(self.width * 0.40), value, name=name, kind=kind)
        self.newline()

    def newline(self, factor: float = 1.55) -> None:
        self.y += int(self.size * factor)

    def finish(self) -> Image.Image:
        """Son satırın altındaki boş alanı kırpar (belge yüksekliği içeriğe göre değişir)."""
        return self.image.crop((0, 0, self.width, min(self.image.height, self.y + self.margin)))

    def rule(self) -> None:
        """Yatay ayırıcı çizgi."""
        self.y += self.size // 3
        self.draw.line((self.margin, self.y, self.width - self.margin, self.y), fill=self.ink, width=1)
        self.y += self.size // 2


def _pick(rng: random.Random, lang: str, tr: list[str], en: list[str]) -> str:
    return rng.choice(tr if lang == "tr" else en)


def _person(rng: random.Random, lang: str) -> str:
    return f"{_pick(rng, lang, TR_FIRST, EN_FIRST)} {_pick(rng, lang, TR_LAST, EN_LAST)}"


def _date(rng: random.Random, lang: str) -> str:
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2021, 2026)
    hh, mm = rng.randint(8, 22), rng.randint(0, 59)
    return f"{d:02d}.{m:02d}.{y} {hh:02d}:{mm:02d}" if lang == "tr" else f"{m:02d}/{d:02d}/{y} {hh:02d}:{mm:02d}"


def render_bank_receipt(rng: random.Random, lang: str, fonts: list[tuple[str, str]]) -> Document:
    """Banka dekontu (havale/EFT) çizer."""
    currency = "TL" if lang == "tr" else rng.choice(["USD", "EUR"])
    country = "TR" if lang == "tr" else rng.choice(["GB", "DE"])
    c = Canvas(1000, 1250, rng.choice(fonts) if fonts else (None, None), rng.randint(22, 26), rng)
    tr = lang == "tr"
    c.text(c.margin, _pick(rng, lang, TR_BANKS, EN_BANKS), bold=True, size=c.size + 10)
    c.newline(2.2)
    c.text(c.margin, "HAVALE / EFT DEKONTU" if tr else "FUNDS TRANSFER RECEIPT", bold=True, size=c.size + 4)
    c.newline(1.8)
    c.rule()
    amount, fee = random_amount(rng), rng.choice([0.0, 2.5, 5.5, 8.75, 12.0])
    c.kv("İşlem Tarihi:" if tr else "Date:", _date(rng, lang), "date", "date")
    c.kv("Referans No:" if tr else "Reference:", str(rng.randint(10**9, 10**10 - 1)), "ref", "text")
    c.rule()
    c.kv("Gönderen:" if tr else "Sender:", _person(rng, lang), "sender", "name")
    if tr:
        c.kv("Gönderen TCKN:", random_tckn(rng), "tckn", "tckn")
    c.kv("Gönderen IBAN:" if tr else "Sender IBAN:", random_iban(rng, country), "iban_sender", "iban")
    c.kv("Alıcı:" if tr else "Beneficiary:", _person(rng, lang), "receiver", "name")
    c.kv("Alıcı IBAN:" if tr else "Beneficiary IBAN:", random_iban(rng, country), "iban", "iban")
    c.kv("Açıklama:" if tr else "Description:", _pick(rng, lang, TR_DESCRIPTIONS, EN_DESCRIPTIONS), "desc", "text")
    c.rule()
    c.kv("İşlem Tutarı:" if tr else "Amount:", format_amount(amount, lang, currency), "amount", "amount", True)
    c.kv("Masraf:" if tr else "Fee:", format_amount(fee, lang, currency), "fee", "amount", True)
    c.kv("Hesaptan Çekilen:" if tr else "Debited:", format_amount(amount + fee, lang, currency),
         "debited", "amount", True)
    c.rule()
    c.text(c.margin, "Bu dekont elektronik ortamda üretilmiştir." if tr else
           "This receipt was generated electronically.", size=c.size - 6)
    return Document(c.finish(), c.fields, lang, "dekont", currency, {"amount": amount, "fee": fee})


def _line_items(rng: random.Random, lang: str) -> list[tuple[str, int, float]]:
    pool = TR_ITEMS if lang == "tr" else EN_ITEMS
    return [(name, rng.randint(1, 5), random_amount(rng, 5, 3000)) for name in rng.sample(pool, rng.randint(2, 5))]


def render_invoice(rng: random.Random, lang: str, fonts: list[tuple[str, str]]) -> Document:
    """Fatura çizer: kalemler, Ara Toplam (Matrah), KDV ve Genel Toplam."""
    currency = "TL" if lang == "tr" else rng.choice(["USD", "EUR"])
    country = "TR" if lang == "tr" else rng.choice(["GB", "DE"])
    tr = lang == "tr"
    c = Canvas(1050, 1450, rng.choice(fonts) if fonts else (None, None), rng.randint(20, 24), rng)
    c.text(c.margin, _pick(rng, lang, TR_COMPANIES, EN_COMPANIES), bold=True, size=c.size + 8)
    c.text(c.width - c.margin, "E-ARŞİV FATURA" if tr else "INVOICE", bold=True, size=c.size + 8, align="right")
    c.newline(2.4)
    c.kv("Fatura No:" if tr else "Invoice No:", f"{rng.choice('ABCDEFG')}{rng.randint(2021, 2026)}{rng.randint(10**8, 10**9 - 1)}",
         "invoice_no", "text")
    c.kv("Tarih:" if tr else "Date:", _date(rng, lang).split()[0], "date", "date")
    c.kv("Sayın:" if tr else "Bill To:", _person(rng, lang), "customer", "name")
    if tr:
        c.kv("Müşteri TCKN:", random_tckn(rng), "tckn", "tckn")
    c.rule()
    col_qty, col_unit, col_total = int(c.width * 0.55), int(c.width * 0.75), c.width - c.margin
    c.text(c.margin, "Açıklama" if tr else "Description", bold=True)
    c.text(col_qty, "Miktar" if tr else "Qty", bold=True, align="right")
    c.text(col_unit, "Birim Fiyat" if tr else "Unit Price", bold=True, align="right")
    c.text(col_total, "Tutar" if tr else "Amount", bold=True, align="right")
    c.newline()
    subtotal = 0.0
    for i, (name, qty, unit) in enumerate(_line_items(rng, lang)):
        line_total = round(qty * unit, 2)
        subtotal += line_total
        c.text(c.margin, name)
        c.text(col_qty, str(qty), align="right", name=f"qty_{i}", kind="amount")
        c.text(col_unit, format_amount(unit, lang, currency), align="right", name=f"unit_{i}", kind="amount")
        c.text(col_total, format_amount(line_total, lang, currency), align="right", name=f"line_{i}", kind="amount")
        c.newline()
    c.rule()
    subtotal = round(subtotal, 2)
    rate = rng.choice([0.01, 0.10, 0.20]) if tr else rng.choice([0.05, 0.10, 0.20])
    vat = round(subtotal * rate, 2)
    total = round(subtotal + vat, 2)
    c.kv("Ara Toplam (Matrah):" if tr else "Subtotal:", format_amount(subtotal, lang, currency), "subtotal", "amount", True)
    c.kv(f"KDV (%{rate * 100:.0f}):" if tr else f"VAT ({rate * 100:.0f}%):", format_amount(vat, lang, currency),
         "vat", "amount", True)
    c.kv("Genel Toplam:" if tr else "Grand Total:", format_amount(total, lang, currency), "total", "amount", True)
    c.rule()
    c.kv("IBAN:", random_iban(rng, country), "iban", "iban")
    return Document(c.finish(), c.fields, lang, "fatura", currency,
                    {"subtotal": subtotal, "vat": vat, "total": total, "rate": rate})


def render_retail_receipt(rng: random.Random, lang: str, fonts: list[tuple[str, str]]) -> Document:
    """Dar formatlı perakende fişi (yazar kasa fişi) çizer."""
    currency = "TL" if lang == "tr" else rng.choice(["USD", "EUR"])
    tr = lang == "tr"
    c = Canvas(640, 1150, rng.choice(fonts) if fonts else (None, None), rng.randint(20, 23), rng)
    c.text(c.margin, _pick(rng, lang, TR_COMPANIES, EN_COMPANIES)[:30], bold=True)
    c.newline(1.8)
    c.text(c.margin, _date(rng, lang), name="date", kind="date")
    c.text(c.width - c.margin, f"{'FİŞ NO' if tr else 'NO'}: {rng.randint(1, 9999):04d}", align="right")
    c.newline(1.8)
    c.rule()
    subtotal = 0.0
    for i, (name, qty, unit) in enumerate(_line_items(rng, lang)):
        line_total = round(qty * unit, 2)
        subtotal += line_total
        c.text(c.margin, f"{qty} x {name[:18]}")
        c.text(c.width - c.margin, format_amount(line_total, lang, currency), align="right",
               name=f"line_{i}", kind="amount")
        c.newline()
    c.rule()
    subtotal = round(subtotal, 2)
    rate = rng.choice([0.01, 0.10, 0.20]) if tr else rng.choice([0.05, 0.10, 0.20])
    vat = round(subtotal * rate, 2)
    total = round(subtotal + vat, 2)
    c.kv("ARA TOPLAM" if tr else "SUBTOTAL", format_amount(subtotal, lang, currency), "subtotal", "amount", True)
    c.kv(f"KDV %{rate * 100:.0f}" if tr else f"VAT {rate * 100:.0f}%", format_amount(vat, lang, currency),
         "vat", "amount", True)
    c.kv("GENEL TOPLAM" if tr else "GRAND TOTAL", format_amount(total, lang, currency), "total", "amount", True)
    c.rule()
    c.text(c.margin, "TEŞEKKÜR EDERİZ" if tr else "THANK YOU", bold=True)
    return Document(c.finish(), c.fields, lang, "fis", currency,
                    {"subtotal": subtotal, "vat": vat, "total": total, "rate": rate})


RENDERERS: dict[str, Callable[[random.Random, str, list[tuple[str, str]]], Document]] = {
    "dekont": render_bank_receipt,
    "fatura": render_invoice,
    "fis": render_retail_receipt,
}


# ---------------------------------------------------------------------------
# Edinim (tarama/kayıt) simülasyonu
# ---------------------------------------------------------------------------

def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Görüntüyü bellekte JPEG'e kodlayıp geri çözer."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def simulate_acquisition(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Kâğıt tonu + sensör gürültüsü + optik bulanıklık + JPEG ile "taranmış belge" üretir."""
    out = image.astype(np.float32)
    base = rng.uniform(232, 252)
    tint = np.array([base, base - rng.uniform(0, 4), base - rng.uniform(0, 10)], dtype=np.float32)
    out = out / 255.0 * tint  # beyaz kâğıdı hafif krem/gri tona çeker
    if rng.random() < 0.7:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 0.9))
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    out += np_rng.normal(0, rng.uniform(1.0, 3.5), out.shape).astype(np.float32)
    return jpeg_roundtrip(np.clip(out, 0, 255).astype(np.uint8), rng.randint(80, 97))


#: Dijital belgelerin kullanıcıya ulaştığı kanallar ve seçilme olasılıkları.
DIGITAL_CHANNELS: dict[str, float] = {
    "png": 0.25,         # PDF → PNG dışa aktarım (kayıpsız)
    "pdf_jpeg": 0.25,    # PDF → yüksek kaliteli JPEG (q 90-98)
    "screenshot": 0.25,  # ekran görüntüsü: ölçek 0.75-1.5 (HiDPI / yakınlaştırma) + PNG ya da JPEG
    "whatsapp": 0.25,    # uzun kenar ≤ 1600 + JPEG q 70-85
}


def sample_digital_channel(rng: random.Random) -> dict[str, object]:
    """Grup için bir iletim kanalı ve parametrelerini seçer (tüm varyantlar aynısını kullanır)."""
    name = rng.choices(list(DIGITAL_CHANNELS), weights=list(DIGITAL_CHANNELS.values()))[0]
    if name == "png":
        return {"name": name, "scale": 1.0, "quality": None}
    if name == "pdf_jpeg":
        return {"name": name, "scale": 1.0, "quality": rng.randint(90, 98)}
    if name == "screenshot":
        return {"name": name, "scale": rng.uniform(0.75, 1.5),
                "quality": rng.choice([None, rng.randint(85, 95)])}
    return {"name": name, "scale": None, "max_side": 1600, "quality": rng.randint(70, 85)}


def apply_digital_channel(image: np.ndarray, mask: np.ndarray, channel: dict[str, object]
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Kanalın geometrik kısmını (ölçekleme) uygular; maske en yakın komşu ile aynı ölçeklenir.

    Sıkıştırma ``_save`` sırasında ``channel['quality']`` ile yapılır (çift JPEG olmaz).
    """
    scale = channel.get("scale")
    if channel.get("max_side"):
        scale = min(1.0, float(channel["max_side"]) / max(image.shape[:2]))
    if scale and abs(scale - 1.0) > 1e-3:
        h, w = image.shape[:2]
        size = (round(w * scale), round(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, size, interpolation=interp)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image, mask


# ---------------------------------------------------------------------------
# Sahtecilik işlemleri
# ---------------------------------------------------------------------------

#: Dijital belgelerde sahte bölgeye uygulanan düzenleme izleri ve seçilme olasılıkları.
EDIT_TRACES: dict[str, float] = {
    "local_jpeg": 0.35,  # bölge başka bir kaynaktan (JPEG) gelmiş: kaydırılmış 8x8 ızgarayla çift sıkıştırma
    "resample": 0.30,    # yapıştırılan metin ölçeklenmiş: farklı kenar yumuşatma (anti-aliasing)
    "noise": 0.20,       # taranmış/fotoğraflanmış kaynaktan yama: düz zeminde sensör gürültüsü
    "color": 0.15,       # yazı ve zemin renginde ±2-5 ton kayması (farklı renk profili / editör)
}


def add_edit_trace(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str]:
    """Dijital belgede sahte bölgeye gerçekçi bir düzenleme izi ekler (yalnızca ``mask > 0`` pikselleri).

    Dijital doğmuş belgede ``rewrite_field`` beyaz zemini beyazla boyayıp aynı fontla yazdığında
    sahte alan gerçek metinle piksel düzeyinde aynıdır: hiçbir görüntü modeli onu ayırt edemez ve
    model "tutar/IBAN alanı = sahte" gibi bir konum ezberine yönelir (orijinallerde yanlış alarm).
    Gerçek düzenleme araçları (görüntü editörüne yapıştırma, başka belgeden kopyalama, ekran
    görüntüsü birleştirme) ise burada benzetilen yerel izlerden en az birini bırakır.
    """
    name = rng.choices(list(EDIT_TRACES), weights=list(EDIT_TRACES.values()))[0]
    if name == "local_jpeg":
        dy, dx = rng.randint(1, 7), rng.randint(1, 7)
        padded = cv2.copyMakeBorder(image, dy, 0, dx, 0, cv2.BORDER_REPLICATE)
        traced = jpeg_roundtrip(padded, rng.randint(55, 85))[dy:, dx:]
    elif name == "resample":
        h, w = image.shape[:2]
        scale = rng.uniform(0.55, 0.85)
        small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        traced = cv2.resize(small, (w, h), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    elif name == "noise":
        np_rng = np.random.default_rng(rng.randint(0, 2**31))
        traced = image.astype(np.float32) + np_rng.normal(0, rng.uniform(1.5, 3.5), image.shape)
    else:
        shift = np.array([rng.choice([-1, 1]) * rng.randint(2, 5) for _ in range(3)], dtype=np.float32)
        traced = image.astype(np.float32) + shift
    traced = np.clip(traced, 0, 255).astype(np.uint8)
    out = image.copy()
    region = mask > 0
    out[region] = traced[region]
    return out, name


def _expand(box: Box, pad: int, shape: tuple[int, ...]) -> Box:
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(shape[1], x1 + pad), min(shape[0], y1 + pad)


def _background_color(image: np.ndarray, box: Box) -> tuple[int, int, int]:
    """Kutunun 3 piksellik dış çerçevesindeki piksellerin medyanı = yerel arka plan rengi.

    Medyan, çerçeveye denk gelen metin piksellerine karşı dayanıklıdır.
    """
    x0, y0, x1, y1 = _expand(box, 3, image.shape)
    ring = np.concatenate([image[y0:y0 + 3, x0:x1].reshape(-1, 3), image[y1 - 3:y1, x0:x1].reshape(-1, 3),
                           image[y0:y1, x0:x0 + 3].reshape(-1, 3), image[y0:y1, x1 - 3:x1].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(ring, axis=0))


def _draw_text(image: np.ndarray, origin: tuple[int, int], text: str, font: ImageFont.FreeTypeFont,
               color: tuple[int, int, int]) -> np.ndarray:
    pil = Image.fromarray(image)
    ImageDraw.Draw(pil).text(origin, text, font=font, fill=color, anchor="la")
    return np.asarray(pil).copy()


def _mark(mask: np.ndarray, box: Box) -> None:
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 255


def rewrite_field(image: np.ndarray, mask: np.ndarray, fld: TextField, new_text: str,
                  rng: random.Random, fonts: list[tuple[str, str]], generative: bool = False) -> np.ndarray:
    """
    Bir alanı silip yeniden yazar (Photoshop "metin değiştirme" veya YZ inpainting benzetimi).

    * Klasik: alan düz arka plan rengiyle boyanır → bölgede tarama gürültüsü
      kaybolur (SRM akışının yakaladığı temel iz). Yeni metin %30 olasılıkla farklı bir
      fontla, ±1 piksel boyut ve ±1 piksel konum kaymasıyla yazılır.
    * Üretken (``generative=True``): bölge ``cv2.inpaint`` ile doldurulur, metin yazılır
      ve bölge Gauss ile yumuşatılır; difüzyon modellerinin düşük gürültülü, hafif
      bulanık dolgu karakteristiği taklit edilir.
    """
    old_box = fld.box()
    region = _expand(old_box, 2, image.shape)
    x0, y0, x1, y1 = region
    if generative:
        inpaint_mask = np.zeros(image.shape[:2], np.uint8)
        inpaint_mask[y0:y1, x0:x1] = 255
        image = cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_TELEA)
    else:
        image = image.copy()
        image[y0:y1, x0:x1] = _background_color(image, old_box)

    font_path = fld.font_path
    if not generative and fonts and rng.random() < 0.3:
        font_path = rng.choice(fonts)[0]
    size = fld.size + rng.choice([-1, 0, 0, 1])
    font = load_font(font_path, size)
    right_aligned = fld.kind == "amount"
    width = int(np.ceil(font.getlength(new_text)))
    ox = (old_box[2] - width if right_aligned else fld.origin[0]) + rng.choice([-1, 0, 0, 1])
    oy = fld.origin[1] + rng.choice([-1, 0, 0, 1])
    image = _draw_text(image, (ox, oy), new_text, font, fld.color)

    ascent, descent = font.getmetrics()
    new_box = _expand((ox, oy, ox + width, oy + ascent + descent), 2, image.shape)
    full = (min(x0, new_box[0]), min(y0, new_box[1]), max(x1, new_box[2]), max(y1, new_box[3]))
    if generative:
        fx0, fy0, fx1, fy1 = full
        image[fy0:fy1, fx0:fx1] = cv2.GaussianBlur(image[fy0:fy1, fx0:fx1], (3, 3), 0.8)
    _mark(mask, full)
    fld.text, fld.origin, fld.font_path, fld.size = new_text, (ox, oy), font_path, size
    return image


def _digit_positions(text: str) -> list[int]:
    return [i for i, ch in enumerate(text) if ch.isdigit()]


def _perturb_amount(value: float, rng: random.Random) -> float:
    """Tutarı inandırıcı biçimde değiştirir (ör. 1.250,00 → 7.250,00 veya ×1.5-10)."""
    if rng.random() < 0.5:
        return round(value * rng.uniform(1.5, 10.0), 2)
    return round(value + rng.choice([1, 2, 5, 9]) * 10 ** rng.randint(1, 3), 2)


def op_copy_move(doc: Document, image: np.ndarray, mask: np.ndarray, rng: random.Random,
                 fonts: list[tuple[str, str]]) -> Optional[np.ndarray]:
    """Belge içi kopyala-taşı: bir tutar alanındaki rakam glifi başka bir tutarın rakamı üzerine yapıştırılır.

    Kaynak ve hedef aynı font/boyutta olmalıdır (aynı belgede çizilen tutarlar). Kopyalanan
    yama hedef hücre genişliğine ölçeklenir. Maske yalnızca hedef hücredir.
    """
    amounts = [f for f in doc.fields.values() if f.kind == "amount" and _digit_positions(f.text)]
    if len(amounts) < 2:
        return None
    src, dst = rng.sample(amounts, 2)
    if (src.font_path, src.size) != (dst.font_path, dst.size):
        return None
    si = rng.choice(_digit_positions(src.text))
    candidates = [i for i in _digit_positions(dst.text) if dst.text[i] != src.text[si]]
    if not candidates:
        return None
    di = rng.choice(candidates)
    sx0, sy0, sx1, sy1 = src.char_box(si)
    dx0, dy0, dx1, dy1 = dst.char_box(di)
    patch = image[sy0:sy1, sx0:sx1]
    if patch.size == 0 or dx1 <= dx0 or dy1 <= dy0:
        return None
    patch = cv2.resize(patch, (dx1 - dx0, dy1 - dy0), interpolation=cv2.INTER_LINEAR)
    image = image.copy()
    image[dy0:dy1, dx0:dx1] = patch
    _mark(mask, (dx0, dy0, dx1, dy1))
    dst.text = dst.text[:di] + src.text[si] + dst.text[di + 1:]
    return image


def op_text_rewrite(doc: Document, image: np.ndarray, mask: np.ndarray, rng: random.Random,
                    fonts: list[tuple[str, str]], generative: bool = False) -> Optional[np.ndarray]:
    """İsim, tarih veya bir tutar alanını tamamen yeniden yazar."""
    names = [k for k, f in doc.fields.items() if f.kind in ("name", "date", "amount")]
    if not names:
        return None
    key = rng.choice(names)
    fld = doc.fields[key]
    if fld.kind == "name":
        new = _person(rng, doc.lang)
    elif fld.kind == "date":
        new = _date(rng, doc.lang)[: len(fld.text)]
    else:
        new = format_amount(random_amount(rng), doc.lang, doc.currency)
    if new == fld.text:
        return None
    return rewrite_field(image, mask, fld, new, rng, fonts, generative)


def op_inpaint_sim(doc: Document, image: np.ndarray, mask: np.ndarray, rng: random.Random,
                   fonts: list[tuple[str, str]]) -> Optional[np.ndarray]:
    """Üretken YZ inpainting benzetimi (bkz. ``rewrite_field(generative=True)``)."""
    return op_text_rewrite(doc, image, mask, rng, fonts, generative=True)


def op_amount_tamper(doc: Document, image: np.ndarray, mask: np.ndarray, rng: random.Random,
                     fonts: list[tuple[str, str]]) -> Optional[np.ndarray]:
    """Yalnızca toplamı (fatura/fiş) veya işlem tutarını (dekont) değiştirir → içerik tutarsızlığı."""
    key = "total" if "total" in doc.fields else "amount"
    base = doc.values.get(key)
    if key not in doc.fields or base is None:
        return None
    new_text = format_amount(_perturb_amount(base, rng), doc.lang, doc.currency)
    return rewrite_field(image, mask, doc.fields[key], new_text, rng, fonts)


def op_amount_consistent(doc: Document, image: np.ndarray, mask: np.ndarray, rng: random.Random,
                         fonts: list[tuple[str, str]]) -> Optional[np.ndarray]:
    """Matrah, KDV ve Toplam'ı matematiksel tutarlılığı koruyarak birlikte değiştirir.

    Bu "akıllı sahtecilik" OCR kurallarını geçer; yalnızca piksel izleriyle yakalanabilir.
    """
    if not {"subtotal", "vat", "total"} <= doc.fields.keys():
        return None
    subtotal = _perturb_amount(doc.values["subtotal"], rng)
    vat = round(subtotal * doc.values["rate"], 2)
    total = round(subtotal + vat, 2)
    for key, value in (("subtotal", subtotal), ("vat", vat), ("total", total)):
        image = rewrite_field(image, mask, doc.fields[key], format_amount(value, doc.lang, doc.currency),
                              rng, fonts)
    return image


def op_iban_tamper(doc: Document, image: np.ndarray, mask: np.ndarray, rng: random.Random,
                   fonts: list[tuple[str, str]]) -> Optional[np.ndarray]:
    """IBAN'ın kontrol haneleri dışındaki 1-2 rakamını karakter düzeyinde değiştirir.

    Mod-97 tek rakam değişikliklerinin tamamını yakaladığı için sonuç her zaman
    geçersiz bir IBAN'dır.
    """
    if "iban" not in doc.fields:
        return None
    fld = doc.fields["iban"]
    positions = [i for i in _digit_positions(fld.text) if i >= 5]  # "TR12 " sonrası
    image = image.copy()
    text = fld.text
    font = fld.font()
    for i in rng.sample(positions, rng.randint(1, 2)):
        x0, y0, x1, y1 = fld.char_box(i)
        new_digit = rng.choice([d for d in "0123456789" if d != text[i]])
        image[y0:y1, x0:x1] = _background_color(image, (x0, y0, x1, y1))
        image = _draw_text(image, (x0, y0), new_digit, font, fld.color)
        _mark(mask, _expand((x0, y0, x1, y1), 1, image.shape))
        text = text[:i] + new_digit + text[i + 1:]
    fld.text = text
    return image


OPERATIONS: dict[str, Callable[..., Optional[np.ndarray]]] = {
    "copy_move": op_copy_move,
    "text_rewrite": op_text_rewrite,
    "amount_tamper": op_amount_tamper,
    "amount_consistent": op_amount_consistent,
    "iban_tamper": op_iban_tamper,
    "inpaint_sim": op_inpaint_sim,
}


# ---------------------------------------------------------------------------
# Ana üretim döngüsü
# ---------------------------------------------------------------------------

def _save(path: Path, image: np.ndarray, quality: Optional[int] = None) -> None:
    """Unicode-güvenli kayıt (cv2.imwrite Windows'ta Türkçe yollarda başarısız olabilir)."""
    ext = path.suffix
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else []
    data = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, data, params)
    assert ok, f"Kodlama başarısız: {path}"
    buf.tofile(str(path))


def generate_group(group_id: str, rng: random.Random, fonts: list[tuple[str, str]], out_dir: Path,
                   n_variants: int, tr_ratio: float, digital_ratio: float = 0.0) -> list[dict[str, object]]:
    """Bir şablon örneği için 1 orijinal + ``n_variants`` sahte görüntü üretir ve diske yazar.

    ``digital_ratio`` olasılıkla grup dijital doğmuş belgedir: edinim simülasyonu atlanır ve
    sonda ``sample_digital_channel`` ile seçilen iletim kanalı uygulanır.
    """
    lang = "tr" if rng.random() < tr_ratio else "en"
    doc_type = rng.choice(list(RENDERERS))
    seed = rng.randint(0, 2**31)
    digital = rng.random() < digital_ratio
    rendered = np.asarray(RENDERERS[doc_type](random.Random(seed), lang, fonts).image.convert("RGB"))
    acquired = rendered.copy() if digital else simulate_acquisition(rendered, rng)

    rows = []
    variants: list[tuple[str, np.ndarray, np.ndarray, str]] = [
        ("orig", acquired, np.zeros(acquired.shape[:2], np.uint8), "")]
    ops = list(OPERATIONS)
    attempts = 0
    while len(variants) <= n_variants and attempts < n_variants * 6:
        attempts += 1
        op_name = rng.choice(ops)
        # Her varyant temiz belge kopyası üzerinde çalışır (alan konumları yeniden üretilir).
        doc = RENDERERS[doc_type](random.Random(seed), lang, fonts)
        mask = np.zeros(acquired.shape[:2], np.uint8)
        tampered = OPERATIONS[op_name](doc, acquired, mask, rng, fonts)
        if tampered is None or mask.max() == 0:
            continue
        trace = ""
        if digital:
            # Taranmış belgede düz boyama tarama gürültüsünü siler (iz zaten var); dijitalde yoksa eklenir.
            tampered, trace = add_edit_trace(tampered, mask, rng)
        variants.append((op_name, tampered, mask, trace))

    # Grup içindeki tüm varyantlar aynı son kanaldan / kaliteyle kaydedilir.
    if digital:
        channel = sample_digital_channel(rng)
        final_quality = channel["quality"]
    else:
        channel = {"name": "scan"}
        final_quality = rng.randint(75, 95)
    ext = ".jpg" if final_quality else ".png"
    for idx, (op_name, image, mask, trace) in enumerate(variants):
        if digital:
            image, mask = apply_digital_channel(image, mask, channel)
        stem = f"{group_id}__{idx:02d}_{op_name}"
        _save(out_dir / "images" / f"{stem}{ext}", image, final_quality)
        _save(out_dir / "masks" / f"{stem}.png", mask)
        rows.append({"image": f"images/{stem}{ext}", "mask": f"masks/{stem}.png", "group_id": group_id,
                     "label": int(op_name != "orig"), "tamper_type": op_name, "lang": lang, "doc_type": doc_type,
                     "acquisition": "digital" if digital else "scan", "channel": channel["name"],
                     "edit_trace": trace})
    return rows


def generate_dataset(out_dir: str | Path, num_docs: int, n_variants: int = 3, seed: int = 42,
                     tr_ratio: float = 0.6, digital_ratio: float = 0.0) -> Path:
    """Sentetik veri kümesini üretir ve ``metadata.csv`` yolunu döndürür."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    fonts = available_fonts()
    if not fonts:
        print("UYARI: TTF font bulunamadı, Pillow varsayılan fontu kullanılacak (çeşitlilik düşük).")
    rng = random.Random(seed)
    meta_path = out_dir / "metadata.csv"
    start = time.time()
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "mask", "group_id", "label", "tamper_type", "lang", "doc_type",
                                                "acquisition", "channel", "edit_trace"])
        writer.writeheader()
        for i in range(num_docs):
            for row in generate_group(f"syn{i:06d}", rng, fonts, out_dir, n_variants, tr_ratio, digital_ratio):
                writer.writerow(row)
            if (i + 1) % 100 == 0 or i + 1 == num_docs:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (num_docs - i - 1) / 60
                print(f"  {i + 1}/{num_docs} belge üretildi ({elapsed / 60:.1f} dk, kalan ~{eta:.1f} dk)", flush=True)
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Türkçe/İngilizce sentetik sahte belge üretici")
    parser.add_argument("--out", default="data/6_synthetic_tr_en_v4")
    parser.add_argument("--num-docs", type=int, default=2000,
                        help="Şablon örneği (grup) sayısı; her grup 1 orijinal + --variants sahte görüntüdür "
                             "(varsayılan 2000 × 4 ≈ 8000 görüntü)")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Hedef toplam görüntü sayısı (ör. 5000-10000); verilirse --num-docs bundan hesaplanır")
    parser.add_argument("--variants", type=int, default=3, help="Grup başına sahte varyant sayısı")
    parser.add_argument("--tr-ratio", type=float, default=0.6, help="Türkçe belge oranı")
    parser.add_argument("--digital-ratio", type=float, default=0.5,
                        help="Dijital doğmuş (PDF/ekran görüntüsü) belge oranı: 0 = hepsi taranmış, "
                             "1 = hepsi dijital (varsayılan 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Hedef klasördeki mevcut veri setinin üzerine yaz")
    args = parser.parse_args()
    if args.num_images:
        args.num_docs = math.ceil(args.num_images / (args.variants + 1))
    if (Path(args.out) / "metadata.csv").exists() and not args.overwrite:
        parser.error(f"{args.out} zaten bir veri seti içeriyor; üzerine yazmak için --overwrite verin "
                     "ya da farklı bir --out seçin.")
    print(f"{args.num_docs} belge × {args.variants + 1} ≈ {args.num_docs * (args.variants + 1)} görüntü → {args.out}")
    path = generate_dataset(args.out, args.num_docs, args.variants, args.seed, args.tr_ratio, args.digital_ratio)
    print(f"Tamamlandı: {path}")


if __name__ == "__main__":
    main()
