"""
PDF yapı analizi: belgeyi görüntüye çevirmeden önce dosyanın kendisindeki düzenleme izleri.

PDF rasterleştirildiğinde (``app.py`` → 200 DPI) en güçlü kanıtlar kaybolur: düzenleme aracının
imzası, sonradan eklenen kayıtlar, tek alanda kullanılan farklı font, beyaz kutuyla gizlenmiş
metin. Banka ve GİB sistemlerinin ürettiği PDF'ler tek seferde, sabit bir yazılımla üretilir;
bu yüzden aşağıdaki izler sistem çıktısında beklenmez.

Kurallar (``PDF_RULE_WEIGHTS``: başarısız kuralın riske katkısı, noisy-OR ile birleşir)
---------------------------------------------------------------------------------------
* ``pdf_signature``   : Dijital imza varsa imzanın kapsadığı bayt aralığından **sonra** dosyaya
                        ekleme yapılmış mı? (İmzadan sonra değişiklik.) Kriptografik doğrulama yapılmaz.
* ``pdf_incremental`` : Dosya ilk kayıttan sonra artımlı olarak yeniden kaydedilmiş mi?
                        (Birden fazla ``%%EOF``; doğrusallaştırılmış dosyanın ikinci işareti ve
                        imzalama kayıtları hariç.)
* ``pdf_editor``      : Producer/Creator/XMP'de PDF düzenleme aracı (Sejda, iLovePDF, PDF-XChange,
                        Foxit, Photoshop ...) var mı? Ofis/yazdırma araçları yalnızca uyarıdır.
* ``pdf_dates``       : Değiştirilme tarihi oluşturulma tarihinden belirgin biçimde sonra mı?
* ``pdf_xmp_history`` : XMP geçmişinde "saved/modified" olayı var mı?
* ``pdf_hidden_text`` : Bir metin, sonradan çizilmiş açık renkli kutu veya görüntü altında gizlenmiş mi?
* ``pdf_overlap``     : Farklı iki metin aynı konuma üst üste yazılmış mı? (Eski değer + yeni değer.)
* ``pdf_font``        : Belgede başka hiçbir yerde kullanılmayan bir font yalnızca bir/birkaç alanda
                        (özellikle rakamlarda) mı geçiyor? Alt kümelenmiş fontların arasında tek bir
                        alt kümelenmemiş font var mı? (Editörlerin eklediği metnin tipik izi.)

Yalnızca bilgi/uyarı: taranmış (görüntü tabanlı) PDF, görünmez metin, beyaz kutu üzerine yazılmış metin.

Kutular (``bbox``), ``check_pdf(..., dpi)`` ile verilen çözünürlükte **piksel** koordinatındadır;
böylece ``app.py``'nin rasterleştirdiği sayfa görüntüsünün üzerine doğrudan çizilebilir.

Örnek:
    from src.pdf_checker import check_pdf
    report = check_pdf(open("dekont.pdf", "rb").read())
    print(report.risk, [f.message for f in report.findings])
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.ocr_checker import FAIL, PASS, WARN, Finding

PDF_RULE_WEIGHTS: dict[str, float] = {
    "pdf_signature": 0.90,
    "pdf_hidden_text": 0.75,
    "pdf_overlap": 0.70,
    "pdf_incremental": 0.60,
    "pdf_editor": 0.50,
    "pdf_font": 0.50,
    "pdf_xmp_history": 0.40,
    "pdf_dates": 0.35,
}

#: Bu araçlar mevcut bir PDF'i düzenlemek için kullanılır; banka/GİB çıktısında beklenmez.
EDITOR_KEYWORDS = (
    "photoshop", "illustrator", "gimp", "inkscape", "canva", "pdf-xchange editor", "pdf-xchange pro",
    "foxit phantompdf", "foxit pdf editor", "foxit editor", "sejda", "ilovepdf", "smallpdf", "pdfescape",
    "pdffiller", "pdf filler", "nitro pro", "nitro pdf", "pdfelement", "wondershare", "docfly", "pdf candy",
    "pdfcandy", "soda pdf", "able2extract", "master pdf editor", "pdf studio", "libreoffice draw",
    "acrobat pro", "acrobat dc", "adobe acrobat 2", "touchup",
)
#: Ofis yazılımı / sanal yazıcı: belge yeniden oluşturulmuş veya yeniden yazdırılmış olabilir (uyarı).
REPRINT_KEYWORDS = (
    "microsoft word", "microsoft: print to pdf", "microsoft print to pdf", "libreoffice", "openoffice",
    "wps", "google docs", "pages", "quartz pdfcontext", "pdfcreator", "cutepdf", "bullzip", "dopdf",
    "skia/pdf", "chrome", "microsoft excel", "powerpoint",
)

#: Değiştirilme − oluşturulma farkı bundan büyükse şüpheli (sistem PDF'leri tek seferde yazılır).
DATE_TOLERANCE = timedelta(minutes=2)
#: "Küçük" kapatma nesnesi: sayfa alanının bu oranından küçük kutu/görüntü (arka plan değil).
SMALL_COVER_FRACTION = 0.05
#: Kapatma kutusunun "açık renk" sayılması için en düşük kanal değeri (0-1).
LIGHT_FILL = 0.85

_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")
_STYLE_RE = re.compile(r"(bold|italic|oblique|regular|medium|light|semibold|black|condensed|narrow|mt|ps|psmt)+$",
                       re.IGNORECASE)
_BYTERANGE_RE = re.compile(rb"/ByteRange\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]")


@dataclass
class PdfFinding(Finding):
    """PDF kuralı sonucu; ``page`` 0 tabanlı sayfa numarasıdır (belge geneli kurallarda None)."""

    page: Optional[int] = None


@dataclass
class PdfReport:
    """PDF yapı analizi raporu (``ContentReport`` ile aynı ``risk``/``has_evidence`` arayüzü)."""

    findings: list[PdfFinding] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    analyzed: bool = True

    @property
    def has_evidence(self) -> bool:
        return self.analyzed

    @property
    def risk(self) -> float:
        """``risk = 1 − Π (1 − w_kural)`` — yalnızca başarısız (FAIL) kurallar, kural başına bir kez."""
        survive = 1.0
        for rule in {f.rule for f in self.findings if f.status == FAIL}:
            survive *= 1.0 - PDF_RULE_WEIGHTS.get(rule, 0.4)
        return 1.0 - survive

    def page_findings(self, page: int) -> list[PdfFinding]:
        return [f for f in self.findings if f.page == page and f.bbox]


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def parse_pdf_date(value: Optional[str]) -> Optional[datetime]:
    """``D:YYYYMMDDHHmmSS+HH'mm'`` biçimindeki PDF tarihini (saat dilimiyle) çözer."""
    if not value:
        return None
    m = re.match(r"D?:?(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?\s*([Zz+\-])?(\d{2})?'?(\d{2})?", value.strip())
    if not m:
        return None
    year, month, day, hour, minute, second = (int(g) if g else d for g, d in zip(m.groups()[:6], (0, 1, 1, 0, 0, 0)))
    tz = timezone.utc
    if m.group(7) in ("+", "-") and m.group(8):
        offset = timedelta(hours=int(m.group(8)), minutes=int(m.group(9) or 0))
        tz = timezone(offset if m.group(7) == "+" else -offset)
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=tz)
    except ValueError:
        return None


def font_family(name: str) -> str:
    """``ABCDEF+Arial-BoldMT`` / ``Arial Bold`` → ``arial``: alt küme öneki ve stil ekleri atılır."""
    base = _SUBSET_RE.sub("", name or "").strip()
    base = re.split(r"[-, ]", base)[0]
    return (_STYLE_RE.sub("", base) or base).lower()


def _area(r: tuple[float, float, float, float]) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _intersection(a, b) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _to_pixels(rect, page_rect, scale: float) -> tuple[int, int, int, int]:
    return (int((rect[0] - page_rect.x0) * scale), int((rect[1] - page_rect.y0) * scale),
            int(round((rect[2] - page_rect.x0) * scale)), int(round((rect[3] - page_rect.y0) * scale)))


def _keyword(text: str, keywords: tuple[str, ...]) -> Optional[str]:
    lowered = text.lower()
    return next((k for k in keywords if k in lowered), None)


# ---------------------------------------------------------------------------
# Belge geneli kurallar
# ---------------------------------------------------------------------------

def _check_revisions(data: bytes, report: PdfReport) -> None:
    """Dijital imza kapsamı ve artımlı kayıt (revizyon) sayısı."""
    eof_count = data.count(b"%%EOF")
    linearized = b"/Linearized" in data[:2048]
    revisions = max(1, eof_count - (1 if linearized and eof_count > 1 else 0))
    ranges = [tuple(int(v) for v in m.groups()) for m in _BYTERANGE_RE.finditer(data)]
    report.info.update({"revizyon": revisions, "imza_sayisi": len(ranges)})

    signed_to_end = False
    if ranges:
        signed_end = max(c + d for _, _, c, d in ranges)
        trailing = data[signed_end:].strip()
        if trailing:
            report.findings.append(PdfFinding(
                "pdf_signature", FAIL,
                f"Dijital imzadan sonra dosyaya {len(trailing)} bayt eklenmiş: belge imzalandıktan sonra değiştirilmiş."))
        else:
            signed_to_end = True
            report.findings.append(PdfFinding(
                "pdf_signature", PASS,
                "Dijital imza dosyanın sonuna kadar kapsıyor (imzadan sonra ekleme yok; sertifika doğrulanmadı)."))

    if revisions > 1 and not signed_to_end:
        report.findings.append(PdfFinding(
            "pdf_incremental", FAIL,
            f"Dosya ilk oluşturulduktan sonra {revisions - 1} kez artımlı olarak yeniden kaydedilmiş "
            f"({revisions} revizyon). Sistem çıktısı PDF'lerde tek revizyon beklenir."))
    elif revisions > 1:
        report.findings.append(PdfFinding(
            "pdf_incremental", PASS, f"{revisions} revizyon var; ek revizyonlar dijital imza kayıtlarından ibaret."))
    else:
        report.findings.append(PdfFinding("pdf_incremental", PASS, "Tek revizyon: dosya sonradan yeniden kaydedilmemiş."))


def _check_metadata(doc, report: PdfReport) -> None:
    """Üretici yazılım, tarih farkı ve XMP geçmişi."""
    meta = doc.metadata or {}
    xmp = doc.get_xml_metadata() or ""
    producer, creator = meta.get("producer", "") or "", meta.get("creator", "") or ""
    created, modified = parse_pdf_date(meta.get("creationDate")), parse_pdf_date(meta.get("modDate"))
    creator_tool = re.search(r"CreatorTool[>=\"']+([^<\"']+)", xmp)
    tools = " | ".join(t for t in (producer, creator, creator_tool.group(1) if creator_tool else "") if t)
    report.info.update({"producer": producer, "creator": creator,
                        "olusturma": created.isoformat() if created else "", "degistirme": modified.isoformat() if modified else ""})

    if not tools:
        report.findings.append(PdfFinding("pdf_editor", WARN, "Üretici yazılım bilgisi yok (metadata silinmiş olabilir)."))
    elif editor := _keyword(tools, EDITOR_KEYWORDS):
        report.findings.append(PdfFinding("pdf_editor", FAIL, f"PDF düzenleme aracı izi: '{editor}' ({tools})."))
    elif reprint := _keyword(tools, REPRINT_KEYWORDS):
        report.findings.append(PdfFinding(
            "pdf_editor", WARN, f"Belge ofis yazılımı/sanal yazıcıyla üretilmiş: '{reprint}' ({tools}). "
                                "Orijinal sistem çıktısı yeniden oluşturulmuş olabilir."))
    else:
        report.findings.append(PdfFinding("pdf_editor", PASS, f"Üretici yazılım: {tools}."))

    if created and modified:
        if modified - created > DATE_TOLERANCE:
            report.findings.append(PdfFinding(
                "pdf_dates", FAIL, f"Belge oluşturulduktan {modified - created} sonra değiştirilmiş "
                                   f"({created:%d.%m.%Y %H:%M} → {modified:%d.%m.%Y %H:%M})."))
        elif created - modified > DATE_TOLERANCE:
            report.findings.append(PdfFinding("pdf_dates", WARN, "Değiştirilme tarihi oluşturulma tarihinden önce (tutarsız metadata)."))
        else:
            report.findings.append(PdfFinding("pdf_dates", PASS, "Oluşturulma ve değiştirilme tarihleri aynı."))

    saved_events = len(re.findall(r"stEvt:action\s*[>=\"']+\s*(?:saved|modified|edited)", xmp, re.IGNORECASE))
    if saved_events:
        report.findings.append(PdfFinding("pdf_xmp_history", FAIL, f"XMP düzenleme geçmişinde {saved_events} kayıt/düzenleme olayı var."))


# ---------------------------------------------------------------------------
# Sayfa düzeyi kurallar
# ---------------------------------------------------------------------------

def _page_spans(page) -> list[dict[str, Any]]:
    """Metin parçaları: metin, font, kutu, çizim sırası (seqno), görünürlük."""
    spans = []
    for s in page.get_texttrace():
        text = "".join(chr(c[0]) for c in s["chars"]).strip()
        if not text:
            continue
        spans.append({"text": text, "font": s["font"], "bbox": tuple(s["bbox"]), "seqno": s["seqno"],
                      "visible": s["type"] != 3 and s.get("opacity", 1) > 0, "n": len(text)})
    return spans


def _check_covers(page, pi: int, spans: list[dict], scale: float, report: PdfReport) -> None:
    """Gizlenmiş metin (metin → sonra üstüne kutu/görüntü) ve beyaz kutu üzerine yazılmış metin."""
    page_area = _area(tuple(page.rect))
    fills = {d["seqno"]: d for d in page.get_drawings() if d.get("fill") is not None}
    covers = []
    for seqno, (kind, rect) in enumerate(page.get_bboxlog()):
        area = _area(rect)
        if not 0 < area < SMALL_COVER_FRACTION * page_area:
            continue
        if kind == "fill-image":
            covers.append((seqno, rect, "görüntü"))
        elif kind == "fill-path" and seqno in fills:
            d = fills[seqno]
            if min(d["fill"]) >= LIGHT_FILL and (d.get("fill_opacity") or 1) >= 0.9:
                covers.append((seqno, rect, "açık renkli kutu"))
    hidden, whiteout = set(), set()
    for seqno, rect, what in covers:
        for span in spans:
            if not span["visible"] or _area(span["bbox"]) == 0:
                continue
            overlap = _intersection(span["bbox"], rect) / _area(span["bbox"])
            if span["seqno"] < seqno and overlap >= 0.6 and span["text"] not in hidden:
                hidden.add(span["text"])
                report.findings.append(PdfFinding(
                    "pdf_hidden_text", FAIL, f"'{span['text']}' metni sonradan çizilmiş bir {what} altında gizlenmiş "
                                             f"(sayfa {pi + 1}).", _to_pixels(span["bbox"], page.rect, scale), pi))
            elif (span["seqno"] > seqno and overlap >= 0.8 and what != "görüntü"
                  and _area(rect) <= 4 * _area(span["bbox"]) and span["text"] not in whiteout):
                whiteout.add(span["text"])
                report.findings.append(PdfFinding(
                    "pdf_whiteout", WARN, f"'{span['text']}' metni kendisine sıkı oturan açık renkli bir kutunun üzerine "
                                          f"yazılmış (sayfa {pi + 1}); düzeltme bandı benzeri düzenleme olabilir.",
                    _to_pixels(rect, page.rect, scale), pi))


def _check_overlaps(page, pi: int, spans: list[dict], scale: float, report: PdfReport) -> None:
    """Aynı konumda üst üste yazılmış farklı metinler (eski değer + yeni değer)."""
    visible = [s for s in spans if s["visible"] and _area(s["bbox"]) > 0][:1500]
    for i, a in enumerate(visible):
        for b in visible[i + 1:]:
            if a["text"] == b["text"]:
                continue  # Aynı metni iki kez çizmek (sahte kalın yazı) olağandır.
            inter = _intersection(a["bbox"], b["bbox"])
            if inter and inter / min(_area(a["bbox"]), _area(b["bbox"])) >= 0.6:
                report.findings.append(PdfFinding(
                    "pdf_overlap", FAIL, f"'{a['text']}' ve '{b['text']}' metinleri aynı konuma üst üste yazılmış "
                                         f"(sayfa {pi + 1}).", _to_pixels(b["bbox"], page.rect, scale), pi))


def _mixed_line(span: dict, spans: list[dict], family: str) -> bool:
    """Aynı satırda (dikeyde ≥ %50 örtüşen) başka font ailesiyle yazılmış görünür metin var mı?

    "Tutar:" etiketi belgenin fontuyla, yanındaki tutar başka fontla yazılmışsa bu, mevcut satıra
    sonradan metin eklemenin tipik izidir. Tek başına satırda duran farklı font (logo, barkod
    numarası) ise olağandır.
    """
    y0, y1 = span["bbox"][1], span["bbox"][3]
    for other in spans:
        if other is span or not other["visible"] or font_family(other["font"]) == family:
            continue
        overlap = min(y1, other["bbox"][3]) - max(y0, other["bbox"][1])
        if overlap >= 0.5 * min(y1 - y0, other["bbox"][3] - other["bbox"][1]):
            return True
    return False


def _check_fonts(pages_spans: list[tuple[Any, int, list[dict]]], scale: float, report: PdfReport) -> None:
    """Az kullanılan font ve alt kümelenmiş fontlar arasında alt kümelenmemiş font.

    Böyle bir font aynı satırda belgenin asıl fontuyla yan yana geçiyorsa FAIL, tek başınaysa WARN.
    """
    chars, spans_by_family, subset_chars = Counter(), defaultdict(list), Counter()
    for page, pi, spans in pages_spans:
        for s in spans:
            if not s["visible"]:
                continue
            fam = font_family(s["font"])
            chars[fam] += s["n"]
            spans_by_family[fam].append((page, pi, s))
            subset_chars[bool(_SUBSET_RE.match(s["font"]))] += s["n"]
    total = sum(chars.values())
    report.info["fontlar"] = {fam: n for fam, n in chars.most_common()}
    if total < 40 or len(chars) < 2:
        return
    mostly_subset = subset_chars[True] >= 0.8 * total
    page_spans = {pi: spans for _, pi, spans in pages_spans}
    for fam, n in chars.items():
        uses = spans_by_family[fam]
        rare = (len(uses) <= 2 and n < 0.10 * total) or (len(uses) <= 3 and n < 0.03 * total)
        lone_full_font = mostly_subset and not any(_SUBSET_RE.match(s["font"]) for _, _, s in uses) and n < 0.05 * total
        if not (rare or lone_full_font):
            continue
        for page, pi, s in uses:
            mixed = _mixed_line(s, page_spans[pi], fam)
            reason = ("alt kümelenmemiş (sonradan gömülmüş) font" if lone_full_font
                      else f"belgede yalnızca burada geçen font ({s['font']})")
            where = "; aynı satırdaki diğer metinler farklı fontla yazılmış" if mixed else ""
            report.findings.append(PdfFinding(
                "pdf_font", FAIL if mixed else WARN, f"'{s['text']}' metni {reason} ile yazılmış{where} (sayfa {pi + 1}).",
                _to_pixels(s["bbox"], page.rect, scale), pi))


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

def check_pdf(data: bytes, dpi: int = 200) -> PdfReport:
    """PDF baytlarını analiz eder. ``bbox``'lar ``dpi`` çözünürlüğündeki sayfa pikselleridir."""
    import pymupdf

    report = PdfReport()
    scale = dpi / 72.0
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # Bozuk dosya
        report.analyzed = False
        report.findings.append(PdfFinding("pdf_structure", WARN, f"PDF açılamadı: {exc.__class__.__name__}: {exc}"))
        return report
    with doc:
        if doc.needs_pass:
            report.analyzed = False
            report.findings.append(PdfFinding("pdf_structure", WARN, "PDF parola korumalı; yapı analizi yapılamadı."))
            return report
        report.info["sayfa"] = doc.page_count
        _check_revisions(data, report)
        _check_metadata(doc, report)
        pages_spans = []
        for pi, page in enumerate(doc):
            spans = _page_spans(page)
            page_area = _area(tuple(page.rect))
            big_image = any(_area(tuple(info["bbox"])) >= 0.5 * page_area for info in page.get_image_info())
            if big_image and not any(s["visible"] for s in spans):
                report.findings.append(PdfFinding(
                    "pdf_structure", WARN, f"Sayfa {pi + 1} taranmış/görüntü tabanlı: yapı analizi sınırlı, "
                                           "görüntü modeli ve OCR kuralları esas alınmalı.", page=pi))
            elif not big_image and any(not s["visible"] for s in spans):
                report.findings.append(PdfFinding(
                    "pdf_invisible_text", WARN, f"Sayfa {pi + 1}'de görünmez metin var.", page=pi))
            _check_covers(page, pi, spans, scale, report)
            _check_overlaps(page, pi, spans, scale, report)
            pages_spans.append((page, pi, spans))
        _check_fonts(pages_spans, scale, report)
    return report
