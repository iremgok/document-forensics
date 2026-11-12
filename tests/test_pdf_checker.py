"""``src/pdf_checker.py``: bellekte ``pymupdf`` ile üretilen sentetik PDF'ler.

Artımlı kayıt testi için pymupdf ``saveIncr`` gerçek bir dosya ister; ``tmp_path`` kullanılır.
"""

import pymupdf
import pytest

from src.ocr_checker import FAIL
from src.pdf_checker import check_pdf

STAMP = "D:20260101120000+03'00'"


def base_doc(producer: str = "Banka Dekont Sistemi") -> pymupdf.Document:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    lines = ["DEKONT", "Gönderen: AHMET YILMAZ", "Alıcı: AYŞE KAYA", "Tutar: 12.450,00 TL",
             "Açıklama: Kira ödemesi Ocak 2026", "Referans: 2026010112345678"]
    for i, text in enumerate(lines):
        page.insert_text((60, 80 + 24 * i), text, fontname="helv", fontsize=11)
    doc.set_metadata({"producer": producer, "creator": producer, "creationDate": STAMP, "modDate": STAMP})
    return doc


def fails(report) -> set[str]:
    return {f.rule for f in report.findings if f.status == FAIL}


def test_clean_pdf_has_no_fail():
    report = check_pdf(base_doc().tobytes(garbage=3))
    assert report.analyzed and fails(report) == set()
    assert report.risk == 0.0 and report.info["revizyon"] == 1


def test_incremental_save_is_flagged(tmp_path):
    path = tmp_path / "dekont.pdf"
    base_doc().save(path)
    doc = pymupdf.open(path)
    doc[0].insert_text((300, 152), "99", fontname="helv", fontsize=11)  # sonradan eklenen değer
    doc.saveIncr()
    doc.close()
    report = check_pdf(path.read_bytes())
    assert "pdf_incremental" in fails(report)
    assert report.info["revizyon"] >= 2 and report.risk > 0


def test_editor_producer_is_flagged():
    assert "pdf_editor" in fails(check_pdf(base_doc("Sejda PDF Desktop").tobytes()))


def test_hidden_text_under_white_box():
    doc = base_doc()
    page = doc[0]
    tutar = page.search_for("Tutar: 12.450,00 TL")[0]
    page.draw_rect(tutar + (-1, -1, 1, 1), color=None, fill=(1, 1, 1))  # metnin üstüne beyaz kutu
    assert "pdf_hidden_text" in fails(check_pdf(doc.tobytes()))
    # Kutunun metnin yalnızca küçük bir kısmını örtmesi (< %60) gizleme sayılmaz.
    doc = base_doc()
    doc[0].draw_rect(doc[0].search_for("TL")[0], color=None, fill=(1, 1, 1))
    assert "pdf_hidden_text" not in fails(check_pdf(doc.tobytes()))


def test_overlapping_texts():
    doc = base_doc()
    doc[0].insert_text((60, 152), "Tutar: 19.450,00 TL", fontname="helv", fontsize=11)  # eski değerin üstüne
    assert "pdf_overlap" in fails(check_pdf(doc.tobytes()))


def test_modified_long_after_creation():
    doc = base_doc()
    doc.set_metadata({"producer": "Banka Dekont Sistemi", "creationDate": STAMP, "modDate": "D:20260115120000+03'00'"})
    assert "pdf_dates" in fails(check_pdf(doc.tobytes()))


def test_template_generator_dates_not_flagged():
    """AEM Forms / LiveCycle: CreationDate şablonun tarihidir (Garanti dekontları: 2017 → üretim günü)."""
    doc = base_doc()
    doc.set_metadata({"producer": "Adobe Experience Manager forms output", "creator": "Designer 6.5",
                      "creationDate": "D:20170508164302+03'00'", "modDate": "D:20261023131152+03'00'"})
    assert "pdf_dates" not in fails(check_pdf(doc.tobytes()))


@pytest.mark.parametrize("a, b", [("FrutigerRoman", "FrutigerBold"), ("ABCDEF+Frutiger-Black", "Frutiger"),
                                  ("TimesNewRomanPSMT", "TimesNewRomanPS-BoldMT"), ("Arial-BoldMT", "Arial")])
def test_font_family_ignores_style_suffixes(a, b):
    from src.pdf_checker import font_family
    assert font_family(a) == font_family(b)


@pytest.mark.parametrize("data", [b"%PDF-1.4 bozuk", b"not a pdf"])
def test_broken_pdf_not_analyzed(data):
    report = check_pdf(data)
    assert not report.analyzed and report.risk == 0.0
