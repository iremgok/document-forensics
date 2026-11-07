"""``src/ocr_checker.py`` kural motoru: IBAN, VKN/TCKN, tutar ayrıştırma, e-Arşiv toplam zinciri."""

import pytest

from src.ocr_checker import (FAIL, PASS, WARN, check_document_text, complete_tckn, complete_vkn,
                             compute_iban_check_digits, find_amounts, iban_is_valid, lines_from_text, normalize_iban,
                             parse_amount, tckn_is_valid, vkn_is_valid)

VALID_IBAN = "TR33 0006 1005 1978 6457 8413 26"


def statuses(text: str, rule: str) -> list[str]:
    return [f.status for f in check_document_text(lines_from_text(text)).findings if f.rule == rule]


# --- IBAN ---------------------------------------------------------------------------------

def test_iban_valid_and_check_digits():
    assert iban_is_valid(VALID_IBAN)
    bban = normalize_iban(VALID_IBAN)[4:]
    assert compute_iban_check_digits("TR", bban) == "33"


def test_iban_tampered_single_digit_fails():
    tampered = VALID_IBAN[:-1] + "7"
    assert not iban_is_valid(tampered)
    assert statuses(f"IBAN: {tampered}", "iban") == [FAIL]


def test_iban_ocr_letter_digit_confusion_is_normalised():
    ocr = "IBAN: TR33 0006 1O05 1978 6457 84I3 26"  # O ↔ 0, I ↔ 1
    assert statuses(ocr, "iban") == [PASS]
    # Gövdenin %80'inden azı gerçek rakamsa (kelime) aday sayılmaz: yanlış pozitif koruması.
    assert statuses("IBAN: TR33 OOO6 1OO5 1978 6457 8413 26", "iban") == []


def test_iban_split_across_lines():
    text = "IBAN: TR33 0006 1005 1978\n6457 8413 26"
    assert statuses(text, "iban") == [PASS]


@pytest.mark.parametrize("text", ["IBAN: BE11NTINIZ ödemeniz için teşekkürler", "TRENDYOL siparişiniz",
                                  "Bilgi itibarıyle IT18ARIYLE geçerlidir IBAN"])
def test_iban_false_positive_words_are_ignored(text):
    assert FAIL not in statuses(text, "iban")


def test_low_confidence_failure_is_warning():
    lines = lines_from_text(f"IBAN: {VALID_IBAN[:-1]}7")
    lines[0].confidence = 0.3
    assert [f.status for f in check_document_text(lines).findings if f.rule == "iban"] == [WARN]


# --- VKN / TCKN ---------------------------------------------------------------------------

def test_vkn_and_tckn_algorithms():
    vkn, tckn = complete_vkn("123456789"), complete_tckn("123456789")
    assert vkn_is_valid(vkn) and tckn_is_valid(tckn)
    assert tckn_is_valid("10000000146")
    bad_vkn = vkn[:-1] + str((int(vkn[-1]) + 1) % 10)
    bad_tckn = tckn[:-1] + str((int(tckn[-1]) + 1) % 10)
    assert not vkn_is_valid(bad_vkn) and not tckn_is_valid(bad_tckn)
    assert not tckn_is_valid("0" + tckn[1:])  # ilk hane 0 olamaz


def test_ids_checked_by_length_after_label():
    vkn, tckn = complete_vkn("987654321"), complete_tckn("987654321")
    assert statuses(f"VKN: {vkn}", "vkn") == [PASS]
    assert statuses(f"VKN/TCKN: {tckn}", "tckn") == [PASS]  # şahıs şirketi: VKN etiketiyle 11 hane
    assert statuses(f"T.C. Kimlik No: {tckn[:-1]}{(int(tckn[-1]) + 1) % 10}", "tckn") == [FAIL]
    assert statuses(f"Sipariş No {tckn}", "tckn") == []  # etiketsiz numara denetlenmez


# --- Tutar ayrıştırma ---------------------------------------------------------------------

@pytest.mark.parametrize("text, value", [("1.234,56", 1234.56), ("1,234.56", 1234.56), ("2. 691,61", 2691.61),
                                         ("691, 61", 691.61), ("12,00", 12.0)])
def test_parse_amount(text, value):
    assert parse_amount(text) == pytest.approx(value)


def test_find_amounts_handles_ocr_spaces_dates_and_rates():
    assert find_amounts("Genel Toplam 2. 691,61 TL") == [pytest.approx(2691.61)]
    assert find_amounts("Tarih: 20.03.2021") == []
    assert find_amounts("KDV %20 180,00") == [pytest.approx(180.0)]


# --- e-Arşiv toplam zinciri ---------------------------------------------------------------

EARSIV = """Mal Hizmet Toplam Tutarı 1.000,00 TL
Toplam İskonto 100,00 TL
Hesaplanan KDV (%20) 180,00 TL
Vergiler Dahil Toplam Tutar 1.080,00 TL"""


def test_earsiv_chain_consistent():
    report = check_document_text(lines_from_text(EARSIV))
    assert report.fields["matrah"] == pytest.approx(900.0)
    assert statuses(EARSIV, "total_math") == [PASS]
    assert statuses(EARSIV, "vat_rate") == [PASS]
    assert report.risk == 0.0


def test_earsiv_chain_tampered_total_fails():
    tampered = EARSIV.replace("1.080,00", "1.880,00")
    assert statuses(tampered, "total_math") == [FAIL]
    assert check_document_text(lines_from_text(tampered)).risk > 0


def test_multi_rate_vat_is_summed():
    text = """Matrah 1.000,00
Hesaplanan KDV (%10) 50,00
Hesaplanan KDV (%20) 100,00
Ödenecek Tutar 1.150,00"""
    assert statuses(text, "total_math") == [PASS]
    assert statuses(text, "vat_rate") == []  # çok oranlı belgede tek oran kontrolü yapılmaz


def test_label_and_amount_on_separate_lines():
    text = "Matrah\n500,00\nHesaplanan KDV (%20) 100,00\nVergiler Dahil Toplam Tutar\n600,00"
    assert statuses(text, "total_math") == [PASS]
