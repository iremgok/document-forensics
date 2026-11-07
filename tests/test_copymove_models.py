"""``src/copymove.py`` (sentetik örnek) ve ``src/models.py`` (küçük kodlayıcı, ağırlık indirmeden)."""

import cv2
import numpy as np
import pytest
import torch

from src.copymove import COPYMOVE_DECISION_RISK, detect_copy_move
from src.models import DualStreamForensicNet, load_compatible_state, load_history, save_checkpoint, \
    sliding_window_maps, sliding_window_predict


def page(noise: float, seed: int = 0) -> np.ndarray:
    img = np.full((400, 700), 235, np.uint8)
    for k, txt in enumerate(["12.450,00 TL", "IBAN TR33 0006 1005", "Tarih 07.11.2026", "Toplam 3.271,90"]):
        cv2.putText(img, txt, (30, 70 + 80 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 30, 2, cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


# --- copymove ---------------------------------------------------------------------------------

def test_copy_move_found_in_noisy_document():
    img = page(3.0)
    forged = img.copy()
    forged[300:345, 514:552] = img[40:85, 24:62]  # ilk satırdaki "1" çevresiyle birlikte kopyalanır
    report = detect_copy_move(forged)
    assert report.risk >= COPYMOVE_DECISION_RISK
    top = report.pairs[0]
    assert {top.box_a[0] // 100, top.box_b[0] // 100} == {0, 5}  # kaynak (x≈30) ve hedef (x≈520)
    assert top.noise_corr > 0.9


def test_natural_repeats_not_flagged():
    assert detect_copy_move(page(3.0)).risk < COPYMOVE_DECISION_RISK   # aynı rakamlar, bağımsız gürültü
    # Düz dijital zemin: aynı glifler piksel piksel aynı, ama kanıt üretilmez (doğal tekrarla ayırt edilemez).
    flat = page(0.0)
    forged = flat.copy()
    forged[300:345, 514:552] = flat[40:85, 24:62]
    assert detect_copy_move(flat).risk == 0.0 and detect_copy_move(forged).risk == 0.0


def test_copy_move_survives_mild_jpeg():
    img = page(3.0)
    forged = img.copy()
    forged[300:345, 514:552] = img[40:85, 24:62]
    _, buf = cv2.imencode(".jpg", forged, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert detect_copy_move(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)).risk >= COPYMOVE_DECISION_RISK


def test_copy_move_handles_dark_or_empty_images():
    assert detect_copy_move(np.zeros((64, 64, 3), np.uint8)).risk == 0.0
    assert detect_copy_move(np.full((64, 64), 255, np.uint8)).pairs == []


# --- models -----------------------------------------------------------------------------------

def small(streams: str) -> DualStreamForensicNet:
    torch.manual_seed(0)
    return DualStreamForensicNet("resnet18", encoder_weights=None, streams=streams).eval()


@pytest.mark.parametrize("streams", ["rgb", "srm", "rgb+srm", "rgb+srm+dct"])
def test_stream_options_forward(streams):
    model = small(streams)
    with torch.no_grad():
        mask_logits, cls_logit = model(torch.randn(2, 3, 64, 64))
    assert mask_logits.shape == (2, 1, 64, 64) and cls_logit.shape == (2,)


def test_unknown_stream_rejected():
    with pytest.raises(ValueError):
        small("rgb+dct")


def test_dct_zero_init_is_identity():
    base = small("rgb+srm")
    dct = small("rgb+srm+dct")
    load_compatible_state(dct, base.state_dict())
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a, b = base(x), dct(x)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)


def test_compatible_state_slices_first_conv():
    rgb = small("rgb")
    load_compatible_state(rgb, small("rgb+srm").state_dict())  # 6 → 3 giriş kanalı
    with torch.no_grad():
        assert rgb(torch.randn(1, 3, 64, 64))[0].shape == (1, 1, 64, 64)


def test_sliding_window_multi_temperature_matches_single():
    model = small("rgb+srm")
    image = torch.randn(3, 80, 100)
    maps, doc_logit, windows, logit_map = sliding_window_maps(model, image, tile=64, temperatures=(1.0, 4.0),
                                                              return_logit_map=True)
    single, single_logit = sliding_window_predict(model, image, tile=64, temperature=4.0)
    assert np.allclose(maps[1], single, atol=1e-6) and doc_logit == pytest.approx(single_logit)
    assert maps[0].shape == logit_map.shape == (80, 100) and len(windows) >= 4
    assert not np.allclose(maps[0], maps[1])


def test_history_round_trip_in_checkpoint(tmp_path):
    model = small("rgb")
    history = [{"epoch": 1, "val/score": 0.5}, {"epoch": 2, "val/score": 0.6}]
    save_checkpoint(tmp_path / "best.pth", model, {"encoder": "resnet18", "streams": "rgb"}, 2, {}, {"history": history})
    loaded = load_history(tmp_path)  # history.csv yok → best.pth içinden
    assert loaded["val/score"].tolist() == [0.5, 0.6]
    (tmp_path / "history.csv").write_text("epoch,val/score\n1,0.9\n", encoding="utf-8")
    assert load_history(tmp_path)["val/score"].tolist() == [0.9]  # CSV önceliklidir
    save_checkpoint(tmp_path / "old.pth", model, {}, 1, {})
    assert load_history(tmp_path / "old.pth") is None  # eski checkpoint
