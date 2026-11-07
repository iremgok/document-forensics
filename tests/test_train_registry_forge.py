"""``train.sampling_weights`` / ön ayarlar / karar metriği, ``src/model_registry`` ve ``forge_real`` edinim kararı."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

import train
from src import model_registry as reg
from src.forge_real import ACQUISITION_FILE, decide_acquisition


# --- train ------------------------------------------------------------------------------------

def test_sampling_weights_balanced_and_type_weighted():
    frame = pd.DataFrame({"label": [0, 0, 1, 1, 1, 1],
                          "tamper_type": ["orig", "orig", "copy_move", "name_swap", "name_swap", "delete"]})
    w = train.sampling_weights(frame)
    assert w.sum() == pytest.approx(1.0)
    assert w[:2].sum() == pytest.approx(0.5) and w[2:].sum() == pytest.approx(0.5)
    w = train.sampling_weights(frame, type_weights={"copy_move": 3.0})
    assert w[2:].sum() == pytest.approx(0.5)            # sahte sınıfın toplam payı değişmez
    assert w[2] == pytest.approx(3 * w[3])              # copy_move 3 kat
    w = train.sampling_weights(frame, balance=False)
    assert np.allclose(w, 1 / 6)


def test_parse_type_weights():
    assert train.parse_type_weights(["copy_move=3", "digit_change=3,delete=2"]) == {
        "copy_move": 3.0, "digit_change": 3.0, "delete": 2.0}


def parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    return train.parse_args()


def test_stage_presets_and_cli_precedence(monkeypatch):
    args = parse(monkeypatch, "--stage", "stageF", "--run-dir", "x")
    assert args.monitor == "decision_tpr_at_fpr02"
    assert args.type_weights == ["copy_move=3", "digit_change=3", "delete=2"]
    assert args.init.replace("\\", "/").endswith("checkpoints/stageE/best.pth") and args.lr < 1e-4
    assert parse(monkeypatch, "--stage", "stageE", "--run-dir", "x").monitor == "decision_tpr_at_fpr02"
    assert parse(monkeypatch, "--stage", "stageE", "--run-dir", "x", "--monitor", "score").monitor == "score"
    assert parse(monkeypatch, "--run-dir", "x").monitor == "score"


def test_decision_metric_uses_rule():
    labels = np.array([0] * 50 + [1] * 50)
    doc_logits = np.r_[np.full(50, -2.0), np.full(50, -2.0)]   # belge skoru ayırt etmiyor
    max_pixel = np.full(100, 0.9)
    regions = np.zeros((100, len(train.DECISION_PIXEL_GRID)))
    regions[50:, :5] = 0.01                                    # bölge kanıtı yalnız sahtelerde
    assert train.decision_tpr_at_fpr(doc_logits, max_pixel, regions, labels, 0.02) == 1.0
    assert np.isnan(train.decision_tpr_at_fpr(doc_logits, max_pixel, regions, np.ones(100), 0.02))


# --- model_registry ----------------------------------------------------------------------

def test_registry_known_and_archived_labels():
    assert reg.describe("checkpoints/stageE/best.pth").status == reg.RECOMMENDED
    moved = reg.describe("checkpoints/stageD/best.pth")       # henüz arşive taşınmamış
    assert moved.status == reg.EXPERIMENTAL and moved.path == "checkpoints/stageD/best.pth"
    assert reg.is_archived("_arsiv/checkpoints/stageD/best.pth")


def test_registry_unknown_model_classified_from_metrics():
    bad = reg.describe("checkpoints/x/best.pth", {"stage": "stageC", "epoch": 3, "val_image_fpr": 0.99})
    good = reg.describe("checkpoints/y/best.pth", {"stage": "stageE", "epoch": 3, "val_image_fpr": 0.1,
                                                   "val_pixel_f1": 0.5})
    assert bad.status == reg.EXPERIMENTAL and good.status == reg.COMPARISON
    assert reg.reliability_warning({"sources": ["doctamper"]})


def test_calibration_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "checkpoints" / "run"
    run.mkdir(parents=True)
    assert reg.calibration_status(run / "best.pth") is None
    assert reg.calibration_label(run / "best.pth") == reg.UNCALIBRATED_LABEL
    (run / "calibration.json").write_text(json.dumps({"temperature": 1.0, "optimal_threshold": 0.6}), encoding="utf-8")
    assert reg.calibration_label(run / "best.pth") == reg.UNCALIBRATED_LABEL  # elle girilmiş
    (run / "calibration.json").write_text(json.dumps({"doc_temperature": 3.0, "doc_threshold": 0.7,
                                                      "pixel_temperature": 1.0, "pixel_threshold": 0.5,
                                                      "val_images": 200}), encoding="utf-8")
    assert reg.is_calibrated(run / "best.pth") and reg.calibration_label(run / "best.pth") == "kalibre"
    assert not (tmp_path / "checkpoints" / "calibration.json").exists()  # sorgu dosya oluşturmaz


# --- forge_real edinim kararı ------------------------------------------------------------

def flat_page() -> np.ndarray:
    page = np.full((300, 300, 3), 250, np.uint8)
    page[100:110, 50:250] = 20
    return page


def noisy_page() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(flat_page().astype(float) + rng.normal(0, 4, (300, 300, 3)), 0, 255).astype(np.uint8)


def test_acquisition_from_source_type(tmp_path):
    inst = tmp_path / "trendyol"
    inst.mkdir()
    # Kayıpsız PNG gürültülü görünse bile kaynak türü kazanır (eski hata: 50 PNG "photo" sanılmıştı).
    assert decide_acquisition(inst / "a.png", tmp_path, image=noisy_page()) == ("digital", "png_kayipsiz")
    assert decide_acquisition(inst / "a.pdf", tmp_path) == ("digital", "pdf")
    assert decide_acquisition(inst / "a.jpg", tmp_path, image=flat_page()) == ("digital", "piksel_gurultusu")
    assert decide_acquisition(inst / "a.jpg", tmp_path, image=noisy_page()) == ("photo", "piksel_gurultusu")
    with pytest.raises(ValueError):
        decide_acquisition(inst / "a.jpg", tmp_path)  # belirsiz kaynak + görüntü yok


def test_acquisition_file_and_override_win(tmp_path):
    inst = tmp_path / "ziraat"
    (inst / "alt").mkdir(parents=True)
    (inst / ACQUISITION_FILE).write_text("# kurum klasörü\nscan\n", encoding="utf-8")
    assert decide_acquisition(inst / "alt" / "a.png", tmp_path) == ("scan", ACQUISITION_FILE)
    assert decide_acquisition(inst / "a.png", tmp_path, override="photo") == ("photo", "arguman")
    (inst / ACQUISITION_FILE).write_text("kamera\n", encoding="utf-8")
    with pytest.raises(ValueError):
        decide_acquisition(inst / "a.png", tmp_path)
    with pytest.raises(ValueError):
        decide_acquisition(inst / "a.png", tmp_path, override="kamera")
