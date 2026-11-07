"""Karar kuralı (``decide_forged`` / ``is_forged``), çalışma noktası seçimi ve kalibrasyon şemasının
eski dosyalarla uyumu (``predict.py``, ``evaluate.py``)."""

import json

import numpy as np
import pandas as pd
import pytest

import evaluate
from predict import (CLEAN_PIXEL_LIMIT, MIN_REGION_FRACTION, Calibration, calibration_path_for, decide_forged,
                     decide_with, find_calibration_file, is_forged, load_calibration, load_calibration_info,
                     select_operating_point)


def blob_map(value: float, size: int = 100, box: int = 0) -> np.ndarray:
    """Düşük zemin (0.01) + ``box``×``box`` piksellik ``value`` olasılıklı bölge."""
    prob = np.full((size, size), 0.01, np.float32)
    prob[:box, :box] = value
    return prob


# --- Karar kuralı -------------------------------------------------------------------------

def test_hybrid_condition_needs_pixel_trace():
    assert decide_forged(0.9, CLEAN_PIXEL_LIMIT, 0.0, 0.5)
    assert not decide_forged(0.9, CLEAN_PIXEL_LIMIT - 1e-3, 0.0, 0.5)  # temiz harita: belge skoru tek başına yetmez
    assert not decide_forged(0.4, 0.9, 0.0, 0.5)


def test_region_condition_alone_is_enough():
    assert decide_forged(0.0, 0.9, MIN_REGION_FRACTION, 0.5)
    assert not decide_forged(0.0, 0.9, MIN_REGION_FRACTION / 2, 0.5)
    assert decide_forged(0.0, 0.9, 0.001, 0.5, min_region_fraction=0.001)


def test_is_forged_uses_separate_pixel_threshold():
    prob = blob_map(0.7, box=10)  # bölge = %1 sayfa
    assert not is_forged(0.1, prob, 0.77)            # eski tek eşik: 0.7 < 0.77 → bölge yok
    assert is_forged(0.1, prob, 0.77, pixel_threshold=0.6)
    cal = Calibration(doc_threshold=0.77, pixel_threshold=0.6)
    assert decide_with(cal, 0.1, prob)


def test_tiny_region_rejected():
    prob = blob_map(0.99, box=2)  # 4 piksel = %0.04 < %0.05
    assert not is_forged(0.0, prob, 0.5)


# --- Çalışma noktası seçimi ---------------------------------------------------------------

def test_select_operating_point_respects_target_fpr():
    labels = np.array([0] * 20 + [1] * 20)
    doc = np.r_[np.linspace(0.0, 0.6, 20), np.linspace(0.4, 1.0, 20)]
    max_pixel = np.full(40, 0.5)
    # 3 aday piksel eşiği. Bölge: sahtelerin yarısı yalnız düşük eşikte, orijinallerden biri de düşük eşikte bölge verir.
    regions = np.zeros((40, 3))
    regions[20:30, 0] = 0.01
    regions[0, 0] = 0.01
    best = select_operating_point(doc, max_pixel, regions, labels, np.round(np.arange(0.05, 0.96, 0.05), 2), 0.05)
    assert best["feasible"] and best["fpr"] <= 0.05
    # Kaba kuvvetle doğrula: aynı ızgarada FPR ≤ 0.05 altındaki en yüksek TPR
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_tpr = max(((((doc >= d) & (max_pixel >= CLEAN_PIXEL_LIMIT)) | (regions[:, p] >= MIN_REGION_FRACTION))[labels == 1].mean()
                    for d in grid for p in range(3)
                    if ((((doc >= d) & (max_pixel >= CLEAN_PIXEL_LIMIT)) | (regions[:, p] >= MIN_REGION_FRACTION))[labels == 0].mean() <= 0.05)),
                   default=0)
    assert best["tpr"] == pytest.approx(best_tpr)
    assert best["pixel_index"] == 0  # bölge kanıtı yalnız en düşük piksel eşiğinde var


def test_select_operating_point_infeasible_returns_lowest_fpr():
    labels = np.array([0, 0, 1, 1])
    best = select_operating_point(np.ones(4), np.ones(4), np.ones((4, 2)), labels, np.array([0.5]), 0.0)
    assert not best["feasible"] and best["fpr"] == 1.0


def test_evaluate_joint_selection_beats_or_equals_tied_threshold():
    rng = np.random.default_rng(0)
    n = 200
    labels = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    pred = pd.DataFrame({"label": labels, "doc_logit": rng.normal(labels * 1.5, 1.0),
                         "max_pixel": rng.uniform(0.05, 1.0, n)})
    regions = np.zeros((n, len(evaluate.THRESHOLDS)))
    strength = rng.uniform(0, 1, n) * (0.5 + 0.5 * labels)
    regions[:] = (strength[:, None] > evaluate.THRESHOLDS[None, :]) * 0.01
    pred.attrs["regions"] = regions
    joint = evaluate.select_thresholds(pred, 1.0, 0.05, joint=True)
    tied = evaluate.select_thresholds(pred, 1.0, 0.05, joint=False)
    assert joint["fpr"] <= 0.05 and tied["fpr"] <= 0.05
    assert joint["tpr"] >= tied["tpr"]
    assert tied["doc_threshold"] == tied["pixel_threshold"]


def test_pixel_temperature_falls_back_to_one_without_masks():
    frame = pd.DataFrame({"label": [0, 1], "mask_path": ["", ""]})
    assert not evaluate.has_pixel_masks(frame)
    pred = pd.DataFrame({"label": [0, 1]})
    assert evaluate.resolve_pixel_temperature("fit", 4.2, pred, masks=False) == 1.0
    assert evaluate.resolve_pixel_temperature("doc", 4.2, pred, masks=True) == 4.2
    assert evaluate.resolve_pixel_temperature("2.5", 4.2, pred, masks=True) == 2.5


# --- Kalibrasyon şeması ---------------------------------------------------------------------

def test_legacy_schema_applies_single_t_to_both_sides():
    cal = Calibration.from_dict({"temperature": 4.2, "optimal_threshold": 0.77, "val_images": 1884})
    assert cal.legacy and cal.calibrated
    assert (cal.doc_temperature, cal.pixel_temperature) == (4.2, 4.2)
    assert (cal.doc_threshold, cal.pixel_threshold) == (0.77, 0.77)
    assert Calibration.from_dict({"temperature": 1.0, "threshold": 0.6}).doc_threshold == 0.6


def test_manual_general_file_is_uncalibrated():
    assert not Calibration.from_dict({"temperature": 1.0, "optimal_threshold": 0.60}).calibrated


def test_new_schema_round_trip_keeps_legacy_keys():
    cal = Calibration(doc_temperature=3.0, doc_threshold=0.7, pixel_temperature=1.2, pixel_threshold=0.55,
                      target_fpr=0.05, val_fpr=0.04, val_tpr=0.8, val_images=100, checkpoint="x/best.pth")
    data = json.loads(json.dumps(cal.to_dict()))
    assert data["temperature"] == 3.0 and data["optimal_threshold"] == 0.7  # eski okuyucular için
    back = Calibration.from_dict(data)
    assert back == cal and not back.legacy


def test_invalid_temperature_rejected():
    with pytest.raises(ValueError):
        Calibration.from_dict({"doc_temperature": 0.0, "doc_threshold": 0.5})


def test_calibration_lookup_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "checkpoints" / "run"
    run.mkdir(parents=True)
    assert calibration_path_for(run / "best.pth") == run / "calibration.json"
    assert calibration_path_for(tmp_path / "checkpoints" / "stageB.pth") == tmp_path / "checkpoints" / "stageB.calibration.json"
    assert find_calibration_file(run / "best.pth") is None
    (run / "calibration.json").write_text(json.dumps({"temperature": 2.0, "optimal_threshold": 0.6}), encoding="utf-8")
    t, thr, path = load_calibration(checkpoint=run / "best.pth")  # eski API: (belge T, belge eşiği, yol)
    assert (t, thr, path) == (2.0, 0.6, run / "calibration.json")


def test_missing_calibration_creates_neutral_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cal = load_calibration_info(checkpoint=tmp_path / "yok" / "best.pth")
    assert (cal.doc_temperature, cal.doc_threshold) == (1.0, 0.5) and not cal.calibrated
    assert (tmp_path / "checkpoints" / "calibration.json").is_file()


def test_repo_stage_e_calibration_still_reads():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "checkpoints" / "stageE" / "calibration.json"
    if not path.is_file():
        pytest.skip("Stage E kalibrasyonu yok")
    cal = load_calibration_info(str(path))
    assert cal.calibrated and cal.doc_temperature == cal.pixel_temperature  # eski tek-T dosya
