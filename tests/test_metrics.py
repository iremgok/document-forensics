"""``src/metrics.py``: AUC, AP ve TPR@FPR scikit-learn ile karşılaştırılır; bootstrap güven aralığı."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.metrics import average_precision, bootstrap_ci, fit_temperature, roc_auc, tpr_at_fpr


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    # Eşit skorlar da içersin (sıra ortalaması / eşik grupları sınanır).
    scores = np.round(rng.normal(labels * 1.2, 1.0), 1)
    return scores, labels


def test_roc_auc_matches_sklearn(scored):
    scores, labels = scored
    assert roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def test_average_precision_matches_sklearn(scored):
    scores, labels = scored
    assert average_precision(scores, labels) == pytest.approx(average_precision_score(labels, scores), abs=1e-12)


def test_tpr_at_fpr_matches_roc_curve(scored):
    scores, labels = scored
    for target in (0.01, 0.02, 0.05, 0.2):
        fpr, tpr, _ = roc_curve(labels, scores)
        expected = tpr[fpr <= target + 1e-12].max()
        assert tpr_at_fpr(scores, labels, target) == pytest.approx(expected, abs=1e-12)


def test_single_class_is_nan():
    assert np.isnan(roc_auc([0.1, 0.2], [1, 1]))
    assert np.isnan(tpr_at_fpr([0.1, 0.2], [0, 0], 0.05))


def test_bootstrap_ci_contains_estimate_and_is_stratified(scored):
    scores, labels = scored
    lo, hi = bootstrap_ci(roc_auc, [scores, labels], n_boot=300, seed=1, strata=labels)
    point = roc_auc(scores, labels)
    assert lo < point < hi
    assert 0.0 < hi - lo < 0.3
    # Aynı tohum → aynı aralık (tekrarlanabilirlik)
    assert (lo, hi) == bootstrap_ci(roc_auc, [scores, labels], n_boot=300, seed=1, strata=labels)


def test_bootstrap_small_sample_keeps_both_classes():
    scores, labels = np.array([0.9, 0.8, 0.1, 0.2, 0.3]), np.array([1, 1, 0, 0, 0])
    lo, hi = bootstrap_ci(roc_auc, [scores, labels], n_boot=200, strata=labels)
    assert lo == hi == 1.0  # katmanlı örnekleme: hiçbir yinelemede sınıf kaybolmaz


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(3)
    true_logits = rng.normal(0, 2, 5000)
    labels = (rng.random(5000) < 1 / (1 + np.exp(-true_logits))).astype(int)
    assert fit_temperature(true_logits * 4.0, labels) == pytest.approx(4.0, rel=0.15)
