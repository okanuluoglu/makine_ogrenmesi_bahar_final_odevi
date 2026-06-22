import numpy as np

from src.evaluation import METRIC_NAMES, compute_all_metrics


def test_metric_keys_complete():
    expected = {
        "accuracy", "precision", "recall", "f1", "macro_f1",
        "roc_auc", "pr_auc", "balanced_accuracy", "brier",
    }
    assert set(METRIC_NAMES) == expected


def test_compute_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
    pred = (p >= 0.5).astype(int)
    m = compute_all_metrics(y, pred, p)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_compute_metrics_returns_floats():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=80)
    p = rng.uniform(size=80)
    pred = (p >= 0.5).astype(int)
    m = compute_all_metrics(y, pred, p)
    for v in m.values():
        assert isinstance(v, float)
