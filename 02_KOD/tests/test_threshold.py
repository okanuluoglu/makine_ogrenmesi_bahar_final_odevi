import numpy as np

from src.threshold import find_best_threshold_f1


def test_default_threshold_when_perfectly_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t, score = find_best_threshold_f1(y, p)
    assert score == 1.0
    assert 0.3 < t < 0.7


def test_threshold_within_grid():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    p = rng.uniform(size=100)
    t, _ = find_best_threshold_f1(y, p)
    assert 0.05 <= t <= 0.95
