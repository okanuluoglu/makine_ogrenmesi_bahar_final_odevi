import numpy as np

from src.feature_selection import MRMRSelector


def test_mrmr_selects_k_features():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    X_info = (y[:, None] * 1.0 + rng.normal(scale=0.5, size=(n, 5)))
    X_noise = rng.normal(size=(n, 15))
    X = np.hstack([X_info, X_noise])

    sel = MRMRSelector(k=5).fit(X, y)
    sup = sel.get_support_indices()
    assert len(sup) == 5
    overlap = set(sup) & set(range(5))
    assert len(overlap) >= 3


def test_mrmr_k_larger_than_features():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    y = rng.integers(0, 2, size=50)
    sel = MRMRSelector(k=10).fit(X, y)
    out = sel.transform(X)
    assert out.shape[1] == 4


def test_mrmr_transform_shape():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 12))
    y = rng.integers(0, 2, size=60)
    sel = MRMRSelector(k=3).fit(X, y)
    out = sel.transform(X)
    assert out.shape == (60, 3)
