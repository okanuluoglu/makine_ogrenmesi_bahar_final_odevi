import numpy as np

from src.preprocessing import CorrelationFilter, build_preprocessor


def test_correlation_filter_drops_duplicate_columns():
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.01, size=n)
    x3 = rng.normal(size=n)
    X = np.column_stack([x1, x2, x3])
    f = CorrelationFilter(threshold=0.95).fit(X)
    out = f.transform(X)
    assert out.shape[1] == 2
    assert len(f.kept_indices_) == 2


def test_correlation_filter_keeps_all_when_independent():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5))
    f = CorrelationFilter(threshold=0.95).fit(X)
    out = f.transform(X)
    assert out.shape[1] == 5


def test_correlation_filter_handles_constant_column():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.normal(size=50), np.ones(50), rng.normal(size=50)])
    f = CorrelationFilter(threshold=0.95).fit(X)
    out = f.transform(X)
    assert out.shape[0] == 50


def test_build_preprocessor_pipeline_fits_and_transforms(synthetic_xy):
    X, y, groups, side = synthetic_xy
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] <= X.shape[1]
    assert np.isfinite(Xt).all()
