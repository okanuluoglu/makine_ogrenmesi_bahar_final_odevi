import numpy as np
from sklearn.linear_model import LogisticRegression

from src.calibration import group_aware_calibrate


def test_calibrate_returns_fit_estimator(synthetic_xy):
    X, y, groups, _ = synthetic_xy
    estimator = LogisticRegression(max_iter=1000)
    calibrated = group_aware_calibrate(estimator, X, y, groups, n_folds=2)
    proba = calibrated.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_calibrate_smoke_on_synthetic_pipeline(synthetic_xy):
    from src.models import build_full_pipeline

    X, y, groups, _ = synthetic_xy
    params = {"mrmr_k": 5, "C": 1.0, "penalty": "l2"}
    pipe = build_full_pipeline("LR", params, cache_dir=None)
    calibrated = group_aware_calibrate(pipe, X, y, groups, n_folds=2)
    preds = calibrated.predict(X)
    assert preds.shape == y.shape
