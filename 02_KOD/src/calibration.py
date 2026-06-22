"""Group-aware sigmoid (Platt) calibration.

sklearn 1.6's `CalibratedClassifierCV` has metadata-routing complications
when used with `StratifiedGroupKFold` (groups don't reach the splitter).
This module implements behaviorally-equivalent Platt scaling that explicitly
threads `groups` through `cross_val_predict`, eliminating the leakage risk.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import _SigmoidCalibration  # type: ignore[attr-defined]
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from src import config


class GroupAwareSigmoidCalibrator(BaseEstimator, ClassifierMixin):
    """Sigmoid (Platt) calibration with patient-grouped CV.

    Equivalent to `CalibratedClassifierCV(method='sigmoid', cv=StratifiedGroupKFold(...))`
    in spirit, but explicit about group routing.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        n_folds: int = config.N_INNER_FOLDS,
    ) -> None:
        self.estimator = estimator
        self.n_folds = n_folds

    def fit(self, X, y, groups):
        cv = StratifiedGroupKFold(
            n_splits=self.n_folds, shuffle=True, random_state=config.SEED,
        )
        oof = cross_val_predict(
            clone(self.estimator),
            X, y,
            groups=groups,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        self.sigmoid_ = _SigmoidCalibration().fit(oof, y)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X):
        raw = self.estimator_.predict_proba(X)[:, 1]
        cal = self.sigmoid_.predict(raw)
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def group_aware_calibrate(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = config.N_INNER_FOLDS,
) -> GroupAwareSigmoidCalibrator:
    """Convenience wrapper: returns a fit GroupAwareSigmoidCalibrator."""
    return GroupAwareSigmoidCalibrator(estimator, n_folds=n_folds).fit(X, y, groups)
