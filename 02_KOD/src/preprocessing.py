"""Preprocessing transformers and Pipeline factories.

Provides:
- `CorrelationFilter`: drops highly correlated features (|r| > threshold).
- `build_preprocessor`: returns a Pipeline of (impute → variance → corr-filter → scale).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src import config


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop columns with |Pearson correlation| > threshold.

    For each correlated pair, keep the column that appears first (lower index).
    Constant columns produce nan correlations and are retained (no pair penalty).
    """

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        if n_features <= 1:
            self.kept_indices_ = np.arange(n_features)
            return self

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        abs_corr = np.abs(corr)

        drop: set[int] = set()
        for i in range(n_features):
            if i in drop:
                continue
            for j in range(i + 1, n_features):
                if j in drop:
                    continue
                if abs_corr[i, j] > self.threshold:
                    drop.add(j)

        self.kept_indices_ = np.array(
            [i for i in range(n_features) if i not in drop], dtype=int
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, self.kept_indices_]

    def get_support(self) -> np.ndarray:
        return self.kept_indices_


def build_preprocessor() -> Pipeline:
    """Median impute → low-variance drop → correlation filter → RobustScaler."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)),
            ("correlation", CorrelationFilter(threshold=config.CORRELATION_THRESHOLD)),
            ("scaler", RobustScaler()),
        ]
    )
