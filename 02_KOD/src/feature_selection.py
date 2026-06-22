"""MRMR (Minimum Redundancy Maximum Relevance) feature selector.

- Relevance: Mutual Information between feature and target.
- Redundancy: |Pearson correlation| between features.
- Greedy selection: at each step, pick feature j maximizing
    relevance[j] - mean(|corr(j, selected)|).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class MRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 50, random_state: int = 42) -> None:
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        n_features = X.shape[1]
        k_eff = min(self.k, n_features)

        relevance = mutual_info_classif(
            X, y, random_state=self.random_state, discrete_features=False
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        abs_corr = np.abs(corr)

        selected: list[int] = []
        remaining = set(range(n_features))

        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.discard(first)

        while len(selected) < k_eff and remaining:
            best_j, best_score = -1, -np.inf
            sel_arr = np.array(selected)
            for j in remaining:
                redundancy = abs_corr[j, sel_arr].mean()
                score = relevance[j] - redundancy
                if score > best_score:
                    best_score = score
                    best_j = j
            selected.append(best_j)
            remaining.discard(best_j)

        self.support_ = np.array(selected, dtype=int)
        self.relevance_ = relevance
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, self.support_]

    def get_support_indices(self) -> np.ndarray:
        return self.support_
