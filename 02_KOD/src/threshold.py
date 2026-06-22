"""Threshold optimization for binary classification by F1 maximization."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold_f1(
    y_true: np.ndarray,
    proba_positive: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return (threshold, F1) maximizing F1 over `grid`.

    Default grid: 91 evenly-spaced thresholds in [0.05, 0.95].
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    y_true = np.asarray(y_true).ravel()
    p = np.asarray(proba_positive).ravel()

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)
