"""Statistical comparison of models over outer-repeat scores."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def run_friedman(scores_wide: pd.DataFrame) -> tuple[float, float]:
    """Friedman test across model columns over rows (outer repeats).

    `scores_wide` has one column per model and one row per outer repeat.
    """
    arrs = [scores_wide[c].values for c in scores_wide.columns]
    stat, p = friedmanchisquare(*arrs)
    return float(stat), float(p)


def run_wilcoxon_bonferroni(
    scores_wide: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank with Bonferroni correction."""
    models = list(scores_wide.columns)
    pairs = list(combinations(models, 2))
    m = len(pairs)
    rows = []
    for a, b in pairs:
        try:
            stat, p = wilcoxon(scores_wide[a], scores_wide[b], zero_method="wilcox")
        except ValueError:
            stat, p = np.nan, 1.0
        p_bonf = min(1.0, p * m)
        rows.append({
            "model_a": a,
            "model_b": b,
            "statistic": float(stat) if not np.isnan(stat) else np.nan,
            "p_value": float(p),
            "p_bonferroni": float(p_bonf),
            "significant": bool(p_bonf < alpha),
        })
    return pd.DataFrame(rows)
