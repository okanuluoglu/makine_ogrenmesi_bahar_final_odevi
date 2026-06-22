import numpy as np
import pandas as pd

from src.statistics import run_friedman, run_wilcoxon_bonferroni


def test_friedman_returns_stat_and_p():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "M1": rng.uniform(0.7, 0.8, size=20),
        "M2": rng.uniform(0.6, 0.7, size=20),
        "M3": rng.uniform(0.5, 0.6, size=20),
    })
    stat, p = run_friedman(df)
    assert isinstance(stat, float)
    assert isinstance(p, float)
    assert p < 0.05


def test_wilcoxon_bonferroni_returns_dataframe():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "M1": rng.uniform(0.7, 0.8, size=20),
        "M2": rng.uniform(0.6, 0.7, size=20),
        "M3": rng.uniform(0.5, 0.6, size=20),
    })
    res = run_wilcoxon_bonferroni(df, alpha=0.05)
    expected_cols = {"model_a", "model_b", "statistic", "p_value", "p_bonferroni", "significant"}
    assert expected_cols <= set(res.columns)
    assert len(res) == 3
