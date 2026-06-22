from src.optimization import run_optuna_study


def test_run_optuna_study_returns_params(synthetic_xy):
    X, y, groups, _ = synthetic_xy
    result = run_optuna_study(
        model_name="LR",
        X=X,
        y=y,
        groups=groups,
        n_trials=3,
        n_inner_folds=2,
        cache_dir=None,
    )
    assert "params" in result
    assert "score" in result
    assert "study" in result
    assert isinstance(result["score"], float)
    assert "mrmr_k" in result["params"]
