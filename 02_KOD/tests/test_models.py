import optuna

from src import config
from src.models import MODEL_REGISTRY, build_full_pipeline, suggest_pipeline_params


def test_registry_has_all_six_models():
    assert set(MODEL_REGISTRY) == set(config.BASE_MODEL_NAMES)


def test_each_builder_returns_estimator(synthetic_xy):
    X, y, _, _ = synthetic_xy
    for name in MODEL_REGISTRY:
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_pipeline_params(trial, name)
        pipe = build_full_pipeline(name, params)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == y.shape


def test_suggest_includes_mrmr_k():
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_pipeline_params(trial, "LR")
    assert "mrmr_k" in params
    assert params["mrmr_k"] in config.MRMR_K_CANDIDATES
