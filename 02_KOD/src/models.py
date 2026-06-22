"""Base model registry and Optuna search spaces.

Each entry maps name → factory that builds an unfit sklearn-compatible estimator
given a hyperparameter dict. A separate `suggest_pipeline_params` function
proposes the dict from an Optuna trial.
"""
from __future__ import annotations

from typing import Any, Callable

import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src import config
from src.feature_selection import MRMRSelector
from src.preprocessing import build_preprocessor


def _build_lr(params: dict[str, Any]) -> BaseEstimator:
    return LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        solver="saga",
        class_weight="balanced",
        max_iter=10000,
        tol=1e-3,
        random_state=config.SEED,
        n_jobs=1,
    )


def _build_svm(params: dict[str, Any]) -> BaseEstimator:
    return SVC(
        C=params["C"],
        gamma=params["gamma"],
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=config.SEED,
    )


def _build_rf(params: dict[str, Any]) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced_subsample",
        random_state=config.SEED,
        n_jobs=1,
    )


def _build_et(params: dict[str, Any]) -> BaseEstimator:
    return ExtraTreesClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced_subsample",
        random_state=config.SEED,
        n_jobs=1,
    )


def _build_gb(params: dict[str, Any]) -> BaseEstimator:
    return GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        random_state=config.SEED,
    )


def _build_knn(params: dict[str, Any]) -> BaseEstimator:
    return KNeighborsClassifier(
        n_neighbors=params["n_neighbors"],
        weights=params["weights"],
        metric=params["metric"],
        n_jobs=1,
    )


MODEL_REGISTRY: dict[str, Callable[[dict[str, Any]], BaseEstimator]] = {
    "LR": _build_lr,
    "SVM": _build_svm,
    "RF": _build_rf,
    "ET": _build_et,
    "GB": _build_gb,
    "KNN": _build_knn,
}


def suggest_pipeline_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Propose hyperparameter dict for the named model from an Optuna trial."""
    params: dict[str, Any] = {
        "mrmr_k": trial.suggest_categorical("mrmr_k", config.MRMR_K_CANDIDATES),
    }
    if model_name == "LR":
        params["C"] = trial.suggest_float("C", 1e-3, 1e2, log=True)
        params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
    elif model_name == "SVM":
        params["C"] = trial.suggest_float("C", 1e-2, 1e2, log=True)
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
    elif model_name in ("RF", "ET"):
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, step=50)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
        params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    elif model_name == "GB":
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, step=50)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
        params["max_depth"] = trial.suggest_int("max_depth", 2, 8)
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    elif model_name == "KNN":
        params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 25)
        params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        params["metric"] = trial.suggest_categorical("metric", ["euclidean", "manhattan"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return params


def build_full_pipeline(
    model_name: str,
    params: dict[str, Any],
    cache_dir: str | None = None,
) -> Pipeline:
    """Compose preprocessor → MRMR → model into a single sklearn Pipeline."""
    preprocessor = build_preprocessor()
    mrmr = MRMRSelector(k=params["mrmr_k"], random_state=config.SEED)
    model = MODEL_REGISTRY[model_name](params)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("mrmr", mrmr),
            ("model", model),
        ],
        memory=cache_dir,
    )
