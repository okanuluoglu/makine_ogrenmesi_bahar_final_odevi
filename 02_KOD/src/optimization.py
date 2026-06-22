"""Optuna TPE hyperparameter search with patient-grouped inner CV.

Objective: mean macro-F1 across `n_inner_folds` StratifiedGroupKFold folds.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from src import config
from src.models import build_full_pipeline, suggest_pipeline_params


def _objective_factory(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_inner_folds: int,
    cache_dir: str | None,
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_pipeline_params(trial, model_name)
        pipe = build_full_pipeline(model_name, params, cache_dir=cache_dir)
        cv = StratifiedGroupKFold(
            n_splits=n_inner_folds, shuffle=True, random_state=config.SEED
        )
        # Parallelize across folds (sklearn-side); Optuna stays serial
        # to avoid nested-parallelism oversubscription.
        scores = cross_val_score(
            pipe, X, y, groups=groups, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        return float(np.mean(scores))

    return objective


def run_optuna_study(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = config.N_TRIALS,
    n_inner_folds: int = config.N_INNER_FOLDS,
    cache_dir: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run an Optuna study and return best params + best score + study object."""
    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = _objective_factory(
        model_name, X, y, groups, n_inner_folds, cache_dir
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    return {
        "params": study.best_params,
        "score": float(study.best_value),
        "study": study,
    }
