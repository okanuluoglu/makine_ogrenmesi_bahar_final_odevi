"""Outer-loop pipeline orchestrator.

Performs:
  for repeat in range(N_OUTER_REPEATS):
      split train+val (70%) vs test (30%) by patient
      for model in BASE_MODEL_NAMES:
          tune via Optuna on train+val with inner StratifiedGroupKFold
          fit best estimator on full train+val
          group-aware sigmoid calibrate
          F1-tune threshold on grouped CV predictions of train+val
          evaluate on test
      build soft-voting ensemble from calibrated RF + ET + GB
      evaluate ensemble on test
  aggregate metrics, save tables/figures, run statistics
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from tqdm import tqdm

from src import config
from src.calibration import group_aware_calibrate
from src.data_loader import load_dataset
from src.ensemble import build_soft_voting_ensemble
from src.evaluation import (
    METRIC_NAMES,
    compute_all_metrics,
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_feature_stability,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
)
from src.models import build_full_pipeline
from src.optimization import run_optuna_study
from src.statistics import run_friedman, run_wilcoxon_bonferroni
from src.threshold import find_best_threshold_f1


# ----------------- Checkpoint helpers -----------------

CHECKPOINT_FILE = "checkpoint.joblib"


def _checkpointable_best(best: dict) -> dict:
    """Strip non-serializable fitted estimators from best_overall."""
    out = {
        "model": best.get("model"),
        "macro_f1": best.get("macro_f1", -1.0),
        "repeat": best.get("repeat", -1),
        "result": None,
    }
    if best.get("result") is not None:
        r = best["result"]
        out["result"] = {
            "y_test": r.get("y_test"),
            "pred_test": r.get("pred_test"),
            "proba_test": r.get("proba_test"),
            "model": r.get("model"),
            "best_threshold": r.get("best_threshold"),
            "metrics": r.get("metrics"),
        }
    return out


def _save_checkpoint(state: dict, path: Path) -> None:
    """Atomically write checkpoint via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(state, tmp, compress=3)
    os.replace(tmp, path)


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Warning: checkpoint at {path} unreadable ({e}); starting fresh")
        return None


def _extract_mrmr_originals(
    calibrated_estimator,
    feature_names: list[str],
) -> list[str]:
    """Walk GroupAwareSigmoidCalibrator.estimator_ to recover MRMR-selected
    feature names in the original feature space.

    Returns an empty list if introspection fails.
    """
    try:
        pipe = calibrated_estimator.estimator_
        pre = pipe.named_steps["preprocess"]
        mrmr = pipe.named_steps["mrmr"]
        var_kept_mask = pre.named_steps["variance"].get_support()
        var_kept = np.where(var_kept_mask)[0]
        corr_kept = pre.named_steps["correlation"].kept_indices_
        after_corr = var_kept[corr_kept]
        mrmr_idx = mrmr.get_support_indices()
        final_original_idx = after_corr[mrmr_idx]
        return [feature_names[i] for i in final_original_idx]
    except Exception:
        return []


def _train_and_evaluate_model(
    model_name: str,
    X_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    n_trials: int,
    n_inner_folds: int,
    cache_dir: str | None,
) -> dict:
    """Tune, calibrate, threshold-optimize, and evaluate one model on one repeat.

    Uses a 70/10/20 three-way split:
      - X_tr (70%): used for Optuna inner-CV hyperparameter search + final fit
      - X_val (10%): held-out validation, used only for threshold optimization
      - X_te (20%): final independent test
    """
    opt = run_optuna_study(
        model_name=model_name,
        X=X_tr, y=y_tr, groups=g_tr,
        n_trials=n_trials,
        n_inner_folds=n_inner_folds,
        cache_dir=cache_dir,
    )
    best_params = opt["params"]

    calibrated = group_aware_calibrate(
        build_full_pipeline(model_name, best_params, cache_dir=cache_dir),
        X_tr, y_tr, g_tr, n_folds=n_inner_folds,
    )

    # Threshold optimization on the held-out validation set (10%)
    proba_val = calibrated.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_threshold_f1(y_val, proba_val)

    # Final test evaluation
    proba_te = calibrated.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= best_t).astype(int)
    metrics = compute_all_metrics(y_te, pred_te, proba_te)

    return {
        "model": model_name,
        "best_params": best_params,
        "best_threshold": best_t,
        "calibrated_estimator": calibrated,
        "proba_val": proba_val,
        "proba_test": proba_te,
        "pred_test": pred_te,
        "y_test": y_te,
        "metrics": metrics,
    }


def run_full_pipeline(
    n_outer_repeats: int = config.N_OUTER_REPEATS,
    n_trials: int = config.N_TRIALS,
    n_inner_folds: int = config.N_INNER_FOLDS,
    save_artifacts: bool = True,
    resume: bool = False,
) -> dict:
    """Top-level orchestrator. Returns dict of aggregated results.

    Checkpointing: after every completed outer repeat, state is written to
    `results/checkpoint.joblib` atomically. When `resume=True`, that file is
    loaded and already-completed repeats are skipped. Splits are deterministic
    (GroupShuffleSplit with fixed seed), so resuming yields the same per-repeat
    splits as the original run.
    """
    config.ensure_dirs()
    bundle = load_dataset()
    feature_names = bundle.feature_names

    checkpoint_path = config.RESULTS_DIR / CHECKPOINT_FILE

    # Initialize state, possibly from checkpoint
    completed_repeats = 0
    accumulated_seconds = 0.0
    rows: list[dict] = []
    per_model_proba: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    feature_selection_counts: Counter = Counter()
    best_overall = {"model": None, "macro_f1": -1.0, "repeat": -1, "result": None}

    if resume:
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt is not None:
            completed_repeats = int(ckpt.get("completed_repeats", 0))
            accumulated_seconds = float(ckpt.get("accumulated_seconds", 0.0))
            rows = list(ckpt.get("rows", []))
            saved_proba = ckpt.get("per_model_proba", {})
            for k, v in saved_proba.items():
                per_model_proba[k] = list(v)
            feature_selection_counts = Counter(ckpt.get("feature_selection_counts", {}))
            best_overall = dict(ckpt.get("best_overall", best_overall))
            print(
                f"[resume] Loaded checkpoint: {completed_repeats} repeats done, "
                f"{accumulated_seconds/60:.1f} min already spent."
            )

    outer = GroupShuffleSplit(
        n_splits=n_outer_repeats,
        test_size=config.TEST_SIZE,  # 0.20 - 20% test
        random_state=config.SEED,
    )
    # Inner split ratio: val / (train+val) = 0.10 / 0.80 = 0.125
    inner_val_frac = config.VAL_SIZE / (1.0 - config.TEST_SIZE)

    session_start = time.time()
    cache_dir = str(config.CACHE_DIR)

    for repeat_idx, (tv_idx, te_idx) in enumerate(
        tqdm(
            outer.split(bundle.X, bundle.y, groups=bundle.patient_id),
            total=n_outer_repeats, desc="Outer repeats", initial=completed_repeats,
        )
    ):
        if repeat_idx < completed_repeats:
            continue

        # Second split: hold out 10% as validation from the 80% train+val
        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=inner_val_frac,
            random_state=config.SEED + repeat_idx,
        )
        rel_tr_idx, rel_val_idx = next(
            val_splitter.split(
                bundle.X[tv_idx], bundle.y[tv_idx], groups=bundle.patient_id[tv_idx]
            )
        )
        tr_idx = tv_idx[rel_tr_idx]
        val_idx = tv_idx[rel_val_idx]

        X_tr, X_val, X_te = bundle.X[tr_idx], bundle.X[val_idx], bundle.X[te_idx]
        y_tr, y_val, y_te = bundle.y[tr_idx], bundle.y[val_idx], bundle.y[te_idx]
        g_tr = bundle.patient_id[tr_idx]
        g_val = bundle.patient_id[val_idx]
        g_te = bundle.patient_id[te_idx]

        # Integrity assertions: no patient overlap across the three sets
        s_tr, s_val, s_te = set(g_tr), set(g_val), set(g_te)
        assert s_tr.isdisjoint(s_val), \
            f"Patient leakage between train and val on repeat {repeat_idx}"
        assert s_tr.isdisjoint(s_te), \
            f"Patient leakage between train and test on repeat {repeat_idx}"
        assert s_val.isdisjoint(s_te), \
            f"Patient leakage between val and test on repeat {repeat_idx}"

        repeat_results: dict[str, dict] = {}
        for model_name in tqdm(
            config.BASE_MODEL_NAMES, leave=False, desc=f"Repeat {repeat_idx + 1}",
        ):
            res = _train_and_evaluate_model(
                model_name=model_name,
                X_tr=X_tr, y_tr=y_tr, g_tr=g_tr,
                X_val=X_val, y_val=y_val,
                X_te=X_te, y_te=y_te,
                n_trials=n_trials,
                n_inner_folds=n_inner_folds,
                cache_dir=cache_dir,
            )
            repeat_results[model_name] = res
            row = {"model": model_name, "repeat": repeat_idx, **res["metrics"]}
            rows.append(row)
            per_model_proba[model_name].append((y_te, res["proba_test"]))

            if res["metrics"]["macro_f1"] > best_overall["macro_f1"]:
                best_overall.update(
                    model=model_name,
                    macro_f1=res["metrics"]["macro_f1"],
                    repeat=repeat_idx,
                    result=res,
                )

            for feat in _extract_mrmr_originals(res["calibrated_estimator"], feature_names):
                feature_selection_counts[feat] += 1

        # Ensemble: average calibrated probabilities of RF + ET + GB
        members = {
            m.lower(): repeat_results[m]["calibrated_estimator"]
            for m in config.ENSEMBLE_MEMBERS
        }
        proba_te_ens = np.mean(
            [m.predict_proba(X_te)[:, 1] for m in members.values()], axis=0,
        )
        proba_val_ens = np.mean(
            [m.predict_proba(X_val)[:, 1] for m in members.values()], axis=0,
        )
        best_t_ens, _ = find_best_threshold_f1(y_val, proba_val_ens)
        pred_te_ens = (proba_te_ens >= best_t_ens).astype(int)
        ens_metrics = compute_all_metrics(y_te, pred_te_ens, proba_te_ens)

        rows.append({"model": "Ensemble", "repeat": repeat_idx, **ens_metrics})
        per_model_proba["Ensemble"].append((y_te, proba_te_ens))

        if ens_metrics["macro_f1"] > best_overall["macro_f1"]:
            best_overall.update(
                model="Ensemble", macro_f1=ens_metrics["macro_f1"],
                repeat=repeat_idx,
                result={
                    "proba_test": proba_te_ens,
                    "pred_test": pred_te_ens,
                    "y_test": y_te,
                    "calibrated_estimator": None,
                    "members": members,
                    "model": "Ensemble",
                    "best_threshold": best_t_ens,
                    "metrics": ens_metrics,
                },
            )

        # ----- Save checkpoint after this outer repeat completes -----
        session_elapsed = time.time() - session_start
        checkpoint_state = {
            "completed_repeats": repeat_idx + 1,
            "accumulated_seconds": accumulated_seconds + session_elapsed,
            "rows": rows,
            "per_model_proba": dict(per_model_proba),
            "feature_selection_counts": dict(feature_selection_counts),
            "best_overall": _checkpointable_best(best_overall),
            "n_outer_repeats": n_outer_repeats,
            "n_trials": n_trials,
            "n_inner_folds": n_inner_folds,
        }
        _save_checkpoint(checkpoint_state, checkpoint_path)

    total_time = accumulated_seconds + (time.time() - session_start)

    metric_df = pd.DataFrame(rows)
    summary = metric_df.groupby("model")[METRIC_NAMES].agg(["mean", "std"])

    wide = metric_df.pivot(index="repeat", columns="model", values="macro_f1")
    friedman_stat, friedman_p = run_friedman(wide)
    wilcoxon_df = run_wilcoxon_bonferroni(wide)

    if save_artifacts:
        metric_df.to_csv(config.TABLES_DIR / "metrics_per_repeat.csv", index=False)
        summary.to_csv(config.TABLES_DIR / "metrics_summary.csv")
        wilcoxon_df.to_csv(config.TABLES_DIR / "wilcoxon_bonferroni.csv", index=False)
        with open(config.TABLES_DIR / "friedman.json", "w", encoding="utf-8") as f:
            json.dump({"statistic": friedman_stat, "p_value": friedman_p}, f, indent=2)

        plot_roc_curves(per_model_proba, config.FIGURES_DIR / "roc_curves.png")
        plot_pr_curves(per_model_proba, config.FIGURES_DIR / "pr_curves.png")
        plot_calibration_curves(
            per_model_proba, config.FIGURES_DIR / "calibration_curves.png",
        )
        plot_model_comparison(
            metric_df, "macro_f1",
            config.FIGURES_DIR / "model_comparison_macro_f1.png",
        )
        plot_model_comparison(
            metric_df, "roc_auc",
            config.FIGURES_DIR / "model_comparison_roc_auc.png",
        )
        plot_feature_stability(
            feature_selection_counts, n_outer_repeats, top_n=20,
            save_path=config.FIGURES_DIR / "feature_stability.png",
        )

        plot_confusion_matrix(
            best_overall["result"]["y_test"],
            best_overall["result"]["pred_test"],
            config.FIGURES_DIR / "confusion_matrix.png",
            title=(
                f"Confusion Matrix — {best_overall['model']} "
                f"(repeat {best_overall['repeat'] + 1})"
            ),
        )

        # Feature importance bar chart (use RF from best repeat if available)
        try:
            best_rf_repeat = best_overall["repeat"]
            # rebuild a representative repeat's RF importances if accessible
            # fallback to MRMR selection frequency as proxy importance
            if feature_selection_counts:
                from src.evaluation import plot_feature_importance
                names = list(feature_selection_counts.keys())
                vals = np.array([feature_selection_counts[n] for n in names], dtype=float)
                plot_feature_importance(
                    vals, names, top_n=min(20, len(names)),
                    save_path=config.FIGURES_DIR / "feature_importance.png",
                    title="Feature Importance Proxy (MRMR Selection Count)",
                )
        except Exception as e:  # pragma: no cover
            print(f"Warning: feature importance plot skipped ({e})")

        manifest = {
            "seed": config.SEED,
            "n_outer_repeats": n_outer_repeats,
            "n_inner_folds": n_inner_folds,
            "n_trials": n_trials,
            "total_seconds": total_time,
            "best_overall": {
                "model": best_overall["model"],
                "macro_f1": best_overall["macro_f1"],
                "repeat": best_overall["repeat"],
            },
            "friedman": {"statistic": friedman_stat, "p_value": friedman_p},
        }
        with open(config.RESULTS_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return {
        "metric_df": metric_df,
        "summary": summary,
        "wilcoxon": wilcoxon_df,
        "friedman": (friedman_stat, friedman_p),
        "per_model_proba": per_model_proba,
        "feature_selection_counts": feature_selection_counts,
        "best_overall": best_overall,
        "total_seconds": total_time,
        "feature_names": feature_names,
        "bundle": bundle,
    }
