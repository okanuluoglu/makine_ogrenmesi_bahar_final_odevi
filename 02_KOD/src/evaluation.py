"""Evaluation metrics and plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES: list[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "brier",
]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba_positive: np.ndarray,
) -> dict[str, float]:
    """Compute all 9 mandatory metrics. Returns dict keyed by metric name."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba_positive)),
        "pr_auc": float(average_precision_score(y_true, proba_positive)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, proba_positive)),
    }


# ---------- Plots ----------


def plot_roc_curves(
    per_model_proba: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    save_path: Path,
) -> None:
    """Aggregate ROC curve plot, one curve per model (mean over repeats)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_fpr = np.linspace(0, 1, 200)
    for name, runs in per_model_proba.items():
        tprs, aucs = [], []
        for y_true, p in runs:
            fpr, tpr, _ = roc_curve(y_true, p)
            interp = np.interp(mean_fpr, fpr, tpr)
            interp[0] = 0.0
            tprs.append(interp)
            aucs.append(roc_auc_score(y_true, p))
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        ax.plot(
            mean_fpr, mean_tpr, lw=2,
            label=f"{name} (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Mean over Outer Repeats")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pr_curves(
    per_model_proba: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    save_path: Path,
) -> None:
    """Aggregate Precision-Recall curve plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_recall = np.linspace(0, 1, 200)
    for name, runs in per_model_proba.items():
        precs, aps = [], []
        for y_true, p in runs:
            prec, rec, _ = precision_recall_curve(y_true, p)
            order = np.argsort(rec)
            interp = np.interp(mean_recall, rec[order], prec[order])
            precs.append(interp)
            aps.append(average_precision_score(y_true, p))
        mean_prec = np.mean(precs, axis=0)
        ax.plot(
            mean_recall, mean_prec, lw=2,
            label=f"{name} (AP={np.mean(aps):.3f}±{np.std(aps):.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Mean over Outer Repeats")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Papilödem"],
        yticklabels=["Normal", "Papilödem"], ax=ax, cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_calibration_curves(
    per_model_proba: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    save_path: Path,
    n_bins: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, runs in per_model_proba.items():
        y_all = np.concatenate([r[0] for r in runs])
        p_all = np.concatenate([r[1] for r in runs])
        try:
            prob_true, prob_pred = calibration_curve(
                y_all, p_all, n_bins=n_bins, strategy="quantile",
            )
        except ValueError:
            continue
        ax.plot(prob_pred, prob_true, marker="o", lw=1.5, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Positive Rate")
    ax.set_title("Calibration Curves")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_model_comparison(
    metric_df: pd.DataFrame,
    metric: str,
    save_path: Path,
) -> None:
    """Bar chart of mean ± std per model for the given metric."""
    summary = metric_df.groupby("model")[metric].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        summary["model"], summary["mean"], yerr=summary["std"],
        capsize=4, color=sns.color_palette("viridis", len(summary)),
    )
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    n_repeats = metric_df["repeat"].nunique()
    ax.set_title(f"Model Comparison — {metric} (mean ± std over {n_repeats} repeats)")
    ymin = max(0.0, summary["mean"].min() - 0.1)
    ymax = min(1.0, summary["mean"].max() + 0.1)
    ax.set_ylim(ymin, ymax)
    for i, (m, s) in enumerate(zip(summary["mean"], summary["std"])):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: list[str],
    top_n: int,
    save_path: Path,
    title: str = "Top Feature Importances",
) -> None:
    order = np.argsort(importance)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(
        range(top_n)[::-1], importance[order],
        color=sns.color_palette("crest", top_n),
    )
    ax.set_yticks(range(top_n)[::-1])
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_feature_stability(
    selection_frequency: dict[str, int],
    n_repeats: int,
    top_n: int,
    save_path: Path,
) -> None:
    """Bonus: how often each feature was selected by MRMR across outer repeats."""
    items = sorted(selection_frequency.items(), key=lambda kv: -kv[1])[:top_n]
    names = [k for k, _ in items]
    freqs = [v / max(n_repeats, 1) for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(
        range(len(items))[::-1], freqs,
        color=sns.color_palette("flare", max(len(items), 1)),
    )
    ax.set_yticks(range(len(items))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel(f"Selection Frequency (out of {n_repeats} repeats)")
    ax.set_xlim(0, 1.05)
    ax.set_title("MRMR Feature Stability Across Outer Repeats")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
