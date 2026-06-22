"""SHAP-based explainability for top model (bonus task)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.base import BaseEstimator


def compute_shap_summary(
    estimator: BaseEstimator,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    save_path: Path,
    max_display: int = 15,
) -> np.ndarray:
    """Compute SHAP values and save a summary plot.

    Returns mean |SHAP| per feature for top-N reporting.
    """
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_explain)
    except Exception:
        bg = shap.sample(X_background, min(50, X_background.shape[0]), random_state=0)
        explainer = shap.KernelExplainer(estimator.predict_proba, bg)
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    # In newer shap (>=0.46), TreeExplainer for binary classifier may return
    # a 3D array of shape (n_samples, n_features, n_classes); reduce to class 1.
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(
        shap_values, X_explain, feature_names=feature_names,
        max_display=max_display, show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    return mean_abs
