"""Bonus: SHAP analysis on the best-performing tree model (ET).

Uses a single deterministic patient-level split (same seed as full pipeline)
to fit ExtraTrees with reasonable defaults, then runs SHAP TreeExplainer
on the test set and writes results/figures/shap_summary.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from src import config  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402
from src.explainability import compute_shap_summary  # noqa: E402
from src.models import build_full_pipeline  # noqa: E402


def main() -> int:
    config.ensure_dirs()
    bundle = load_dataset()

    # Use the FIRST deterministic split (same seed/test_size as the full pipeline).
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.SEED,
    )
    tv_idx, te_idx = next(splitter.split(bundle.X, bundle.y, groups=bundle.patient_id))

    X_tv, X_te = bundle.X[tv_idx], bundle.X[te_idx]
    y_tv, y_te = bundle.y[tv_idx], bundle.y[te_idx]
    print(f"Train+val: {X_tv.shape}, Test: {X_te.shape}")

    # ET with reasonable defaults — picks middle of Optuna search ranges
    params = {
        "mrmr_k": 50,
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    }
    pipe = build_full_pipeline("ET", params, cache_dir=None)
    print("Fitting ExtraTrees...")
    pipe.fit(X_tv, y_tv)
    print(f"Fit acc on test = {pipe.score(X_te, y_te):.4f}")

    # Recover the post-MRMR feature space dimensions and names for SHAP
    X_tv_post = pipe[:-1].transform(X_tv)
    X_te_post = pipe[:-1].transform(X_te)
    print(f"Post-pipeline (post-MRMR) shape: X_tv={X_tv_post.shape}, X_te={X_te_post.shape}")

    # Map MRMR indices back to original feature names for legible labels
    pre = pipe.named_steps["preprocess"]
    mrmr = pipe.named_steps["mrmr"]
    var_kept = np.where(pre.named_steps["variance"].get_support())[0]
    corr_kept = pre.named_steps["correlation"].kept_indices_
    after_corr = var_kept[corr_kept]
    mrmr_idx = mrmr.get_support_indices()
    final_orig_idx = after_corr[mrmr_idx]
    feature_names_kept = [bundle.feature_names[i] for i in final_orig_idx]

    out_path = config.FIGURES_DIR / "shap_summary.png"
    print(f"Computing SHAP on {len(X_te_post)} test samples × "
          f"{len(feature_names_kept)} features...")
    mean_abs = compute_shap_summary(
        estimator=pipe.named_steps["model"],
        X_background=X_tv_post,
        X_explain=X_te_post,
        feature_names=feature_names_kept,
        save_path=out_path,
        max_display=15,
    )

    # Save top-10 importance table
    order = np.argsort(mean_abs)[::-1][:10]
    top10 = [(feature_names_kept[i], float(mean_abs[i])) for i in order]
    import json
    (config.TABLES_DIR / "shap_top10.json").write_text(
        json.dumps(top10, indent=2), encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    print(f"Wrote {config.TABLES_DIR / 'shap_top10.json'}")
    print("Top 10 SHAP features:")
    for name, val in top10:
        print(f"  {name}: {val:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
