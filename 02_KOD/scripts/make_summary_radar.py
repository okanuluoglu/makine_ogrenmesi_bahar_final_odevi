"""Radar chart comparing top models across all metrics.

Used in the SONUÇ chapter as a summary visualization.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402


def main() -> Path:
    summary = pd.read_csv(
        config.TABLES_DIR / "metrics_summary.csv", header=[0, 1], index_col=0
    )

    # We will plot the higher-is-better metrics. Brier (lower-is-better) is
    # excluded from the radar so all axes share the same direction.
    metric_keys = ["accuracy", "precision", "recall", "f1", "macro_f1",
                   "roc_auc", "pr_auc", "balanced_accuracy"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "Macro-F1",
                     "ROC-AUC", "PR-AUC", "Balanced\nAcc."]

    # Pick top-3 by macro-F1 plus Ensemble for comparison
    macro_means = summary[("macro_f1", "mean")].sort_values(ascending=False)
    top3 = list(macro_means.index[:3])
    if "Ensemble" not in top3:
        top3.append("Ensemble")

    means_df = pd.DataFrame({
        m: [summary.loc[m, (k, "mean")] for k in metric_keys]
        for m in top3
    }, index=metric_keys)

    # Radar angles
    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    palette = ["#C00000", "#4472C4", "#70AD47", "#FFC000"]

    for i, m in enumerate(top3):
        vals = means_df[m].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, lw=2.2, color=palette[i % len(palette)], label=m)
        ax.fill(angles, vals, color=palette[i % len(palette)], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0.65, 0.95)
    ax.set_yticks(np.arange(0.7, 1.0, 0.05))
    ax.set_yticklabels([f"{v:.2f}" for v in np.arange(0.7, 1.0, 0.05)],
                       fontsize=9, color="#555")
    ax.tick_params(pad=10)
    ax.grid(linestyle=":", color="#bbb")
    ax.spines["polar"].set_color("#bbb")

    ax.set_title("En İyi Modellerin Çok-Metrik Profili",
                 fontsize=14, weight="bold", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.05),
              frameon=False, fontsize=11)

    out = config.FIGURES_DIR / "ozet_radar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
