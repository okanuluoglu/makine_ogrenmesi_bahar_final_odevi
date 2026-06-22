"""Render Tablo 4.1 (metrics summary) as a styled image for the report."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402


def main() -> Path:
    summary = pd.read_csv(
        config.TABLES_DIR / "metrics_summary.csv", header=[0, 1], index_col=0
    )
    metric_keys = [
        "accuracy", "precision", "recall", "f1", "macro_f1",
        "roc_auc", "pr_auc", "balanced_accuracy", "brier",
    ]
    metric_labels = [
        "Accuracy", "Precision", "Recall", "F1", "Macro-F1",
        "ROC-AUC", "PR-AUC", "Bal. Acc.", "Brier",
    ]

    # Mean and std arrays
    means = pd.DataFrame(index=summary.index)
    stds = pd.DataFrame(index=summary.index)
    for k in metric_keys:
        means[k] = summary[(k, "mean")]
        stds[k] = summary[(k, "std")]

    # Sort models by macro-F1 for nicer reading
    order = means["macro_f1"].sort_values(ascending=False).index
    means = means.loc[order]
    stds = stds.loc[order]

    # Cell text
    cell_text = [[f"{means.iloc[i, j]:.3f}\n±{stds.iloc[i, j]:.3f}"
                  for j in range(len(metric_keys))]
                 for i in range(len(means))]

    # Best / worst markers
    # For Brier, lower is better. For everything else, higher is better.
    higher_better = [True, True, True, True, True, True, True, True, False]
    cell_colors = [["#ffffff"] * len(metric_keys) for _ in range(len(means))]
    for j, hb in enumerate(higher_better):
        col_means = means.iloc[:, j]
        best_idx = col_means.idxmax() if hb else col_means.idxmin()
        worst_idx = col_means.idxmin() if hb else col_means.idxmax()
        bi = list(means.index).index(best_idx)
        wi = list(means.index).index(worst_idx)
        cell_colors[bi][j] = "#d5edda"   # soft green for best
        cell_colors[wi][j] = "#fbe2c4"   # soft amber for worst

    # Figure
    fig, ax = plt.subplots(figsize=(13.5, 0.7 + 0.55 * len(means)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=means.index,
        colLabels=metric_labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header row
    for j in range(len(metric_keys)):
        h = table[(0, j)]
        h.set_facecolor("#2c3e50")
        h.set_text_props(color="white", weight="bold")
        h.set_height(0.10)
    # Style model column (row labels)
    for i in range(len(means)):
        rl = table[(i + 1, -1)]
        rl.set_facecolor("#ecf0f1")
        rl.set_text_props(weight="bold")
    # Borders for all cells
    for cell in table.get_celld().values():
        cell.set_edgecolor("#7f8c8d")
        cell.set_linewidth(0.5)

    out_path = config.FIGURES_DIR / "tablo_4_1_metrics.png"
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
