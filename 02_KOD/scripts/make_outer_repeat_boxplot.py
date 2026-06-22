"""Boxplot of macro-F1 distribution across the 20 outer repeats per model.

Goes into TARTIŞMA as a visual support for the discussion of variance
between repeats and model robustness.
"""
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
    df = pd.read_csv(config.TABLES_DIR / "metrics_per_repeat.csv")
    # Pivot wide: rows=repeat, cols=model, values=macro_f1
    wide = df.pivot(index="repeat", columns="model", values="macro_f1")

    # Sort columns by mean (best to worst, left to right)
    order = wide.mean(axis=0).sort_values(ascending=False).index.tolist()
    wide = wide[order]

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = ["#4472C4", "#5B9BD5", "#70AD47", "#A5A5A5",
               "#FFC000", "#ED7D31", "#C00000"]

    bp = ax.boxplot(
        [wide[m].values for m in order],
        labels=order,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="o", markerfacecolor="white",
                       markeredgecolor="black", markersize=7),
        medianprops=dict(color="#222", linewidth=1.5),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, color in zip(bp["boxes"], palette[:len(order)]):
        patch.set_facecolor(color + "33")
        patch.set_edgecolor(color)

    # Overlay raw datapoints with jitter
    import numpy as np
    rng = np.random.default_rng(7)
    for i, m in enumerate(order, start=1):
        vals = wide[m].values
        x = rng.normal(i, 0.05, size=len(vals))
        ax.scatter(x, vals, alpha=0.55, s=30,
                   color=palette[(i - 1) % len(palette)],
                   edgecolors="none", zorder=3)

    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title("Modellerin 20 Dış Tekrarda Macro-F1 Dağılımı",
                 fontsize=13, weight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation under chart
    ax.text(0.5, -0.12,
            "Kutu = IQR, çizgi = medyan, beyaz nokta = ortalama, "
            "renkli noktalar = tek tek 20 tekrar.",
            transform=ax.transAxes, ha="center", fontsize=9.5,
            style="italic", color="#555")

    fig.tight_layout()
    out = config.FIGURES_DIR / "model_varyans_boxplot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
