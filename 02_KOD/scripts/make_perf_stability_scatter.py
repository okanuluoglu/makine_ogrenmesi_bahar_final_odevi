"""Scatter of mean macro-F1 vs std macro-F1 per model.

Shows the 'accuracy vs stability' tradeoff. Used in TARTIŞMA 5.2.
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
    summary = pd.read_csv(
        config.TABLES_DIR / "metrics_summary.csv", header=[0, 1], index_col=0
    )
    mean = summary[("macro_f1", "mean")]
    std = summary[("macro_f1", "std")]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    palette = {
        "SVM": "#C00000", "ET": "#4472C4", "KNN": "#70AD47",
        "RF": "#A5A5A5", "Ensemble": "#FFC000",
        "GB": "#ED7D31", "LR": "#000080",
    }

    for m in mean.index:
        ax.scatter(mean[m], std[m], s=240,
                   color=palette.get(m, "#888"),
                   edgecolor="black", linewidth=1.2, zorder=3)
        # offset label
        offset_y = 0.006 if m not in ("GB",) else -0.012
        ax.annotate(m, (mean[m], std[m]),
                    xytext=(8, offset_y * 1000), textcoords="offset points",
                    fontsize=11, weight="bold")

    # "İdeal bölge" highlight
    ax.axhspan(0, 0.075, xmin=0.6, xmax=1.0, color="#e8f5e9",
               alpha=0.5, zorder=1)
    ax.text(0.94, 0.073, "İdeal bölge:\nyüksek doğruluk +\ndüşük varyans",
            ha="right", va="top", fontsize=10, color="#2e7d32",
            style="italic")

    ax.set_xlabel("Ortalama Macro-F1 (yüksek = iyi)", fontsize=12)
    ax.set_ylabel("Macro-F1 Standart Sapma (düşük = kararlı)", fontsize=12)
    ax.set_title("Modellerin Doğruluk-Kararlılık Profili",
                 fontsize=13, weight="bold", pad=12)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.74, 0.96)
    ax.set_ylim(0, 0.20)

    ax.text(0.5, -0.18,
            "Sağ alt köşeye yakın modeller hem yüksek skor üretir hem de "
            "tekrarlar arasında daha kararlıdır.",
            transform=ax.transAxes, ha="center", fontsize=10,
            style="italic", color="#555")

    fig.tight_layout()
    out = config.FIGURES_DIR / "dogruluk_kararlilik.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
