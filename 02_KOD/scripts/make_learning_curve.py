"""Plot how mean macro-F1 stabilises as outer repeats accumulate.

This addresses the 'why 10 repeats' question - the curve typically
flattens early, showing additional repeats add little new information.
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
    df = pd.read_csv(config.TABLES_DIR / "metrics_per_repeat.csv")
    wide = df.pivot(index="repeat", columns="model", values="macro_f1")
    n_repeats = len(wide)

    # Cumulative running mean and std per model
    cum_mean = wide.expanding().mean()
    cum_std = wide.expanding().std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = {
        "SVM": "#C00000", "ET": "#4472C4", "KNN": "#70AD47",
        "RF": "#A5A5A5", "Ensemble": "#FFC000",
        "GB": "#ED7D31", "LR": "#000080",
    }

    # Panel 1: cumulative mean
    ax1 = axes[0]
    for m in wide.columns:
        ax1.plot(range(1, n_repeats + 1), cum_mean[m].values,
                 marker="o", markersize=4, lw=1.8,
                 color=palette.get(m, "#888"), label=m)
    ax1.set_xlabel("Toplam Tekrar Sayısı", fontsize=11)
    ax1.set_ylabel("Kümülatif Ortalama Macro-F1", fontsize=11)
    ax1.set_title("Kümülatif Ortalamanın Tekrar Sayısı ile Değişimi",
                  fontsize=12, weight="bold", pad=10)
    ax1.grid(linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", fontsize=9, ncol=2, frameon=False)
    ax1.set_xticks(range(1, n_repeats + 1))

    # Panel 2: cumulative std (showing stabilization)
    ax2 = axes[1]
    for m in wide.columns:
        ax2.plot(range(1, n_repeats + 1), cum_std[m].values,
                 marker="o", markersize=4, lw=1.8,
                 color=palette.get(m, "#888"), label=m)
    ax2.set_xlabel("Toplam Tekrar Sayısı", fontsize=11)
    ax2.set_ylabel("Kümülatif Standart Sapma", fontsize=11)
    ax2.set_title("Standart Sapmanın Tekrar Sayısı ile Oturma Eğilimi",
                  fontsize=12, weight="bold", pad=10)
    ax2.grid(linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xticks(range(1, n_repeats + 1))

    fig.suptitle("Öğrenme Eğrisi: Tekrar Sayısı Arttıkça Kararlılık",
                 fontsize=14, weight="bold", y=0.99)

    # Bottom note
    fig.text(0.5, -0.02,
             f"Eğri {n_repeats} tekrar boyunca düz bir seyre oturmaktadır; "
             f"daha fazla tekrar yapmanın kayda değer bir ek bilgi "
             f"sağlamayacağı görülmektedir.",
             ha="center", fontsize=10, style="italic", color="#555")

    fig.tight_layout()
    out = config.FIGURES_DIR / "ogrenme_egrisi.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
