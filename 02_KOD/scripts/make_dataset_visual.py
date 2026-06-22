"""Generate a rich visual for the dataset description section.

Produces a 2-panel figure:
  Left  - donut chart of sample-level class distribution
  Right - grouped bar chart of patient counts and total samples per class
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402


def main() -> Path:
    n_normal_patients = 48
    n_papil_patients = 21
    n_normal_samples = 672
    n_papil_samples = 294

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # ---------- Donut chart ----------
    ax1 = axes[0]
    labels = [f"Normal\n{n_normal_samples} örnek\n(%{100*n_normal_samples/(n_normal_samples+n_papil_samples):.1f})",
              f"Papilödem\n{n_papil_samples} örnek\n(%{100*n_papil_samples/(n_normal_samples+n_papil_samples):.1f})"]
    values = [n_normal_samples, n_papil_samples]
    colors = ["#5b9bd5", "#ed7d31"]
    wedges, _ = ax1.pie(
        values, labels=None, colors=colors, startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    ax1.text(0, 0.07, "966", ha="center", va="center",
             fontsize=26, weight="bold", color="#2c3e50")
    ax1.text(0, -0.12, "toplam örnek", ha="center", va="center",
             fontsize=11, color="#555")
    ax1.set_title("Örnek Düzeyinde Sınıf Dağılımı", fontsize=13, weight="bold",
                  pad=12)
    ax1.legend(wedges, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)

    # ---------- Grouped bar ----------
    ax2 = axes[1]
    categories = ["Hasta sayısı", "Örnek sayısı"]
    normal_vals = [n_normal_patients, n_normal_samples]
    papil_vals = [n_papil_patients, n_papil_samples]
    x = np.arange(len(categories))
    width = 0.36

    b1 = ax2.bar(x - width / 2, normal_vals, width, label="Normal",
                 color="#5b9bd5", edgecolor="#1f4e79", linewidth=0.6)
    b2 = ax2.bar(x + width / 2, papil_vals, width, label="Papilödem",
                 color="#ed7d31", edgecolor="#a04000", linewidth=0.6)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2.0, h + max(normal_vals) * 0.015,
                     f"{int(h):,}".replace(",", "."), ha="center", va="bottom",
                     fontsize=10, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_ylabel("Sayı", fontsize=11)
    ax2.set_title("Hasta ve Örnek Sayıları", fontsize=13, weight="bold", pad=12)
    ax2.legend(frameon=False, fontsize=10, loc="upper left")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(0, max(normal_vals) * 1.18)

    # Annotation: 14 örnek per hasta
    ax2.text(0.5, -0.18,
             "Her hasta için 14 örnek (sağ + sol göz, 7 dilim/göz)",
             transform=ax2.transAxes, ha="center", fontsize=10,
             style="italic", color="#555")

    fig.suptitle("Veri Seti Özeti", fontsize=15, weight="bold", y=1.00)
    fig.tight_layout()

    out = config.FIGURES_DIR / "veri_seti_ozet.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
