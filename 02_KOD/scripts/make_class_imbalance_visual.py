"""Generate a visual for the class imbalance discussion (Section 2.3)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402


def main() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5),
                             gridspec_kw={"width_ratios": [1, 1.1]})

    # ----- Left panel: imbalance bar -----
    ax1 = axes[0]
    normal = 672
    papil = 294
    total = normal + papil
    bars_y = ["Eğitim sırasındaki örnek dağılımı"]
    ax1.barh(bars_y, [normal], color="#5b9bd5", edgecolor="#1f4e79",
             label=f"Normal (%{100*normal/total:.1f})")
    ax1.barh(bars_y, [papil], left=[normal], color="#ed7d31",
             edgecolor="#a04000",
             label=f"Papilödem (%{100*papil/total:.1f})")
    ax1.set_xlim(0, total)
    ax1.set_xlabel("Örnek sayısı", fontsize=11)
    ax1.set_title("Sınıflar Arası Dengesizlik (~70 / 30)",
                  fontsize=12, weight="bold", pad=10)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2,
               frameon=False, fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0)
    # Annotations on bar
    ax1.text(normal / 2, 0, f"{normal}", ha="center", va="center",
             color="white", fontsize=14, weight="bold")
    ax1.text(normal + papil / 2, 0, f"{papil}", ha="center", va="center",
             color="white", fontsize=14, weight="bold")

    # ----- Right panel: SMOTE breaks patient grouping (conceptual) -----
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    # Patient circles
    patient_color = "#5b9bd5"
    synth_color = "#e74c3c"

    # Two real patients
    p1 = mpatches.Circle((2, 6.5), 0.8, facecolor=patient_color,
                         edgecolor="#1f4e79", linewidth=1.5)
    p2 = mpatches.Circle((5, 6.5), 0.8, facecolor=patient_color,
                         edgecolor="#1f4e79", linewidth=1.5)
    ax2.add_patch(p1)
    ax2.add_patch(p2)
    ax2.text(2, 6.5, "P1", ha="center", va="center", fontsize=11,
             color="white", weight="bold")
    ax2.text(5, 6.5, "P2", ha="center", va="center", fontsize=11,
             color="white", weight="bold")

    # Samples around each patient
    for cx, cy in [(2, 6.5), (5, 6.5)]:
        for angle in [0, 60, 120, 180, 240, 300]:
            sx = cx + 1.3 * np.cos(np.radians(angle))
            sy = cy + 1.3 * np.sin(np.radians(angle))
            d = mpatches.Circle((sx, sy), 0.18, facecolor=patient_color,
                                alpha=0.7, edgecolor="#1f4e79")
            ax2.add_patch(d)

    # Synthetic sample - no patient
    synth = mpatches.Circle((8, 6.5), 0.5, facecolor=synth_color,
                            edgecolor="#7d0d0d", linewidth=1.5)
    ax2.add_patch(synth)
    ax2.text(8, 6.5, "?", ha="center", va="center", fontsize=14,
             color="white", weight="bold")

    ax2.annotate("SMOTE\nsentetik\nörnek",
                 xy=(8, 5.6), xytext=(8, 4.0),
                 ha="center", fontsize=10, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))

    # Title on top
    ax2.text(5, 9.3, "SMOTE neden hasta yapısını bozar?",
             ha="center", fontsize=12, weight="bold")

    # Bottom explanation
    ax2.text(5, 2.0,
             "Gerçek örnekler hangi hastaya ait olduğunu bilir.\n"
             "SMOTE'un ürettiği sentetik örnek hiçbir hastaya bağlı\n"
             "değildir; bu da hasta seviyesinde çapraz doğrulama\n"
             "yapısını sessizce kırar.",
             ha="center", va="top", fontsize=9.5, color="#444",
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef5e7",
                       edgecolor="#e67e22", linewidth=1))

    fig.suptitle("Sınıf Dengesizliği ve SMOTE'un Risk Noktası",
                 fontsize=14, weight="bold", y=1.0)
    fig.tight_layout()

    out = config.FIGURES_DIR / "sinif_dengesizligi.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
