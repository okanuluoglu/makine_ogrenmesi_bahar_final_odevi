"""Generate a clean 4+3 flowchart of the analysis pipeline.

Top row: 4 boxes (steps 1-4) flowing left to right.
Bottom row: 3 boxes (steps 5-7) flowing left to right.
A curved return arrow connects step 4 (top right) to step 5 (bottom left).
Box sizes are large enough that all text is comfortably readable.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402


STEPS = [
    ("1\nVeri Yükleme",
     "Normal + Papilödem\nCSV birleştirme\nHasta ID prefix\nIntegrity check",
     "#4472C4"),
    ("2\nHasta Bazlı\nBölme",
     "GroupShuffleSplit\n%70 train + val\n%30 test\n10 dış tekrar",
     "#5B9BD5"),
    ("3\nÖn İşleme",
     "Median imputation\nDüşük varyans elemesi\nKorelasyon filtresi\nRobustScaler",
     "#70AD47"),
    ("4\nÖzellik\nSeçimi",
     "MRMR algoritması\nBilgilendirici\nve birbirine\nbenzemeyen özellikler",
     "#A5A5A5"),
    ("5\nHiperparametre\nArama",
     "Optuna TPE\n50 deneme\nStratifiedGroupKFold\n6 model",
     "#FFC000"),
    ("6\nKalibrasyon\n+ Ensemble",
     "Sigmoid Platt\nscaling\nSoft voting\nRF + ET + GB",
     "#ED7D31"),
    ("7\nTest\nDeğerlendirmesi",
     "9 metrik\nFriedman + Wilcoxon\nSHAP\nyorumlanabilirlik",
     "#C00000"),
]


def main() -> Path:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Box dimensions
    box_w, box_h = 19, 26

    # Top row: 4 boxes, centers evenly spaced.
    # Total content width = 4*19 + 3*gap_top, plus side margins ~ 4 each.
    n_top = 4
    gap_top = (100 - 8 - n_top * box_w) / (n_top - 1)
    top_positions = [
        (4 + box_w / 2 + i * (box_w + gap_top), 72)
        for i in range(n_top)
    ]

    # Bottom row: 3 boxes, same gap rules
    n_bot = 3
    gap_bot = (100 - 8 - n_bot * box_w) / (n_bot - 1)
    bot_positions = [
        (4 + box_w / 2 + i * (box_w + gap_bot), 25)
        for i in range(n_bot)
    ]

    positions = top_positions + bot_positions

    for (title, body, color), (x, y) in zip(STEPS, positions):
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            linewidth=2.0,
            edgecolor=color,
            facecolor=color + "22",
        )
        ax.add_patch(rect)
        # Title at upper part of box
        ax.text(x, y + 7, title, ha="center", va="center",
                fontsize=12, weight="bold", color=color,
                linespacing=1.25)
        # Body in lower part
        ax.text(x, y - 5, body, ha="center", va="center",
                fontsize=10, color="#222", linespacing=1.45)

    # Straight arrows on top row (1->2, 2->3, 3->4)
    for i in range(n_top - 1):
        x1, y1 = top_positions[i]
        x2, y2 = top_positions[i + 1]
        xs = x1 + box_w / 2 + 0.5
        xe = x2 - box_w / 2 - 0.5
        ax.annotate(
            "", xy=(xe, y1), xytext=(xs, y1),
            arrowprops=dict(
                arrowstyle="-|>,head_length=1.0,head_width=0.6",
                lw=2.5, color="#555",
            ),
        )

    # Straight arrows on bottom row (5->6, 6->7)
    for i in range(n_bot - 1):
        x1, y1 = bot_positions[i]
        x2, y2 = bot_positions[i + 1]
        xs = x1 + box_w / 2 + 0.5
        xe = x2 - box_w / 2 - 0.5
        ax.annotate(
            "", xy=(xe, y1), xytext=(xs, y1),
            arrowprops=dict(
                arrowstyle="-|>,head_length=1.0,head_width=0.6",
                lw=2.5, color="#555",
            ),
        )

    # Curved return arrow: box 4 (top right) -> box 5 (bottom left)
    x4, y4 = top_positions[-1]
    x5, y5 = bot_positions[0]
    ax.annotate(
        "",
        xy=(x5, y5 + box_h / 2 + 0.5),
        xytext=(x4, y4 - box_h / 2 - 0.5),
        arrowprops=dict(
            arrowstyle="-|>,head_length=1.0,head_width=0.6",
            lw=2.5, color="#555",
            connectionstyle="arc3,rad=-0.35",
        ),
    )

    fig.suptitle("Çalışmanın Yöntem Akışı", fontsize=17, weight="bold",
                 y=0.96)
    fig.tight_layout()

    out = config.FIGURES_DIR / "yontem_akisi.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
