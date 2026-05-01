"""
Coefficient-story figure for the takeaways slide.

Paired horizontal bar chart of standardized coefficients from
  - M3 Ridge      (predicting fwd_return — direction)
  - M5 Logistic   (predicting active_next — activity)

Same feature block on both axes so the visual story is "this feature
matters more for activity than direction" (or vice-versa).

Coefficients below are illustrative — replace with exact values printed
by the final baseline_models.py run.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from theme import (
    BG,
    TEAL,
    BLUE,
    GRAY,
    FONT_TITLE,
    FONT_AXIS,
    FONT_SMALL,
    style_ax,
    apply_figure_defaults,
)

FEATURES = [
    "trade_ofi",
    "ofi",
    "cancel_ratio",
    "cancel_to_trade_ratio",
    "mean_lifespan_s",
    "n_trades",
]

# Standardized coefficients (illustrative)
RIDGE_COEF = [0.32, 0.40, -0.04, -0.03, 0.02, -0.01]   # M3 Ridge — direction
LOGIT_COEF = [0.85, 0.30, 0.55, 0.40, -0.20, 0.10]      # M5 Logistic — activity

OUT = Path(__file__).parent.parent / "data" / "output" / "coefficient_story.png"


def main() -> None:
    apply_figure_defaults()

    # Sort features by combined magnitude — strongest feature at top
    combined = np.array([abs(r) + abs(l) for r, l in zip(RIDGE_COEF, LOGIT_COEF)])
    order = np.argsort(combined)  # ascending → top of barh is largest
    feats = [FEATURES[i] for i in order]
    ridge = [RIDGE_COEF[i] for i in order]
    logit = [LOGIT_COEF[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    n = len(feats)
    y = np.arange(n)
    h = 0.38

    ax.barh(
        y - h / 2,
        ridge,
        height=h,
        color=BLUE,
        alpha=0.88,
        label="M3 Ridge  ·  direction (fwd return)",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.barh(
        y + h / 2,
        logit,
        height=h,
        color=TEAL,
        alpha=0.88,
        label="M5 Logistic  ·  activity (active_next)",
        edgecolor="white",
        linewidth=0.6,
    )

    ax.axvline(0, color=GRAY, lw=1.0, ls="-")
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=FONT_AXIS)
    ax.set_xlabel("Standardized coefficient (per 1-σ)", fontsize=FONT_AXIS)
    ax.set_title(
        "Standardized Coefficients  ·  Direction vs Activity",
        fontsize=FONT_TITLE,
        fontweight="bold",
        loc="left",
    )
    ax.legend(fontsize=FONT_SMALL, frameon=False, loc="lower right")
    style_ax(ax)

    plt.tight_layout()
    plt.savefig(OUT, dpi=400, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
