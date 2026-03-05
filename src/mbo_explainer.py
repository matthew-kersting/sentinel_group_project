"""
Slide figure: MBO order lifecycle illustration.

Shows synthetic order timelines to explain that MBO data captures every
event (Add, Cancel, Execute/Fill) for each order, linked by order ID.
Time flows top-to-bottom. No title or legend — event labels are inline.
"""

import matplotlib.pyplot as plt
import numpy as np

from theme import (
    BG, TEAL, BLUE, ORANGE, GREEN, RED, GRAY,
    FONT_SMALL, FONT_TINY,
    apply_figure_defaults,
)


# ── Synthetic order lifecycle data ────────────────────────────────────────────
# Each order: (column label, [(t, event_type, display_label)])
ORDERS = [
    ("Order 1", [
        (0.8,  "A", "Add"),
        (2.5,  "A", "Modify"),
        (5.2,  "F", "Fill"),
    ]),
    ("Order 2", [
        (1.2,  "A", "Add"),
        (3.8,  "C", "Cancel"),
    ]),
    ("Order 3", [
        (0.4,  "A", "Add"),
        (2.0,  "A", "Modify"),
        (4.6,  "T", "Trade"),
    ]),
    ("Order 4", [
        (1.6,  "A", "Add"),
        (3.1,  "C", "Cancel"),
    ]),
]

EVENT_COLOR = {
    "A": BLUE,
    "C": ORANGE,
    "F": GREEN,
    "T": RED,
}

EVENT_MARKER = {
    "A": "o",
    "C": "X",
    "F": "*",
    "T": "D",
}


def plot(out_path: str = "data/output/mbo_explainer.png") -> None:
    apply_figure_defaults()

    fig, ax = plt.subplots(figsize=(5, 7.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n_orders = len(ORDERS)
    T_MAX    = 6.5
    T_MIN    = 0.0

    # Evenly space orders on the x-axis
    x_positions = np.linspace(1, n_orders, n_orders)

    # Y-axis: time goes downward, so we invert
    ax.set_ylim(T_MAX + 0.4, T_MIN - 0.8)   # inverted
    ax.set_xlim(x_positions[0] - 0.8, x_positions[-1] + 0.8)

    for x, (order_label, events) in zip(x_positions, ORDERS):
        # Vertical timeline
        ax.vlines(x, T_MIN, T_MAX, colors="#DDDDDD", lw=1.5, zorder=1)

        # Order label at the top
        ax.text(x, T_MIN - 0.55, order_label,
                ha="center", va="center", fontsize=FONT_SMALL,
                color="#333333", fontweight="bold")

        # Draw events top-to-bottom
        t_prev = None
        for i, (t, etype, elabel) in enumerate(events):
            color  = EVENT_COLOR[etype]
            marker = EVENT_MARKER[etype]
            ms     = 13 if etype == "F" else 10

            ax.plot(x, t, marker=marker, color=color, markersize=ms,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)

            # Alternate labels left/right
            hoffset = -0.22 if i % 2 == 0 else 0.22
            ha      = "right" if hoffset < 0 else "left"
            ax.text(x + hoffset, t, elabel,
                    ha=ha, va="center", fontsize=FONT_TINY,
                    color=color, fontweight="bold")

            # Arrow from previous event downward
            if t_prev is not None:
                ax.annotate("", xy=(x, t - 0.08), xytext=(x, t_prev + 0.08),
                            arrowprops=dict(
                                arrowstyle="-|>",
                                color="#BBBBBB",
                                lw=1.0,
                            ), zorder=2)
            t_prev = t

    # ── Time arrow on the left ────────────────────────────────────────────────
    left_x = x_positions[0] - 0.65
    ax.annotate("", xy=(left_x, T_MAX + 0.2), xytext=(left_x, T_MIN - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2))
    ax.text(left_x, (T_MIN + T_MAX) / 2, "Time (ns)",
            ha="center", va="center", fontsize=FONT_TINY, color=GRAY,
            rotation=90)

    # ── Axes off ──────────────────────────────────────────────────────────────
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    plot()
