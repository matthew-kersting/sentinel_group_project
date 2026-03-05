"""Shared visual theme for all QBTS slide figures."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL   = "#1BAAB8"   # primary accent (titles, column names, line charts)
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"
GRAY   = "#666666"
BG     = "white"

# Semantic mappings
ACTION_COLORS = {"A": BLUE, "C": ORANGE, "F": GREEN, "T": RED}

# Overview slide donut: Add=light gray, Cancel=dark gray, Trade=near-black, Fill=teal
PIE_COLORS = ["#C0C0C0", "#555555", "#1A1A1A", TEAL]  # A, C, T, F order

# Schema table
HEADER_COLOR = "#1C3040"           # dark header background
ROW_COLORS   = ["#EEF1FA", "#F8F9FE"]

# ── Font sizes ────────────────────────────────────────────────────────────────
FONT_SUPTITLE = 13
FONT_TITLE    = 11
FONT_AXIS     = 10
FONT_TICK     =  9
FONT_SMALL    =  8
FONT_TINY     =  7.5


# ── Helpers ───────────────────────────────────────────────────────────────────
def style_ax(ax):
    """Remove top/right spines and set standard tick label size."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT_TICK)


def millions_formatter():
    """Return a FuncFormatter that displays values as e.g. '3.2M'."""
    return mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M")


def apply_figure_defaults():
    """Call once per script to set rcParams shared across all figures."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   BG,
        "font.family":      "sans-serif",
        "axes.grid":        False,
    })
