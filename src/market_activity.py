"""
Slide figure: market activity deep-dive for QBTS NASDAQ ITCH MBO.

Panels
------
A  Daily message volume       — temporal coverage across the month
B  Intraday activity profile  — hourly counts showing market-hours structure
C  Order size distribution    — log-log histogram (heavy-tailed character)
D  Action × Side heatmap      — how event types and sides pair in MBO encoding
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

import polars as pl

from theme import (
    BG, TEAL, HEADER_COLOR, GRAY, RED,
    FONT_SUPTITLE, FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_SMALL, FONT_TINY,
    style_ax, millions_formatter, apply_figure_defaults,
)

SLATE    = "#455A64"   # panel C histogram
MUTED    = "#90A4AE"   # extended-hours bars / None-side bars


def _utc_to_et_label(h: int) -> str:
    et = h - 4  # EDT
    suffix = "AM" if et < 12 else "PM"
    return f"{et % 12 or 12}{suffix}"


def plot(df: pl.DataFrame, out_path: str = "data/output/market_activity.png") -> None:
    apply_figure_defaults()

    # ── Pre-compute ───────────────────────────────────────────────────────────
    date_min = df["ts_event"].min().strftime("%b %d")
    date_max = df["ts_event"].max().strftime("%b %d %Y")
    n_rows   = len(df)

    # A — daily volume
    daily = (
        df.with_columns(pl.col("ts_event").dt.date().alias("date"))
        .group_by("date").len().sort("date")
    )

    # B — hourly volume
    hourly = (
        df.with_columns(pl.col("ts_event").dt.hour().alias("hour"))
        .group_by("hour").len().sort("hour")
    )

    # C — order size distribution
    sizes = (
        df.filter(
            pl.col("action").is_in(["A", "T"]) &
            (pl.col("size") > 0) &
            (pl.col("size") <= 5_000)
        )["size"].to_numpy()
    )

    # D — action × side cross-tab
    cross = (
        df.filter(pl.col("action") != "R")
        .group_by(["action", "side"]).len()
    )
    actions = ["A", "C", "F", "T"]
    sides   = ["B", "A", "N"]
    cross_mat = np.zeros((len(actions), len(sides)), dtype=float)
    for row in cross.iter_rows(named=True):
        r = actions.index(row["action"])
        c = sides.index(row["side"])
        cross_mat[r, c] = row["len"]
    row_sums = cross_mat.sum(axis=1, keepdims=True)
    cross_pct = np.where(row_sums > 0, cross_mat / row_sums * 100, 0)

    # C — histogram bins (pre-computed so CSV and plot share the same bins)
    log_bins = np.logspace(0, np.log10(5_000), 60)
    hist_counts, hist_edges = np.histogram(sizes, bins=log_bins)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    csv_dir = os.path.dirname(out_path)

    # A — daily volume
    daily.rename({"len": "message_count"}).write_csv(
        os.path.join(csv_dir, "daily_volume.csv")
    )

    # B — intraday activity
    (
        hourly
        .rename({"len": "message_count"})
        .with_columns(
            pl.col("hour")
            .map_elements(_utc_to_et_label, return_dtype=pl.String)
            .alias("hour_et"),
            pl.when((pl.col("hour") >= 13) & (pl.col("hour") <= 19))
            .then(pl.lit("regular"))
            .otherwise(pl.lit("extended"))
            .alias("session"),
        )
        .select(["hour", "hour_et", "session", "message_count"])
        .write_csv(os.path.join(csv_dir, "intraday_activity.csv"))
    )

    # C — order size histogram bins
    pl.DataFrame({
        "bin_start": hist_edges[:-1].round(2),
        "bin_end":   hist_edges[1:].round(2),
        "count":     hist_counts,
    }).write_csv(os.path.join(csv_dir, "order_size_histogram.csv"))

    # D — action × side percentages
    pl.DataFrame({
        "action":   ["Add (A)", "Cancel (C)", "Fill (F)", "Trade (T)"],
        "bid_pct":  cross_pct[:, 1].round(2),
        "ask_pct":  cross_pct[:, 0].round(2),
        "none_pct": cross_pct[:, 2].round(2),
    }).write_csv(os.path.join(csv_dir, "action_side.csv"))

    print(f"CSVs written to {csv_dir}/")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"QBTS  ·  NASDAQ ITCH MBO  ·  {date_min} – {date_max}  ·  {n_rows:,} events",
        fontsize=FONT_SUPTITLE, fontweight="bold", y=0.995,
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: Daily volume ─────────────────────────────────────────────────
    dates  = [str(d) for d in daily["date"]]
    counts = daily["len"].to_list()

    ax_a.bar(range(len(dates)), counts, color=HEADER_COLOR, alpha=0.9, width=0.7)
    ax_a.set_xticks(range(len(dates)))
    ax_a.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right",
                         fontsize=FONT_TINY)
    ax_a.yaxis.set_major_formatter(millions_formatter())
    ax_a.set_ylabel("Message count", fontsize=FONT_AXIS)
    ax_a.set_title("A  ·  Daily Message Volume", fontsize=FONT_TITLE,
                   fontweight="bold", loc="left")
    style_ax(ax_a)

    # ── Panel B: Intraday hourly profile ──────────────────────────────────────
    hours   = hourly["hour"].to_list()
    hcounts = hourly["len"].to_list()
    bar_colors = [TEAL if (13 <= h <= 19) else MUTED for h in hours]

    ax_b.bar(hours, hcounts, color=bar_colors, alpha=0.85, width=0.7)
    ax_b.yaxis.set_major_formatter(millions_formatter())
    ax_b.set_xlabel("Hour (UTC)", fontsize=FONT_SMALL)
    ax_b.set_ylabel("Message count", fontsize=FONT_AXIS)
    ax_b.set_title("B  ·  Intraday Activity Profile", fontsize=FONT_TITLE,
                   fontweight="bold", loc="left")
    style_ax(ax_b)

    for h in [h for h in hours if h in {8, 13, 16, 20}]:
        ax_b.text(h, ax_b.get_ylim()[1] * 0.02, _utc_to_et_label(h),
                  ha="center", va="bottom", fontsize=FONT_TINY, color=GRAY,
                  style="italic")

    ax_b.legend(
        handles=[Patch(color=TEAL, label="Regular session (9:30–4 PM ET)"),
                 Patch(color=MUTED, label="Extended hours")],
        fontsize=FONT_SMALL, frameon=False, loc="upper right",
    )

    # ── Panel C: Order size distribution ──────────────────────────────────────
    ax_c.hist(sizes, bins=log_bins, color=SLATE, alpha=0.9,
              edgecolor="white", linewidth=0.3)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax_c.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_c.set_xlabel("Order size (shares)", fontsize=FONT_SMALL)
    ax_c.set_ylabel("Count (log scale)", fontsize=FONT_AXIS)
    ax_c.set_title("C  ·  Order Size Distribution  (Add + Trade, ≤5,000 shares)",
                   fontsize=FONT_TITLE, fontweight="bold", loc="left")
    style_ax(ax_c)

    med = int(np.median(sizes))
    ax_c.axvline(med, color=RED, lw=1.2, ls="--")
    ax_c.text(med * 1.15, ax_c.get_ylim()[1] * 0.5,
              f"median = {med}", fontsize=FONT_SMALL, color=RED)

    # ── Panel D: Action × Side grouped bar chart ──────────────────────────────
    # Bid/Ask are always ~50/50 — a grouped bar makes the Trade "None" slice clear
    action_labels = ["Add (A)", "Cancel (C)", "Fill (F)", "Trade (T)"]
    x     = np.arange(len(actions))
    width = 0.26

    bid_pct  = cross_pct[:, 1]   # B column
    ask_pct  = cross_pct[:, 0]   # A column
    none_pct = cross_pct[:, 2]   # N column

    ax_d.bar(x - width, bid_pct,  width, label="Bid",  color=HEADER_COLOR, alpha=0.9)
    ax_d.bar(x,         ask_pct,  width, label="Ask",  color=TEAL,         alpha=0.9)
    ax_d.bar(x + width, none_pct, width, label="None", color=MUTED,        alpha=0.9)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(action_labels, fontsize=FONT_TICK)
    ax_d.set_ylabel("% of action type", fontsize=FONT_AXIS)
    ax_d.set_ylim(0, 65)
    ax_d.set_title("D  ·  Action × Side  (% of each action type)",
                   fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_d.legend(fontsize=FONT_SMALL, frameon=False, loc="upper right")
    style_ax(ax_d)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    df = pl.read_parquet("data/formatted/xnas_itch_mbo.parquet")
    plot(df)
