"""
Slide figure: high-level dataset overview for QBTS NASDAQ ITCH MBO.

Layout
------
Left   Slide title ("Background / Introduction") + schema card
Right  (top)    Action-type donut  — legend to the right
Right  (bottom) Daily close price series
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

import polars as pl

from theme import (
    BG, TEAL, PIE_COLORS, HEADER_COLOR, ROW_COLORS,
    FONT_TITLE, FONT_TICK, FONT_SMALL, FONT_TINY,
    apply_figure_defaults,
)


def _fmt_dtype(dtype) -> str:
    return {
        "Datetime(time_unit='ns', time_zone='UTC')": "datetime[ns, UTC]",
        "String": "str",
        "UInt8": "u8", "UInt16": "u16", "UInt32": "u32", "UInt64": "u64",
        "Int8": "i8", "Int16": "i16", "Int32": "i32", "Int64": "i64",
        "Float32": "f32", "Float64": "f64",
    }.get(str(dtype), str(dtype))


DESCRIPTIONS = {
    "ts_event":      "Nanosecond-precision event timestamp",
    "rtype":         "Record type (160 = MBO)",
    "publisher_id":  "Publisher ID (2 = NASDAQ ITCH)",
    "instrument_id": "Internal instrument identifier",
    "action":        "A=Add C=Cancel T=Trade F=Fill R=Reset",
    "side":          "B=Bid A=Ask N=None",
    "price":         "Limit order price in USD",
    "size":          "Quantity in shares",
    "channel_id":    "Feed channel (always 0 for XNAS)",
    "order_id":      "Links → cancel / fill lifecycle",
    "flags":         "Message Flags",
    "ts_in_delta":   "Feed-to-exchange latency delta (ns)",
    "sequence":      "Exchange sequence number",
    "symbol":        "Ticker symbol",
}


def plot(df: pl.DataFrame, out_path: str = "data/output/data_overview.png") -> None:
    apply_figure_defaults()

    # ── Derived data ──────────────────────────────────────────────────────────
    action_props = (
        df["action"]
        .value_counts(normalize=True)
        .filter(pl.col("action") != "R")
        .sort("action")
    )
    action_map = dict(zip(action_props["action"], action_props["proportion"]))

    # Order matches PIE_COLORS: Add, Cancel, Trade, Fill
    pie_labels = ["Add", "Cancel", "Trade", "Fill"]
    pie_sizes  = [action_map.get(k, 0) for k in ["A", "C", "T", "F"]]

    # Daily close: last trade within regular session (13–20 UTC = 9:30–4 PM ET)
    daily_close = (
        df.filter(
            (pl.col("action") == "T") &
            (pl.col("price") > 1) &
            (pl.col("price") < 1_000) &
            (pl.col("ts_event").dt.hour() >= 13) &
            (pl.col("ts_event").dt.hour() < 20)
        )
        .sort("ts_event")
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("price").last().alias("close"))
        .sort("date")
    )

    n_rows = len(df)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))

    # ── Left: title + schema card ─────────────────────────────────────────────
    ax = fig.add_axes([0.02, 0.03, 0.44, 0.94])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Slide title section
    ax.text(0.01, 0.988, "Background",
            ha="left", va="top", fontsize=10, color=TEAL)
    ax.text(0.01, 0.968, "Introduction",
            ha="left", va="top", fontsize=28, fontweight="bold", color="#1a1a1a")
    ax.text(0.01, 0.872, "Data Source & Structure",
            ha="left", va="top", fontsize=13, color=TEAL, fontweight="bold")
    # Decorative rule below subtitle
    ax.plot([0.01, 0.99], [0.854, 0.854], color=TEAL, lw=1.5, alpha=0.7)

    # Schema table
    columns = [
        (col, _fmt_dtype(dtype), DESCRIPTIONS.get(col, ""))
        for col, dtype in df.schema.items()
    ]

    XS       = [0.01, 0.30, 0.52]
    HEADER_H = 0.045
    ROW_H    = 0.052
    y0       = 0.840   # top of header block

    ax.add_patch(Rectangle((0, y0 - HEADER_H), 1, HEADER_H,
        facecolor=HEADER_COLOR, edgecolor="none"))
    for x, lbl in zip(XS, ["Column", "Type", "Description"]):
        ax.text(x + 0.012, y0 - HEADER_H / 2, lbl,
                ha="left", va="center", fontsize=10, fontweight="bold", color="white")

    for i, (name, dtype, desc) in enumerate(columns):
        y = (y0 - HEADER_H) - i * ROW_H
        ax.add_patch(Rectangle((0, y - ROW_H + 0.002), 1, ROW_H - 0.002,
            facecolor=ROW_COLORS[i % 2], edgecolor="#D0D8F0", linewidth=0.4))
        ax.text(XS[0] + 0.012, y - ROW_H / 2 + 0.002, name,
                ha="left", va="center", fontsize=9, fontfamily="monospace",
                color=TEAL)
        ax.text(XS[1] + 0.012, y - ROW_H / 2 + 0.002, dtype,
                ha="left", va="center", fontsize=FONT_SMALL, fontfamily="monospace",
                color="#888")
        ax.text(XS[2] + 0.012, y - ROW_H / 2 + 0.002, desc,
                ha="left", va="center", fontsize=FONT_SMALL, color="#444")

    # ── Right top: action-type donut ──────────────────────────────────────────
    ax_donut = fig.add_axes([0.50, 0.40, 0.36, 0.57])

    wedges, _, autotexts = ax_donut.pie(
        pie_sizes, labels=None, colors=PIE_COLORS,
        autopct=lambda p: f"{p:.0f}%", pctdistance=0.72,
        startangle=90,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2.0),
    )
    for at in autotexts:
        at.set_fontsize(FONT_TICK)
        at.set_fontweight("bold")
        at.set_color("white")

    ax_donut.legend(wedges, pie_labels,
                    loc="center left", fontsize=FONT_TICK, frameon=False,
                    bbox_to_anchor=(1.02, 0.5), labelspacing=1.0)
    ax_donut.set_title(
        f"EVENT TYPE DISTRIBUTION\n{n_rows:,} total Messages",
        fontsize=11, fontweight="bold", pad=12,
    )

    # ── Right bottom: daily close price ───────────────────────────────────────
    ax_price = fig.add_axes([0.50, 0.06, 0.46, 0.30])

    close_dates = daily_close["date"].to_list()
    close_vals  = daily_close["close"].to_list()

    ax_price.plot(close_dates, close_vals, lw=1.5, color=TEAL, label="Close")
    ax_price.set_title("D-Wave Price", fontsize=FONT_TITLE, fontweight="bold")
    ax_price.spines[["top", "right"]].set_visible(False)
    ax_price.tick_params(axis="x", labelsize=FONT_TINY, rotation=45)
    ax_price.tick_params(axis="y", labelsize=FONT_TICK)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d"))
    ax_price.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_price.grid(axis="y", color="#cccccc", lw=0.5)
    ax_price.legend(fontsize=FONT_SMALL, frameon=False, loc="lower right")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    df = pl.read_parquet("data/formatted/xnas_itch_mbo.parquet")
    plot(df)
