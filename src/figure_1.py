import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter

import polars as pl


def _fmt_dtype(dtype) -> str:
    """Convert verbose polars dtype repr to a compact display string."""
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
    "action":        "A=Add  C=Cancel  T=Trade  F=Fill  R=Reset",
    "side":          "B=Bid  A=Ask  N=None",
    "price":         "Limit order price in USD",
    "size":          "Quantity in shares",
    "channel_id":    "Feed channel (always 0 for XNAS)",
    "order_id":      "Links add → cancel / fill lifecycle",
    "flags":         "Message flags bitmask",
    "ts_in_delta":   "Feed-to-exchange latency delta (ns)",
    "sequence":      "Exchange sequence number",
    "symbol":        "Ticker symbol",
}


def plot(df: pl.DataFrame, out_path: str = "data/output/qbts_data_structure.png") -> None:
    # ── Derived data ─────────────────────────────────────────────────────────
    action_props = (
        df["action"]
        .value_counts(normalize=True)
        .filter(pl.col("action") != "R")
        .sort("action")
    )
    action_map = dict(zip(action_props["action"], action_props["proportion"]))
    pie_labels = ["Add (A)", "Cancel (C)", "Fill (F)", "Trade (T)"]
    pie_sizes  = [action_map.get(k, 0) for k in ["A", "C", "F", "T"]]
    pie_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    trades = df.filter(
        (pl.col("action") == "T") &
        (pl.col("price") > 1) &
        (pl.col("price") < 1000)
    )

    n_rows   = len(df)
    n_cols   = len(df.columns)
    date_min = df["ts_event"].min().strftime("%b %d")
    date_max = df["ts_event"].max().strftime("%b %d %Y")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("white")

    # ── Left: schema card (all columns) ──────────────────────────────────────
    ax = fig.add_axes([0.02, 0.04, 0.44, 0.93])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.978, "Schema  —  one row = one order-book event",
            ha="center", va="top", fontsize=12, fontweight="bold", color="#1a1a2e")

    columns = [
        (col, _fmt_dtype(dtype), DESCRIPTIONS.get(col, ""))
        for col, dtype in df.schema.items()
    ]

    HEADER_COLOR = "#1565C0"
    ROW_COLORS   = ["#EEF1FA", "#F8F9FE"]
    XS           = [0.01, 0.33, 0.53]
    HEADER_H     = 0.055
    ROW_H        = 0.060
    y0           = 0.925   # top of header

    ax.add_patch(Rectangle((0, y0 - HEADER_H), 1, HEADER_H,
        facecolor=HEADER_COLOR, edgecolor="white", linewidth=1.5))
    for x, lbl in zip(XS, ["Column", "Type", "Description"]):
        ax.text(x + 0.012, y0 - HEADER_H / 2, lbl,
                ha="left", va="center", fontsize=11, fontweight="bold", color="white")

    for i, (name, dtype, desc) in enumerate(columns):
        y = (y0 - HEADER_H) - i * ROW_H
        ax.add_patch(Rectangle((0, y - ROW_H + 0.002), 1, ROW_H - 0.002,
            facecolor=ROW_COLORS[i % 2], edgecolor="#D0D8F0", linewidth=0.5))
        ax.text(XS[0] + 0.012, y - ROW_H / 2 + 0.002, name,
                ha="left", va="center", fontsize=9, fontfamily="monospace",
                fontweight="bold", color="#1a1a2e")
        ax.text(XS[1] + 0.012, y - ROW_H / 2 + 0.002, dtype,
                ha="left", va="center", fontsize=8.5, fontfamily="monospace", color="#555")
        ax.text(XS[2] + 0.012, y - ROW_H / 2 + 0.002, desc,
                ha="left", va="center", fontsize=8.5, color="#333")

    ax.text(0.5, 0.012,
            f"{n_rows:,} rows  ·  {n_cols} columns  ·  0 nulls  ·  {date_min} – {date_max}",
            ha="center", va="bottom", fontsize=9, color="#666", style="italic")

    # ── Right top: action-type donut ─────────────────────────────────────────
    ax_donut = fig.add_axes([0.51, 0.50, 0.45, 0.47])

    wedges, _, autotexts = ax_donut.pie(
        pie_sizes, labels=None, colors=pie_colors,
        autopct=lambda p: f"{p:.1f}%", pctdistance=0.78,
        startangle=90,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")

    ax_donut.legend(wedges, pie_labels,
                    loc="lower center", ncol=2, fontsize=9, frameon=False,
                    bbox_to_anchor=(0.5, -0.08), labelspacing=0.6, columnspacing=0.8)
    ax_donut.set_title(f"Event Type Distribution\n{n_rows:,} total messages",
                       fontsize=11, fontweight="bold", pad=10)
    ax_donut.text(0, 0, "QBTS\nNASDAQ", ha="center", va="center",
                  fontsize=10, fontweight="bold", color="#333")

    # ── Right bottom: trade price ─────────────────────────────────────────────
    ax_price = fig.add_axes([0.51, 0.07, 0.45, 0.37])

    ax_price.plot(trades["ts_event"], trades["price"], lw=0.3, alpha=0.7, color="#4C72B0")
    ax_price.set_ylabel("Trade Price ($)", fontsize=10)
    ax_price.set_title(f"Trade Price  —  {date_min} to {date_max}",
                       fontsize=11, fontweight="bold")
    ax_price.spines[["top", "right"]].set_visible(False)
    ax_price.tick_params(axis="x", labelsize=8, rotation=20)
    ax_price.tick_params(axis="y", labelsize=9)
    ax_price.xaxis.set_major_formatter(DateFormatter("%b %d"))

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    df = pl.read_parquet("data/formatted/xnas_itch_mbo.parquet")
    plot(df)
