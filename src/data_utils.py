"""
Data utilities for order-imbalance → short-term return prediction.

Pipeline
--------
1. load_raw()            – read the MBO parquet, filter to regular session
2. build_ofi_features()  – aggregate per time bucket: order-flow imbalance + VWAP
3. add_forward_returns() – compute horizon-ahead log-returns (the label)
4. split_by_date()       – time-ordered train / val / test split

Default time splits (based on Oct 2025 data, ~32 trading days):
    train : Oct  1 – Oct 21  (~65 %)
    val   : Oct 22 – Oct 26  (~15 %)
    test  : Oct 27 – Oct 31  (~20 %)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "data" / "formatted" / "xnas_itch_mbo.parquet"

# Regular NYSE/NASDAQ session in UTC (9:30–16:00 ET = 13:30–20:00 UTC)
SESSION_START_UTC = 13
SESSION_END_UTC   = 20

# Default chronological split boundaries
DEFAULT_VAL_START  = "2025-10-22"
DEFAULT_TEST_START = "2025-10-27"

_EPS = 1e-9   # prevent divide-by-zero in OFI


# ── Step 1: Load ──────────────────────────────────────────────────────────────

_NEEDED_COLS = ["ts_event", "action", "side", "price", "size"]


def load_raw(
    path: str | Path = DATA_PATH,
    regular_session_only: bool = True,
) -> pl.DataFrame:
    """
    Load the MBO parquet and return a clean base DataFrame.

    Only the five columns needed for OFI feature-building are loaded;
    the remaining nine are discarded at read time to save memory.

    Filters applied:
    - Drop Reset (R) events — bookkeeping artefacts, not meaningful signals
    - Optionally restrict to regular session hours (default: True)
    - Drop rows with price <= 0 or size == 0
    """
    hour = pl.col("ts_event").dt.hour()
    session_filter = (hour >= SESSION_START_UTC) & (hour < SESSION_END_UTC)

    lf = (
        pl.scan_parquet(path)
        .select(_NEEDED_COLS)
        .filter(pl.col("action") != "R")
        .filter((pl.col("price") > 0) & (pl.col("size") > 0))
    )
    if regular_session_only:
        lf = lf.filter(session_filter)

    return lf.sort("ts_event").collect()


# ── Step 2: Build features ────────────────────────────────────────────────────

def build_ofi_features(
    df: pl.DataFrame,
    freq: str = "1m",
) -> pl.DataFrame:
    """
    Aggregate MBO events into time buckets and compute per-bucket features.

    Order-Flow Imbalance (OFI)
    --------------------------
    Uses limit-order Add events (action=A) as the signal:
        buy_vol  = Σ size  where action=A, side=B (bids placed)
        sell_vol = Σ size  where action=A, side=A (asks placed)
        ofi      = (buy_vol − sell_vol) / (buy_vol + sell_vol)

    OFI is bounded in [−1, +1].  +1 → all new orders are buys; −1 → all sells.

    Additional features per bucket
    --------------------------------
    - trade_buy_vol / trade_sell_vol  : signed trade volume (T events by side)
    - trade_ofi                       : same imbalance metric for executed trades
    - vwap                            : volume-weighted average trade price
    - n_trades                        : number of trade events
    - cancel_ratio                    : cancel volume / add volume (liquidity proxy)

    Parameters
    ----------
    df   : output of load_raw()
    freq : polars duration string, e.g. "1m", "5m", "30s"

    Returns
    -------
    DataFrame indexed by `bucket` (UTC timestamp, start of interval), one row
    per non-empty bucket, sorted by time.
    """
    # ── Add events (limit order placement) ───────────────────────────────────
    adds = df.filter(pl.col("action") == "A")

    add_agg = (
        adds
        .sort("ts_event")
        .group_by_dynamic("ts_event", every=freq)
        .agg([
            pl.col("size").filter(pl.col("side") == "B").sum().alias("buy_vol"),
            pl.col("size").filter(pl.col("side") == "A").sum().alias("sell_vol"),
            pl.col("size").filter(pl.col("side") == "B").count().alias("n_bids"),
            pl.col("size").filter(pl.col("side") == "A").count().alias("n_asks"),
        ])
        .rename({"ts_event": "bucket"})
    )

    # ── Cancel events ─────────────────────────────────────────────────────────
    cancels = df.filter(pl.col("action") == "C")

    cancel_agg = (
        cancels
        .sort("ts_event")
        .group_by_dynamic("ts_event", every=freq)
        .agg(pl.col("size").sum().alias("cancel_vol"))
        .rename({"ts_event": "bucket"})
    )

    # ── Trade events (action=T) ───────────────────────────────────────────────
    trades = df.filter(pl.col("action").is_in(["T", "F"]))

    trade_agg = (
        trades
        .sort("ts_event")
        .group_by_dynamic("ts_event", every=freq)
        .agg([
            pl.col("size").filter(pl.col("side") == "B").sum().alias("trade_buy_vol"),
            pl.col("size").filter(pl.col("side") == "A").sum().alias("trade_sell_vol"),
            # VWAP: Σ(price × size) / Σ(size)
            (pl.col("price") * pl.col("size")).sum().alias("_pv"),
            pl.col("size").sum().alias("_tv"),
            pl.col("price").count().alias("n_trades"),
        ])
        .with_columns(
            (pl.col("_pv") / pl.col("_tv")).alias("vwap")
        )
        .drop(["_pv", "_tv"])
        .rename({"ts_event": "bucket"})
    )

    # ── Join and derive imbalance metrics ─────────────────────────────────────
    features = (
        add_agg
        .join(cancel_agg, on="bucket", how="left")
        .join(trade_agg,  on="bucket", how="left")
        .fill_null(0)
        .with_columns([
            # Order-flow imbalance from limit adds
            (
                (pl.col("buy_vol") - pl.col("sell_vol")) /
                (pl.col("buy_vol") + pl.col("sell_vol") + _EPS)
            ).alias("ofi"),

            # Trade-direction imbalance
            (
                (pl.col("trade_buy_vol") - pl.col("trade_sell_vol")) /
                (pl.col("trade_buy_vol") + pl.col("trade_sell_vol") + _EPS)
            ).alias("trade_ofi"),

            # Fraction of add volume that was subsequently cancelled (same bucket)
            (
                pl.col("cancel_vol") /
                (pl.col("buy_vol") + pl.col("sell_vol") + _EPS)
            ).alias("cancel_ratio"),
        ])
        .sort("bucket")
    )

    return features


# ── Step 3: Labels ────────────────────────────────────────────────────────────

def add_forward_returns(
    features: pl.DataFrame,
    horizon: int = 5,
) -> pl.DataFrame:
    """
    Append `fwd_return` — the log-return `horizon` buckets ahead.

        fwd_return[t] = log( vwap[t + horizon] / vwap[t] )

    Rows where vwap is 0 or the future window is unavailable are dropped.

    Parameters
    ----------
    features : output of build_ofi_features()
    horizon  : number of time buckets to look ahead (default: 5)
    """
    df = (
        features
        .filter(pl.col("vwap") > 0)
        .with_columns(
            pl.col("vwap").shift(-horizon).alias("vwap_fwd")
        )
        .filter(pl.col("vwap_fwd").is_not_null() & (pl.col("vwap_fwd") > 0))
        .with_columns(
            (pl.col("vwap_fwd") / pl.col("vwap")).log(base=2.71828).alias("fwd_return")
        )
        .drop("vwap_fwd")
    )
    return df


# ── Step 4: Train / Val / Test split ─────────────────────────────────────────

def split_by_date(
    df: pl.DataFrame,
    val_start:  str = DEFAULT_VAL_START,
    test_start: str = DEFAULT_TEST_START,
    date_col:   str = "bucket",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a time-sorted DataFrame into train / val / test on hard date cuts.

    All splits are non-overlapping and strictly ordered in time — no
    shuffling, no random sampling.  This prevents look-ahead leakage.

    Default boundaries
    ------------------
    train : [start,      2025-10-22)   ~65 % of days
    val   : [2025-10-22, 2025-10-27)   ~15 %
    test  : [2025-10-27, end]          ~20 %

    Parameters
    ----------
    df         : any DataFrame with a datetime column
    val_start  : inclusive start date for validation set (ISO string)
    test_start : inclusive start date for test set (ISO string)
    date_col   : name of the timestamp column to split on

    Returns
    -------
    (train, val, test) as Polars DataFrames
    """
    val_dt  = pl.lit(val_start).str.to_date().cast(pl.Datetime("ns", "UTC"))
    test_dt = pl.lit(test_start).str.to_date().cast(pl.Datetime("ns", "UTC"))

    col = pl.col(date_col)

    train = df.filter(col < val_dt)
    val   = df.filter((col >= val_dt) & (col < test_dt))
    test  = df.filter(col >= test_dt)

    return train, val, test


# ── Convenience wrapper ───────────────────────────────────────────────────────

def load_splits(
    path:       str | Path = DATA_PATH,
    freq:       str        = "1m",
    horizon:    int        = 5,
    val_start:  str        = DEFAULT_VAL_START,
    test_start: str        = DEFAULT_TEST_START,
    regular_session_only: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Full pipeline in one call.

    Returns (train, val, test) DataFrames with columns:
        bucket, buy_vol, sell_vol, n_bids, n_asks,
        cancel_vol, trade_buy_vol, trade_sell_vol, n_trades,
        vwap, ofi, trade_ofi, cancel_ratio, fwd_return
    """
    df       = load_raw(path, regular_session_only=regular_session_only)
    features = build_ofi_features(df, freq=freq)
    labeled  = add_forward_returns(features, horizon=horizon)
    return split_by_date(labeled, val_start=val_start, test_start=test_start)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    train, val, test = load_splits(freq="1m", horizon=5)

    total = len(train) + len(val) + len(test)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        pct = 100 * len(split) / total
        date_min = split["bucket"].min()
        date_max = split["bucket"].max()
        print(f"  {name:5s}  {len(split):5d} rows ({pct:.0f}%)  "
              f"{date_min.date()} → {date_max.date()}")

    print("\nSample feature stats (train):")
    print(train.select(["ofi", "trade_ofi", "cancel_ratio", "fwd_return"]).describe())
