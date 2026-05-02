"""
Pre-process raw MBO .dbn.zst files into 1-second snapshots.
Uses databento_dbn directly (no pyarrow dependency).
"""

import os
import glob
import numpy as np
import pandas as pd
import databento_dbn as dbn
import zstandard
from typing import Optional


def load_day_dbn(filepath: str) -> pd.DataFrame:
    """Read a .dbn.zst file into a DataFrame using the low-level decoder."""
    decoder = dbn.DBNDecoder()
    rows = []

    with open(filepath, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(f)

        while True:
            chunk = reader.read(16 * 1024 * 1024)
            if not chunk:
                break
            records = decoder.write_and_decode(chunk)
            for r in records:
                if isinstance(r, dbn.MBOMsg):
                    rows.append({
                        "ts_event": pd.Timestamp(r.ts_event, unit="ns", tz="UTC"),
                        "action": chr(r.action),
                        "side": chr(r.side),
                        "price": r.price / dbn.FIXED_PRICE_SCALE,
                        "size": int(r.size),
                        "order_id": int(r.order_id),
                    })

    df = pd.DataFrame(rows)
    return df


def reconstruct_book_features_for_day(df: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    """
    Process a single day of MBO events into time-bucketed snapshots.
    Tracks the order book incrementally and emits features per bucket.
    """
    df = df.sort_values("ts_event").reset_index(drop=True)
    orders = {}

    ts_col = df["ts_event"].values
    actions = df["action"].values
    sides = df["side"].values
    prices = df["price"].values
    sizes = df["size"].values.astype(np.int64)
    order_ids = df["order_id"].values.astype(np.int64)

    t_start = pd.Timestamp(ts_col[0]).floor(freq)
    t_end = pd.Timestamp(ts_col[-1]).ceil(freq)
    buckets = pd.date_range(t_start, t_end, freq=freq)
    bucket_ends_ns = buckets[1:].values if len(buckets) > 1 else np.array([t_end.asm8])

    bucket_idx = 0

    trades_buy_prices = []
    trades_buy_sizes = []
    trades_sell_prices = []
    trades_sell_sizes = []
    add_bid_vol = 0
    add_ask_vol = 0
    cancel_bid_vol = 0
    cancel_ask_vol = 0
    n_events = 0

    prev_bb = np.nan
    prev_bb_sz = 0
    prev_ba = np.nan
    prev_ba_sz = 0

    records = []

    def get_bbo():
        bids_p = [o[0] for o in orders.values() if o[2] == "B"]
        asks_p = [o[0] for o in orders.values() if o[2] == "A"]
        if bids_p:
            bb = max(bids_p)
            bb_sz = sum(o[1] for o in orders.values() if o[2] == "B" and o[0] == bb)
        else:
            bb, bb_sz = np.nan, 0
        if asks_p:
            ba = min(asks_p)
            ba_sz = sum(o[1] for o in orders.values() if o[2] == "A" and o[0] == ba)
        else:
            ba, ba_sz = np.nan, 0
        return bb, bb_sz, ba, ba_sz

    for i in range(len(df)):
        ts = ts_col[i]

        while bucket_idx < len(bucket_ends_ns) and ts >= bucket_ends_ns[bucket_idx]:
            bb, bb_sz, ba, ba_sz = get_bbo()
            mid = (bb + ba) / 2 if not (np.isnan(bb) or np.isnan(ba)) else np.nan
            spread = (ba - bb) if not (np.isnan(bb) or np.isnan(ba)) else np.nan

            d_bid = (bb_sz - prev_bb_sz) if bb == prev_bb else bb_sz
            d_ask = (ba_sz - prev_ba_sz) if ba == prev_ba else -ba_sz
            ofi = d_bid - d_ask

            total_top = bb_sz + ba_sz
            bimb = (bb_sz - ba_sz) / total_top if total_top > 0 else 0.0

            records.append({
                "timestamp": buckets[bucket_idx],
                "mid_price": mid,
                "best_bid": bb, "best_ask": ba,
                "best_bid_size": bb_sz, "best_ask_size": ba_sz,
                "spread": spread, "ofi": ofi, "book_imbalance": bimb,
                "add_bid_vol": add_bid_vol, "add_ask_vol": add_ask_vol,
                "cancel_bid_vol": cancel_bid_vol, "cancel_ask_vol": cancel_ask_vol,
                "n_events": n_events,
                "max_buy_trade_price": max(trades_buy_prices) if trades_buy_prices else np.nan,
                "min_sell_trade_price": min(trades_sell_prices) if trades_sell_prices else np.nan,
                "buy_trade_volume": sum(trades_buy_sizes),
                "sell_trade_volume": sum(trades_sell_sizes),
                "n_trades": len(trades_buy_prices) + len(trades_sell_prices),
            })

            prev_bb, prev_bb_sz, prev_ba, prev_ba_sz = bb, bb_sz, ba, ba_sz
            trades_buy_prices, trades_buy_sizes = [], []
            trades_sell_prices, trades_sell_sizes = [], []
            add_bid_vol = add_ask_vol = cancel_bid_vol = cancel_ask_vol = n_events = 0
            bucket_idx += 1

        n_events += 1
        action = actions[i]
        side = sides[i]
        price = prices[i]
        size = int(sizes[i])
        oid = int(order_ids[i])

        if action == "R":
            orders.clear()
        elif action == "A":
            orders[oid] = (price, size, side)
            if side == "B":
                add_bid_vol += size
            elif side == "A":
                add_ask_vol += size
        elif action == "C":
            if oid in orders:
                o = orders.pop(oid)
                if o[2] == "B":
                    cancel_bid_vol += o[1]
                elif o[2] == "A":
                    cancel_ask_vol += o[1]
        elif action == "M":
            if oid in orders:
                old = orders[oid]
                orders[oid] = (price, size, old[2])
        elif action in ("T", "F"):
            if side == "B":
                trades_buy_prices.append(price)
                trades_buy_sizes.append(size)
            elif side == "A":
                trades_sell_prices.append(price)
                trades_sell_sizes.append(size)
            if oid in orders:
                old = orders[oid]
                new_sz = old[1] - size
                if new_sz <= 0:
                    del orders[oid]
                else:
                    orders[oid] = (old[0], new_sz, old[2])

    return pd.DataFrame(records)


def add_derived_features(df: pd.DataFrame, vol_window: int = 60) -> pd.DataFrame:
    df = df.copy()
    df["mid_return"] = df["mid_price"].pct_change()
    df["volatility"] = df["mid_return"].rolling(vol_window, min_periods=1).std()
    df["trade_intensity"] = df["n_trades"].rolling(vol_window, min_periods=1).mean()
    cancel_rate = df["cancel_bid_vol"] + df["cancel_ask_vol"]
    add_rate = df["add_bid_vol"] + df["add_ask_vol"]
    df["cancel_add_ratio"] = np.where(add_rate > 0, cancel_rate / add_rate, 0)
    df["ofi_rolling"] = df["ofi"].rolling(30, min_periods=1).mean()
    df["spread_ma"] = df["spread"].rolling(vol_window, min_periods=1).mean()
    return df


def preprocess_day(filepath: str, output_dir: str, freq: str = "1s") -> str:
    date_str = os.path.basename(filepath).split("-")[-1].replace(".mbo.dbn.zst", "")
    output_path = os.path.join(output_dir, f"snapshots_{date_str}.csv")

    if os.path.exists(output_path):
        print(f"  Skipping {date_str} (already processed)")
        return output_path

    print(f"  Processing {date_str}...", end="", flush=True)
    df = load_day_dbn(filepath)
    print(f" {len(df):,} events", end="", flush=True)
    snapshots = reconstruct_book_features_for_day(df, freq=freq)
    snapshots = add_derived_features(snapshots)

    if not snapshots.empty:
        h = snapshots["timestamp"].dt.hour
        m = snapshots["timestamp"].dt.minute
        t = h + m / 60.0
        snapshots = snapshots[(t >= 13.5) & (t < 20.0)].reset_index(drop=True)

    snapshots.to_csv(output_path, index=False)
    print(f" → {len(snapshots)} snapshots saved")
    return output_path


def preprocess_all(data_dir: str, output_dir: str, freq: str = "1s") -> list:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "*.mbo.dbn.zst")))
    if not files:
        raise FileNotFoundError(f"No .mbo.dbn.zst files found in {data_dir}")
    print(f"Found {len(files)} day files to process")
    paths = [preprocess_day(f, output_dir, freq) for f in files]
    print(f"Done! Processed {len(paths)} days")
    return paths


def load_snapshots(snapshot_dir: str, train_days: int = None) -> tuple:
    files = sorted(glob.glob(os.path.join(snapshot_dir, "snapshots_*.csv")))
    if not files:
        raise FileNotFoundError(f"No snapshot files in {snapshot_dir}")

    dfs = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    if train_days is None:
        train_days = len(dfs) - 5

    train = pd.concat(dfs[:train_days], ignore_index=True) if train_days > 0 else pd.DataFrame()
    test = pd.concat(dfs[train_days:], ignore_index=True)

    print(f"Train: {len(train):,} snapshots ({train_days} days)")
    print(f"Test:  {len(test):,} snapshots ({len(dfs) - train_days} days)")
    return train, test


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "XNAS-20260227-QVD7UYV7GQ"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "snapshots"
    preprocess_all(data_dir, out_dir)
