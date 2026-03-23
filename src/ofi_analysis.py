"""
Quantify order-flow imbalance (OFI) and its relationship to short-term returns.

Panels
------
A  OFI distribution          — how (im)balanced limit-order flow is
B  OFI time series           — 30-bucket rolling mean across the training period
C  Binned OFI vs fwd return  — does higher imbalance predict higher returns?
D  Daily IC                  — per-day Pearson corr(OFI, fwd_return) over time

All analysis is performed on the training split only to avoid any look-ahead.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import load_splits
from theme import (
    BG, TEAL, BLUE, ORANGE, RED, GRAY,
    FONT_SUPTITLE, FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_SMALL, FONT_TINY,
    style_ax, apply_figure_defaults,
)

FREQ    = "1m"
HORIZON = 5        # buckets ahead (~5 min with freq="1m")
N_BINS  = 10       # OFI quantile bins for panel C
ROLL    = 30       # rolling window (buckets) for panel B smoother


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, returns nan if either array has zero variance."""
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _linreg(x: np.ndarray, y: np.ndarray) -> dict:
    """
    OLS: y = α + β·x.  Returns slope, intercept, R², RMSE, t-stat, p-value.
    t-stat uses the standard OLS formula; p-value is two-tailed.
    """
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coeffs

    y_hat  = alpha + beta * x
    resid  = y - y_hat
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse   = np.sqrt(ss_res / n)

    # t-statistic for β
    s2    = ss_res / (n - 2)
    x_dev = x - x.mean()
    se_b  = np.sqrt(s2 / (x_dev ** 2).sum()) if (x_dev ** 2).sum() > 0 else float("nan")
    t_stat = beta / se_b if se_b > 0 else float("nan")

    # Two-tailed p-value approximated via normal (large n)
    p_val = 2 * (1 - _normal_cdf(abs(t_stat))) if not np.isnan(t_stat) else float("nan")

    return dict(alpha=alpha, beta=beta, r2=r2, rmse=rmse, t_stat=t_stat, p_val=p_val)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via the Abramowitz & Stegun approximation."""
    t = 1 / (1 + 0.2316419 * abs(z))
    p = 1 - (0.319381530 * t
             - 0.356563782 * t**2
             + 1.781477937 * t**3
             - 1.821255978 * t**4
             + 1.330274429 * t**5) * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return p if z >= 0 else 1 - p


def _eval_split(model: dict, df: pl.DataFrame) -> dict:
    """Apply a fitted model to a split and return R² and RMSE."""
    x = df["ofi"].to_numpy()
    y = df["fwd_return"].to_numpy()
    y_hat = model["alpha"] + model["beta"] * x
    resid  = y - y_hat
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return dict(
        r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        rmse = np.sqrt(ss_res / len(y)),
        ic   = _pearson(x, y),
    )


def fit_linear_model(
    train: pl.DataFrame,
    val:   pl.DataFrame,
    test:  pl.DataFrame,
) -> dict:
    """
    Fit fwd_return = α + β·OFI on train; evaluate on all three splits.
    Prints a results table and returns the model dict.
    """
    x_tr = train["ofi"].to_numpy()
    y_tr = train["fwd_return"].to_numpy()

    model  = _linreg(x_tr, y_tr)
    splits = {
        "train": _eval_split(model, train),
        "val":   _eval_split(model, val),
        "test":  _eval_split(model, test),
    }

    print(f"\n{'─'*60}")
    print(f"  Linear regression:  fwd_return = α + β·OFI")
    print(f"{'─'*60}")
    print(f"  α (intercept) : {model['alpha']:+.6f}")
    print(f"  β (OFI coeff) : {model['beta']:+.6f}")
    print(f"  t-stat        : {model['t_stat']:+.2f}")
    print(f"  p-value       : {model['p_val']:.4f}")
    print(f"{'─'*60}")
    print(f"  {'Split':<8}  {'R²':>8}  {'RMSE':>12}  {'IC':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*8}")
    for name, m in splits.items():
        print(f"  {name:<8}  {m['r2']:>+8.5f}  {m['rmse']:>12.6f}  {m['ic']:>+8.4f}")
    print(f"{'─'*60}\n")

    return model


def _print_stats(train: pl.DataFrame) -> None:
    """Print a compact summary table to stdout."""
    ofi = train["ofi"].to_numpy()
    ret = train["fwd_return"].to_numpy()

    ic_overall = _pearson(ofi, ret)

    print(f"\n{'─'*52}")
    print(f"  OFI quantification summary  (train, freq={FREQ}, h={HORIZON})")
    print(f"{'─'*52}")
    print(f"  Buckets          : {len(train):,}")
    print(f"  OFI mean         : {ofi.mean():+.4f}")
    print(f"  OFI std          : {ofi.std():.4f}")
    print(f"  OFI p5 / p95     : {np.percentile(ofi, 5):+.3f} / {np.percentile(ofi, 95):+.3f}")
    print(f"  fwd_return mean  : {ret.mean():+.6f}")
    print(f"  fwd_return std   : {ret.std():.6f}")
    print(f"  IC (OFI, ret)    : {ic_overall:+.4f}")
    print(f"{'─'*52}\n")


def plot(
    train: pl.DataFrame,
    model: dict | None = None,
    out_path: str = "data/output/ofi_analysis.png",
) -> None:
    apply_figure_defaults()

    # ── Pre-compute ───────────────────────────────────────────────────────────
    ofi_arr = train["ofi"].to_numpy()
    ret_arr = train["fwd_return"].to_numpy()

    # Panel B: time + rolling mean
    buckets   = train["bucket"].to_list()
    ofi_roll  = (
        train
        .with_columns(pl.col("ofi").rolling_mean(window_size=ROLL).alias("ofi_roll"))
        ["ofi_roll"]
        .to_numpy()
    )

    # Panel C: bin OFI into deciles, compute mean fwd_return per bin
    ofi_s  = pl.Series(ofi_arr)
    ret_s  = pl.Series(ret_arr)
    labels = ofi_s.qcut(N_BINS, labels=[str(i) for i in range(N_BINS)])
    binned = (
        pl.DataFrame({"bin": labels.cast(pl.Int32), "ret": ret_s, "ofi": ofi_s})
        .group_by("bin")
        .agg([
            pl.col("ret").mean().alias("mean_ret"),
            pl.col("ofi").mean().alias("mean_ofi"),
            pl.col("ret").count().alias("n"),
        ])
        .sort("bin")
    )
    bin_ofi  = binned["mean_ofi"].to_numpy()
    bin_ret  = binned["mean_ret"].to_numpy() * 1e4   # convert to basis points
    bin_col  = [TEAL if r >= 0 else RED for r in bin_ret]

    # Panel D: daily IC
    daily_ic = (
        train
        .with_columns(pl.col("bucket").dt.date().alias("date"))
        .group_by("date")
        .map_groups(lambda g: g.with_columns(
            pl.lit(_pearson(g["ofi"].to_numpy(), g["fwd_return"].to_numpy()))
            .alias("ic")
        ))
        .select(["date", "ic"])
        .unique("date")
        .sort("date")
    )
    ic_dates = daily_ic["date"].to_list()
    ic_vals  = daily_ic["ic"].to_numpy()
    ic_mean  = float(np.nanmean(ic_vals))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Order-Flow Imbalance Analysis  ·  QBTS  ·  freq={FREQ}  ·  "
        f"horizon={HORIZON} min  ·  train split  ·  {len(train):,} buckets",
        fontsize=FONT_SUPTITLE, fontweight="bold", y=0.998,
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: OFI distribution ─────────────────────────────────────────────
    ax_a.hist(ofi_arr, bins=80, color=BLUE, alpha=0.85,
              edgecolor="white", linewidth=0.3)
    ax_a.axvline(0,            color=GRAY, lw=1.0, ls="--", alpha=0.7)
    ax_a.axvline(ofi_arr.mean(), color=ORANGE, lw=1.5, ls="-",
                 label=f"mean = {ofi_arr.mean():+.3f}")
    ax_a.set_xlabel("OFI  [(buy_vol − sell_vol) / total_vol]", fontsize=FONT_SMALL)
    ax_a.set_ylabel("Bucket count", fontsize=FONT_AXIS)
    ax_a.set_title("A  ·  OFI Distribution", fontsize=FONT_TITLE,
                   fontweight="bold", loc="left")
    ax_a.legend(fontsize=FONT_SMALL, frameon=False)
    style_ax(ax_a)

    # ── Panel B: OFI time series ──────────────────────────────────────────────
    ax_b.plot(buckets, ofi_arr, color=BLUE, alpha=0.18, lw=0.5)
    ax_b.plot(buckets, ofi_roll, color=TEAL, lw=1.4,
              label=f"{ROLL}-bucket rolling mean")
    ax_b.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax_b.set_ylabel("OFI", fontsize=FONT_AXIS)
    ax_b.set_title("B  ·  OFI over Time  (train period)",
                   fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d"))
    ax_b.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax_b.tick_params(axis="x", labelsize=FONT_TINY, rotation=30)
    ax_b.tick_params(axis="y", labelsize=FONT_TICK)
    ax_b.legend(fontsize=FONT_SMALL, frameon=False, loc="upper right")
    style_ax(ax_b)

    # ── Panel C: Binned OFI vs mean forward return ────────────────────────────
    bars = ax_c.bar(range(N_BINS), bin_ret, color=bin_col, alpha=0.88,
                    edgecolor="white", linewidth=0.5)
    ax_c.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.6)

    # Annotate mean OFI per bin along x-axis
    ax_c.set_xticks(range(N_BINS))
    ax_c.set_xticklabels(
        [f"{v:+.2f}" for v in bin_ofi],
        fontsize=FONT_TINY, rotation=35, ha="right",
    )
    ax_c.set_xlabel("Mean OFI in decile  (low → high imbalance)", fontsize=FONT_SMALL)
    ax_c.set_ylabel("Mean forward return (bps)", fontsize=FONT_AXIS)
    ax_c.set_title(f"C  ·  OFI Decile vs {HORIZON}-min Forward Return",
                   fontsize=FONT_TITLE, fontweight="bold", loc="left")

    if model is not None:
        x_line = np.linspace(bin_ofi.min(), bin_ofi.max(), 100)
        y_line = (model["alpha"] + model["beta"] * x_line) * 1e4
        # Map x_line to panel C's 0..N_BINS-1 x-axis via linear interpolation
        x_idx  = np.interp(x_line, [bin_ofi.min(), bin_ofi.max()], [0, N_BINS - 1])
        ax_c.plot(x_idx, y_line, color=ORANGE, lw=1.8, ls="--",
                  label=f"OLS fit  β={model['beta']:+.4f}  R²={model['r2']:.4f}")
        ax_c.legend(fontsize=FONT_SMALL, frameon=False)

    style_ax(ax_c)

    # ── Panel D: Daily IC ─────────────────────────────────────────────────────
    bar_cols = [TEAL if v >= 0 else RED for v in ic_vals]
    ax_d.bar(ic_dates, ic_vals, color=bar_cols, alpha=0.85,
             edgecolor="white", linewidth=0.3)
    ax_d.axhline(0,       color=GRAY,   lw=0.8, ls="--", alpha=0.6)
    ax_d.axhline(ic_mean, color=ORANGE, lw=1.4, ls="-",
                 label=f"mean IC = {ic_mean:+.3f}")
    ax_d.set_ylabel("Pearson IC  (OFI vs fwd_return)", fontsize=FONT_AXIS)
    ax_d.set_title("D  ·  Daily Information Coefficient",
                   fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d"))
    ax_d.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_d.tick_params(axis="x", labelsize=FONT_TINY, rotation=30)
    ax_d.tick_params(axis="y", labelsize=FONT_TICK)
    ax_d.legend(fontsize=FONT_SMALL, frameon=False)
    style_ax(ax_d)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    print(f"Loading data  (freq={FREQ}, horizon={HORIZON})...")
    train, val, test = load_splits(freq=FREQ, horizon=HORIZON)
    print(train.head())
    _print_stats(train)
    model = fit_linear_model(train, val, test)
    plot(train, model=model)
