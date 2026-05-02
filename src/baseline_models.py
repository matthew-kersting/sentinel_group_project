"""
Baseline predictive models: OFI → short-term returns and trade execution activity.

Regression  (fwd_return, 5-min horizon)
  M1  OLS univariate    fwd_return ~ ofi
  M2  OLS multivariate  fwd_return ~ ofi + trade_ofi + cancel_ratio + n_trades
  M3  Ridge             same features as M2, L2-regularized

Classification  (active_next: n_trades[t+1] > training-set median)
  M4  Logistic          active_next ~ ofi + trade_ofi + cancel_ratio
  M5  Logistic + lags   active_next ~ ofi + trade_ofi + cancel_ratio + ofi_lag1 + ofi_lag2

All models are fit on train, evaluated on train / val / test.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import load_raw, build_ofi_features, add_forward_returns, split_by_date
from theme import (
    BG,
    TEAL,
    BLUE,
    ORANGE,
    RED,
    GRAY,
    GREEN,
    FONT_SUPTITLE,
    FONT_TITLE,
    FONT_AXIS,
    FONT_TICK,
    FONT_SMALL,
    FONT_TINY,
    style_ax,
    apply_figure_defaults,
)

FREQ = "1m"
HORIZON = 5
RIDGE_ALPHA = 1.0

REG_UNI = ["ofi"]
REG_MULTI = [
    "ofi",
    "trade_ofi",
    "cancel_ratio",
    "n_trades",
    "cancel_rate",
    "cancel_to_trade_ratio",
    "mean_lifespan_s",
]
CLF_BASE = [
    "ofi",
    "trade_ofi",
    "cancel_ratio",
    "cancel_rate",
    "cancel_to_trade_ratio",
    "mean_lifespan_s",
]
CLF_LAG = [
    "ofi",
    "trade_ofi",
    "cancel_ratio",
    "cancel_rate",
    "cancel_to_trade_ratio",
    "mean_lifespan_s",
    "ofi_lag1",
    "ofi_lag2",
]

SPLIT_COLORS = {
    "train": BLUE,
    "val": ORANGE,
    "test": TEAL,
}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ── Data preparation ──────────────────────────────────────────────────────────


def build_dataset() -> pl.DataFrame:
    print("Loading raw MBO data...")
    raw = load_raw()

    print(f"Building {FREQ} OFI features...")
    features = build_ofi_features(raw, freq=FREQ)

    features = features.with_columns(
        [
            pl.col("ofi").shift(1).alias("ofi_lag1"),
            pl.col("ofi").shift(2).alias("ofi_lag2"),
            pl.col("n_trades").shift(-1).alias("n_trades_next"),
        ]
    )

    features = add_forward_returns(features, horizon=HORIZON)
    return features.drop_nulls()


def make_active_label(
    train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Binary 1 if next-bucket n_trades exceeds the training-set median."""
    threshold = float(train["n_trades_next"].median())

    def binarize(df: pl.DataFrame) -> np.ndarray:
        return (df["n_trades_next"].to_numpy() > threshold).astype(int)

    return binarize(train), binarize(val), binarize(test), threshold


def arrays(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(cols).to_numpy()


# ── Fitting and evaluation ────────────────────────────────────────────────────


def fit_regression(
    name: str,
    feature_cols: list[str],
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    y_tr: np.ndarray,
    y_v: np.ndarray,
    y_te: np.ndarray,
) -> tuple[dict, object, StandardScaler]:
    X_tr = arrays(train, feature_cols)
    X_v = arrays(val, feature_cols)
    X_te = arrays(test, feature_cols)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_v_s, X_te_s = (
        scaler.transform(X_tr),
        scaler.transform(X_v),
        scaler.transform(X_te),
    )

    model = Ridge(alpha=RIDGE_ALPHA) if "Ridge" in name else LinearRegression()
    model.fit(X_tr_s, y_tr)

    results = {}
    for split_name, X_s, y in [
        ("train", X_tr_s, y_tr),
        ("val", X_v_s, y_v),
        ("test", X_te_s, y_te),
    ]:
        y_hat = model.predict(X_s)
        results[split_name] = dict(
            r2=r2_score(y, y_hat),
            rmse=np.sqrt(mean_squared_error(y, y_hat)),
            ic=_pearson(y_hat, y),
        )
    return results, model, scaler


def fit_classification(
    name: str,
    feature_cols: list[str],
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    y_tr: np.ndarray,
    y_v: np.ndarray,
    y_te: np.ndarray,
) -> tuple[dict, object, StandardScaler]:
    X_tr = arrays(train, feature_cols)
    X_v = arrays(val, feature_cols)
    X_te = arrays(test, feature_cols)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_v_s, X_te_s = (
        scaler.transform(X_tr),
        scaler.transform(X_v),
        scaler.transform(X_te),
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr_s, y_tr)

    results = {}
    for split_name, X_s, y in [
        ("train", X_tr_s, y_tr),
        ("val", X_v_s, y_v),
        ("test", X_te_s, y_te),
    ]:
        y_prob = model.predict_proba(X_s)[:, 1]
        results[split_name] = dict(
            auc=roc_auc_score(y, y_prob),
            acc=accuracy_score(y, model.predict(X_s)),
        )
    return results, model, scaler


# ── Console output ────────────────────────────────────────────────────────────


def print_results(
    reg_results: dict,
    clf_results: dict,
    threshold: float,
) -> None:
    w = 62
    print(f"\n{'─' * w}")
    print(f"  REGRESSION — fwd_return (horizon={HORIZON} min)")
    print(f"{'─' * w}")
    print(f"  {'Model':<22} {'Split':<7} {'R²':>8}  {'IC':>8}  {'RMSE':>12}")
    print(f"  {'─' * 22} {'─' * 7} {'─' * 8}  {'─' * 8}  {'─' * 12}")
    for model_name, splits in reg_results.items():
        for i, (split_name, m) in enumerate(splits.items()):
            label = model_name if i == 0 else ""
            print(
                f"  {label:<22} {split_name:<7} {m['r2']:>+8.5f}  "
                f"{m['ic']:>+8.4f}  {m['rmse']:>12.6f}"
            )
        print(f"  {'':22}")

    print(f"{'─' * w}")
    print(f"  CLASSIFICATION — active_next  (n_trades[t+1] > {threshold:.0f})")
    print(f"{'─' * w}")
    print(f"  {'Model':<22} {'Split':<7} {'AUC':>8}  {'Accuracy':>8}")
    print(f"  {'─' * 22} {'─' * 7} {'─' * 8}  {'─' * 8}")
    for model_name, splits in clf_results.items():
        for i, (split_name, m) in enumerate(splits.items()):
            label = model_name if i == 0 else ""
            print(f"  {label:<22} {split_name:<7} {m['auc']:>8.4f}  {m['acc']:>8.4f}")
        print(f"  {'':22}")
    print(f"{'─' * w}\n")


# ── Figure ────────────────────────────────────────────────────────────────────


def plot(
    reg_results: dict,
    clf_results: dict,
    reg_models: dict,
    reg_scalers: dict,
    clf_models: dict,
    clf_scalers: dict,
    test: pl.DataFrame,
    y_te_clf: np.ndarray,
    out_path: str = "data/output/baseline_models.png",
) -> None:
    apply_figure_defaults()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Baseline Model Suite  ·  QBTS  ·  freq={FREQ}  ·  horizon={HORIZON} min",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
        y=0.998,
    )
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    splits = ["train", "val", "test"]
    n_splits = len(splits)

    # ── Panel A: IC by regression model ──────────────────────────────────────
    reg_names = list(reg_results.keys())
    n_reg = len(reg_names)
    width = 0.22
    x_reg = np.arange(n_reg)

    for j, split_name in enumerate(splits):
        ic_vals = [reg_results[m][split_name]["ic"] for m in reg_names]
        ax_a.bar(
            x_reg + (j - 1) * width,
            ic_vals,
            width,
            color=SPLIT_COLORS[split_name],
            alpha=0.85,
            label=split_name,
            edgecolor="white",
            linewidth=0.5,
        )

    ax_a.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax_a.set_xticks(x_reg)
    ax_a.set_xticklabels(reg_names, fontsize=FONT_SMALL)
    ax_a.set_ylabel("IC  (Pearson r)", fontsize=FONT_AXIS)
    ax_a.set_title(
        "A  ·  Regression IC by Model",
        fontsize=FONT_TITLE,
        fontweight="bold",
        loc="left",
    )
    ax_a.legend(fontsize=FONT_SMALL, frameon=False)
    style_ax(ax_a)

    # ── Panel B: Standardized coefficients (M2 and M3) ───────────────────────
    coef_models = {k: v for k, v in reg_models.items() if k != "M1 OLS (uni)"}
    coef_scalers = {k: v for k, v in reg_scalers.items() if k != "M1 OLS (uni)"}
    feat_labels = REG_MULTI

    coef_data = {}
    for name, model in coef_models.items():
        coef_data[name] = model.coef_

    n_feats = len(feat_labels)
    x_coef = np.arange(n_feats)
    coef_names = list(coef_data.keys())
    coef_colors = [BLUE, ORANGE]
    width_c = 0.35

    for j, (name, coefs) in enumerate(coef_data.items()):
        ax_b.bar(
            x_coef + (j - 0.5) * width_c,
            coefs,
            width_c,
            color=coef_colors[j],
            alpha=0.85,
            label=name,
            edgecolor="white",
            linewidth=0.5,
        )

    ax_b.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax_b.set_xticks(x_coef)
    ax_b.set_xticklabels(feat_labels, fontsize=FONT_SMALL)
    ax_b.set_ylabel("Standardized coefficient", fontsize=FONT_AXIS)
    ax_b.set_title(
        "B  ·  Regression Coefficients  (M2 vs M3)",
        fontsize=FONT_TITLE,
        fontweight="bold",
        loc="left",
    )
    ax_b.legend(fontsize=FONT_SMALL, frameon=False)
    style_ax(ax_b)

    # ── Panel C: AUC by classification model ─────────────────────────────────
    clf_names = list(clf_results.keys())
    n_clf = len(clf_names)
    x_clf = np.arange(n_clf)

    for j, split_name in enumerate(splits):
        auc_vals = [clf_results[m][split_name]["auc"] for m in clf_names]
        ax_c.bar(
            x_clf + (j - 1) * width,
            auc_vals,
            width,
            color=SPLIT_COLORS[split_name],
            alpha=0.85,
            label=split_name,
            edgecolor="white",
            linewidth=0.5,
        )

    ax_c.axhline(0.5, color=GRAY, lw=0.8, ls="--", alpha=0.6, label="random")
    ax_c.set_ylim(0.45, 1.0)
    ax_c.set_xticks(x_clf)
    ax_c.set_xticklabels(clf_names, fontsize=FONT_SMALL)
    ax_c.set_ylabel("AUC", fontsize=FONT_AXIS)
    ax_c.set_title(
        "C  ·  Classification AUC by Model",
        fontsize=FONT_TITLE,
        fontweight="bold",
        loc="left",
    )
    ax_c.legend(fontsize=FONT_SMALL, frameon=False)
    style_ax(ax_c)

    # ── Panel D: ROC curves on test set ───────────────────────────────────────
    clf_plot_colors = [TEAL, ORANGE]
    for (name, model), scaler, color in zip(
        clf_models.items(), clf_scalers.values(), clf_plot_colors
    ):
        feat_cols = CLF_BASE if "M4" in name else CLF_LAG
        X_te_s = scaler.transform(arrays(test, feat_cols))
        y_prob = model.predict_proba(X_te_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_te_clf, y_prob)
        auc_val = clf_results[name]["test"]["auc"]
        ax_d.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC={auc_val:.3f}")

    ax_d.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="Random")
    ax_d.set_xlabel("False Positive Rate", fontsize=FONT_AXIS)
    ax_d.set_ylabel("True Positive Rate", fontsize=FONT_AXIS)
    ax_d.set_title(
        "D  ·  ROC Curves on Test Set  (active_next)",
        fontsize=FONT_TITLE,
        fontweight="bold",
        loc="left",
    )
    ax_d.legend(fontsize=FONT_SMALL, frameon=False)
    style_ax(ax_d)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    df = build_dataset()
    train, val, test = split_by_date(df)

    print(f"Split sizes  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    y_tr_reg = train["fwd_return"].to_numpy()
    y_v_reg = val["fwd_return"].to_numpy()
    y_te_reg = test["fwd_return"].to_numpy()

    y_tr_clf, y_v_clf, y_te_clf, threshold = make_active_label(train, val, test)
    print(f"Activity threshold (train median n_trades): {threshold:.0f}")

    # ── Regression ─────────────────────────────────────────────────────────────
    reg_results, reg_models, reg_scalers = {}, {}, {}

    for name, feats in [
        ("M1 OLS (uni)", REG_UNI),
        ("M2 OLS (multi)", REG_MULTI),
        ("M3 Ridge", REG_MULTI),
    ]:
        print(f"Fitting {name}...")
        r, m, s = fit_regression(
            name, feats, train, val, test, y_tr_reg, y_v_reg, y_te_reg
        )
        reg_results[name], reg_models[name], reg_scalers[name] = r, m, s

    # ── Classification ─────────────────────────────────────────────────────────
    clf_results, clf_models, clf_scalers = {}, {}, {}

    for name, feats in [
        ("M4 Logistic", CLF_BASE),
        ("M5 Logistic+lags", CLF_LAG),
    ]:
        print(f"Fitting {name}...")
        r, m, s = fit_classification(
            name, feats, train, val, test, y_tr_clf, y_v_clf, y_te_clf
        )
        clf_results[name], clf_models[name], clf_scalers[name] = r, m, s

    print_results(reg_results, clf_results, threshold)

    plot(
        reg_results,
        clf_results,
        reg_models,
        reg_scalers,
        clf_models,
        clf_scalers,
        test,
        y_te_clf,
    )


if __name__ == "__main__":
    main()
