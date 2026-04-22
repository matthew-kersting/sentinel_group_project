# ============================================================
# Random Forest Price Direction Prediction
# Based on NASDAQ ITCH MBO Order Flow Data
# Final Project - Machine Learning Theory
# ============================================================

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.inspection import permutation_importance

import polars as pl
from pathlib import Path

# ── Global plot style ────────────────────────────────────────
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]

OUT_DIR = Path(__file__).parent.parent / "data" / "output"

# ============================================================
# 1. Data Loading
# ============================================================
DATA_PATH = (
    Path(__file__).parent.parent / "data" / "formatted" / "xnas_itch_mbo.parquet"
)

MAX_FILES = 3  # number of daily files to simulate
CHUNK_ROWS = 50000  # rows per file

df_raw = pl.scan_parquet(DATA_PATH).head(MAX_FILES * CHUNK_ROWS).collect().to_pandas()
df_raw["symbol"] = "QBTS"
df_raw = df_raw.sort_values("ts_event").reset_index(drop=True)
print(f"\nRaw data shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
print(df_raw.head(3))


# ============================================================
# 2. Feature Engineering
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from raw MBO order-flow data.
    Each row is one order event; price is an integer (divide by 1e9 for USD).
    """
    data = df.copy()

    # ── 2.1 Basic field cleaning ──────────────────────────────
    data["price_usd"] = data["price"] / 1e9
    data["ts_event"] = pd.to_datetime(data["ts_event"], utc=True)

    # ── 2.2 Order direction encoding ─────────────────────────
    # side: 'A'=ask (sell), 'B'=bid (buy), 'N'=none
    side_map = {"A": -1, "B": 1, "N": 0}
    data["side_code"] = data["side"].map(side_map).fillna(0)

    # action: 'A'=add, 'C'=cancel, 'M'=modify, 'T'=trade, 'F'=fill, 'R'=reset
    action_map = {"A": 1, "C": -1, "M": 0, "T": 2, "F": 2, "R": -2}
    data["action_code"] = data["action"].map(action_map).fillna(0)

    # ── 2.3 Rolling features grouped by symbol ────────────────
    feat_list = []

    for sym, grp in data.groupby("symbol"):
        grp = grp.sort_index().copy()
        p = grp["price_usd"]
        s = grp["size"]
        sc = grp["side_code"]
        ac = grp["action_code"]

        for w in [10, 30, 100]:
            # Rolling price statistics
            grp[f"price_mean_{w}"] = p.rolling(w, min_periods=1).mean()
            grp[f"price_std_{w}"] = p.rolling(w, min_periods=1).std().fillna(0)
            grp[f"price_min_{w}"] = p.rolling(w, min_periods=1).min()
            grp[f"price_max_{w}"] = p.rolling(w, min_periods=1).max()
            grp[f"price_range_{w}"] = grp[f"price_max_{w}"] - grp[f"price_min_{w}"]

            # Rolling volume statistics
            grp[f"size_sum_{w}"] = s.rolling(w, min_periods=1).sum()
            grp[f"size_mean_{w}"] = s.rolling(w, min_periods=1).mean()

            # Order Flow Imbalance (buy vol - sell vol) / total vol
            buy_vol = (s * (sc == 1)).rolling(w, min_periods=1).sum()
            sell_vol = (s * (sc == -1)).rolling(w, min_periods=1).sum()
            grp[f"ofi_{w}"] = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)

            # Trade event ratio
            is_trade = (ac == 2).astype(int)
            grp[f"trade_ratio_{w}"] = is_trade.rolling(w, min_periods=1).mean()

        # Price momentum (returns)
        for lag in [1, 5, 20]:
            grp[f"ret_{lag}"] = p.pct_change(lag).fillna(0)

        # Price deviation from rolling mean (mean-reversion signal)
        grp["price_dev_10"] = (p - grp["price_mean_10"]) / (grp["price_std_10"] + 1e-9)
        grp["price_dev_30"] = (p - grp["price_mean_30"]) / (grp["price_std_30"] + 1e-9)
        grp["price_dev_100"] = (p - grp["price_mean_100"]) / (
            grp["price_std_100"] + 1e-9
        )

        # Time features
        grp["hour"] = grp["ts_event"].dt.hour
        grp["minute"] = grp["ts_event"].dt.minute
        # Market session: pre-market / regular / after-hours
        grp["session"] = pd.cut(
            grp["hour"], bins=[0, 9, 16, 24], labels=[0, 1, 2], right=False
        ).astype(float)

        feat_list.append(grp)

    result = pd.concat(feat_list).sort_index()
    return result


print("\nBuilding features (may take 1-2 minutes)...")
df_feat = build_features(df_raw)
print(f"Feature engineering complete. Shape: {df_feat.shape}")

# ============================================================
# 3. Label Construction (prediction target)
# ============================================================
HORIZON = 10  # predict price direction N events ahead
THRESHOLD = 0.0001  # 0.01% threshold to distinguish up/flat/down

label_list = []
for sym, grp in df_feat.groupby("symbol"):
    grp = grp.sort_index().copy()
    future_price = grp["price_usd"].shift(-HORIZON)
    ret_future = (future_price - grp["price_usd"]) / (grp["price_usd"] + 1e-9)

    # Three-class label: 1=up, 0=flat, -1=down
    label = np.where(
        ret_future > THRESHOLD, 1, np.where(ret_future < -THRESHOLD, -1, 0)
    )
    grp["label"] = label
    label_list.append(grp)

df_feat = pd.concat(label_list).sort_index()
df_feat = df_feat.dropna(subset=["label"])
df_feat["label"] = df_feat["label"].astype(int)

print("\nLabel distribution:")
print(
    df_feat["label"].value_counts().rename({1: "Up(+1)", 0: "Flat(0)", -1: "Down(-1)"})
)

# ============================================================
# 4. Train / Test Split (chronological, no data leakage)
# ============================================================
FEATURE_COLS = [
    c
    for c in df_feat.columns
    if any(
        c.startswith(p)
        for p in [
            "price_mean",
            "price_std",
            "price_min",
            "price_max",
            "price_range",
            "size_sum",
            "size_mean",
            "ofi_",
            "trade_ratio_",
            "ret_",
            "price_dev",
            "hour",
            "minute",
            "session",
            "side_code",
            "action_code",
            "size",
        ]
    )
]
FEATURE_COLS = list(set(FEATURE_COLS))
print(f"\nNumber of features: {len(FEATURE_COLS)}")

df_model = df_feat[FEATURE_COLS + ["label"]].dropna()
df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

# Chronological 80/20 split (simulates real trading conditions)
split_idx = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:]

X_train = train_df[FEATURE_COLS].values
y_train = train_df["label"].values
X_test = test_df[FEATURE_COLS].values
y_test = test_df["label"].values

print(f"\nTrain set: {X_train.shape[0]} rows  |  Test set: {X_test.shape[0]} rows")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 5. Train Random Forest Model
# ============================================================
print("\nTraining Random Forest (may take 1-3 minutes)...")

rf = RandomForestClassifier(
    n_estimators=300,  # number of trees
    max_depth=12,  # maximum tree depth
    min_samples_leaf=20,  # minimum samples per leaf (regularization)
    max_features="sqrt",  # features per split = sqrt(n_features)
    class_weight="balanced",  # handle class imbalance
    n_jobs=-1,  # use all CPU cores
    random_state=42,
)
rf.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# 6. Model Evaluation
# ============================================================
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)
classes = rf.classes_

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["Down(-1)", "Flat(0)", "Up(+1)"]
    )
)

# ============================================================
# 7. Cross-Validation
# ============================================================
print("\nRunning 5-fold cross-validation on training set...")
cv = StratifiedKFold(n_splits=5, shuffle=False)
cv_scores = cross_val_score(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)
print(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ============================================================
# 8. Plots
# ============================================================
class_labels = {-1: "Down(-1)", 0: "Flat(0)", 1: "Up(+1)"}

# ── Figure 1: Label Distribution ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
label_counts = pd.Series(y_train).value_counts().sort_index()
bars = ax.bar(
    ["Down(-1)", "Flat(0)", "Up(+1)"],
    label_counts.values,
    color=COLORS[:3],
    edgecolor="white",
    linewidth=0.8,
)
for bar, val in zip(bars, label_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{val:,}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.set_title("Training Set Label Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Sample Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_label_distribution.png", bbox_inches="tight")

# ── Figure 2: Confusion Matrix ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

sns.heatmap(
    cm_pct,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=["Down", "Flat", "Up"],
    yticklabels=["Down", "Flat", "Up"],
    linewidths=0.5,
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix (Row-Normalized %)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_confusion_matrix.png", bbox_inches="tight")

# ── Figure 3: Feature Importance Top 20 ──────────────────────
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
top20 = feat_imp.head(20)

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = [
    COLORS[0] if i < 5 else COLORS[2] if i < 10 else COLORS[3]
    for i in range(len(top20))
]
ax.barh(
    top20.index[::-1], top20.values[::-1], color=colors_bar[::-1], edgecolor="white"
)
ax.set_xlabel("Gini Importance", fontsize=12)
ax.set_title(
    "Random Forest Feature Importance - Top 20", fontsize=14, fontweight="bold"
)
ax.axvline(
    top20.mean(),
    color="red",
    linestyle="--",
    linewidth=1.2,
    label=f"Mean = {top20.mean():.4f}",
)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_feature_importance.png", bbox_inches="tight")

# ── Figure 4: ROC Curves (One-vs-Rest) ───────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for i, cls in enumerate(classes):
    y_bin = (y_test == cls).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(
        fpr,
        tpr,
        color=COLORS[i],
        lw=2,
        label=f"{class_labels[cls]}  AUC = {roc_auc:.3f}",
    )

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_roc_curve.png", bbox_inches="tight")

# ── Figure 5: Precision-Recall Curves ────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for i, cls in enumerate(classes):
    y_bin = (y_test == cls).astype(int)
    prec, rec, _ = precision_recall_curve(y_bin, y_pred_prob[:, i])
    ap = average_precision_score(y_bin, y_pred_prob[:, i])
    ax.plot(
        rec, prec, color=COLORS[i], lw=2, label=f"{class_labels[cls]}  AP = {ap:.3f}"
    )

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig5_pr_curve.png", bbox_inches="tight")

# ── Figure 6: Cross-Validation Fold Accuracy ─────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, 6), cv_scores, color=COLORS[3], edgecolor="white", linewidth=0.8)
ax.axhline(
    cv_scores.mean(),
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {cv_scores.mean():.4f}",
)
ax.fill_between(
    range(0, 7),
    cv_scores.mean() - cv_scores.std(),
    cv_scores.mean() + cv_scores.std(),
    alpha=0.15,
    color="red",
    label=f"+/-1 SD = {cv_scores.std():.4f}",
)
ax.set_xticks(range(1, 6))
ax.set_xlabel("Fold", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=14, fontweight="bold")
ax.set_ylim(max(0, cv_scores.min() - 0.05), min(1, cv_scores.max() + 0.05))
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_cv_scores.png", bbox_inches="tight")

# ── Figure 7: Predicted Probability Distributions ────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, (cls, lbl) in enumerate(class_labels.items()):
    idx = list(classes).index(cls)
    axes[i].hist(
        y_pred_prob[:, idx], bins=40, color=COLORS[i], edgecolor="white", linewidth=0.5
    )
    axes[i].set_title(f"{lbl} Predicted Probability", fontsize=11)
    axes[i].set_xlabel("Predicted Probability")
    axes[i].set_ylabel("Frequency")
plt.suptitle(
    "Predicted Probability Distributions by Class", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig7_prob_distribution.png", bbox_inches="tight")

# ── Figure 8: Calibration Curves ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, (cls, lbl) in enumerate(class_labels.items()):
    idx = list(classes).index(cls)
    probs = y_pred_prob[:, idx]
    correct = (y_test == cls).astype(int)

    bins = np.linspace(0, 1, 11)
    mids, acc_list = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            mids.append((lo + hi) / 2)
            acc_list.append(correct[mask].mean())

    axes[i].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect Calibration")
    axes[i].plot(mids, acc_list, "o-", color=COLORS[i], lw=2, label="Model")
    axes[i].set_title(f"{lbl} Calibration Curve", fontsize=11)
    axes[i].set_xlabel("Predicted Probability")
    axes[i].set_ylabel("Actual Accuracy")
    axes[i].legend(fontsize=9)
plt.suptitle("Confidence Calibration Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig8_calibration.png", bbox_inches="tight")

# ============================================================
# 9. Sensitivity Analysis (Permutation Importance)
# ============================================================
print("\nComputing permutation importance (sensitivity analysis)...")

perm_result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring="accuracy"
)

perm_imp_mean = pd.Series(perm_result.importances_mean, index=FEATURE_COLS)
perm_imp_std = pd.Series(perm_result.importances_std, index=FEATURE_COLS)
perm_top20 = perm_imp_mean.sort_values(ascending=False).head(20)
perm_std_top = perm_imp_std[perm_top20.index]

# ── Figure 9: Permutation Importance ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(
    perm_top20.index[::-1],
    perm_top20.values[::-1],
    xerr=perm_std_top.values[::-1],
    color=COLORS[1],
    edgecolor="white",
    capsize=3,
    linewidth=0.5,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Accuracy Drop (higher = more important)", fontsize=12)
ax.set_title(
    "Sensitivity Analysis: Permutation Feature Importance - Top 20",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig9_permutation_importance.png", bbox_inches="tight")

# ── Figure 10: Hyperparameter Sensitivity - n_estimators ─────
print("\nHyperparameter sensitivity: n_estimators...")
n_est_range = [10, 30, 50, 100, 200, 300]
train_accs, test_accs = [], []

for n in n_est_range:
    tmp_rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    tmp_rf.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, tmp_rf.predict(X_train)))
    test_accs.append(accuracy_score(y_test, tmp_rf.predict(X_test)))
    print(
        f"  n_estimators={n:3d}  train={train_accs[-1]:.4f}  test={test_accs[-1]:.4f}"
    )

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(n_est_range, train_accs, "o-", color=COLORS[0], lw=2, label="Train Accuracy")
ax.plot(n_est_range, test_accs, "s-", color=COLORS[1], lw=2, label="Test Accuracy")
ax.fill_between(
    n_est_range,
    train_accs,
    test_accs,
    alpha=0.15,
    color="gray",
    label="Overfitting Gap",
)
ax.set_xlabel("n_estimators (Number of Trees)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title(
    "Hyperparameter Sensitivity: Number of Trees vs. Accuracy",
    fontsize=14,
    fontweight="bold",
)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig10_hyperparam_sensitivity.png", bbox_inches="tight")

# ── Figure 11: Hyperparameter Sensitivity - max_depth ────────
print("\nHyperparameter sensitivity: max_depth...")
depth_range = [2, 4, 6, 8, 10, 12, 15, 20, None]
depth_labels = [str(d) for d in depth_range]
train_accs_d, test_accs_d = [], []

for d in depth_range:
    tmp_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=d,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    tmp_rf.fit(X_train, y_train)
    train_accs_d.append(accuracy_score(y_train, tmp_rf.predict(X_train)))
    test_accs_d.append(accuracy_score(y_test, tmp_rf.predict(X_test)))

fig, ax = plt.subplots(figsize=(8, 4))
x = range(len(depth_labels))
ax.plot(x, train_accs_d, "o-", color=COLORS[0], lw=2, label="Train Accuracy")
ax.plot(x, test_accs_d, "s-", color=COLORS[1], lw=2, label="Test Accuracy")
ax.fill_between(x, train_accs_d, test_accs_d, alpha=0.15, color="gray")
ax.set_xticks(x)
ax.set_xticklabels(depth_labels)
ax.set_xlabel("max_depth", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title(
    "Hyperparameter Sensitivity: Tree Depth vs. Accuracy",
    fontsize=14,
    fontweight="bold",
)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig11_depth_sensitivity.png", bbox_inches="tight")

# ── Figure 12: Learning Curve ─────────────────────────────────
print("\nGenerating learning curve...")
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    X_train,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    train_sizes,
    train_scores.mean(axis=1),
    "o-",
    color=COLORS[0],
    lw=2,
    label="Train Set",
)
ax.fill_between(
    train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1),
    alpha=0.15,
    color=COLORS[0],
)
ax.plot(
    train_sizes,
    test_scores.mean(axis=1),
    "s-",
    color=COLORS[1],
    lw=2,
    label="Validation Set",
)
ax.fill_between(
    train_sizes,
    test_scores.mean(axis=1) - test_scores.std(axis=1),
    test_scores.mean(axis=1) + test_scores.std(axis=1),
    alpha=0.15,
    color=COLORS[1],
)
ax.set_xlabel("Training Sample Size", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Learning Curve", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig12_learning_curve.png", bbox_inches="tight")

# ============================================================
# 10. Summary
# ============================================================
print("\n" + "=" * 55)
print("              Final Results Summary")
print("=" * 55)
print(f"  Test Set Accuracy          : {acc:.4f}")
print(
    f"  5-Fold CV Mean Accuracy    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
)

for i, cls in enumerate(classes):
    idx = list(classes).index(cls)
    y_bin = (y_test == cls).astype(int)
    roc_auc = auc(*roc_curve(y_bin, y_pred_prob[:, idx])[:2])
    print(f"  AUC ({class_labels[cls]:10s})       : {roc_auc:.4f}")

print("\n  Saved figures:")
for i in range(1, 13):
    print(f"    fig{i}_*.png")
print("=" * 55)
