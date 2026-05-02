# Sentinel Group Project — Theory of Machine Learning

Predicting short-term equity returns and market activity from NASDAQ ITCH market-by-order (MBO) data using order-flow imbalance (OFI) signals and reinforcement learning.

Data source: **QBTS (D-Wave Quantum)** — NASDAQ ITCH MBO feed via Databento, October 2025 (~32 trading days, 46.8M events).

---

## Project Structure

```
.
├── data/
│   ├── raw/                        # Downloaded zip + extracted .dbn.zst files
│   ├── formatted/
│   │   └── xnas_itch_mbo.parquet   # Processed MBO dataset (46.8M rows, 14 cols)
│   └── output/                     # Generated figures and CSVs
│
├── src/
│   ├── theme.py                    # Shared matplotlib theme (colors, fonts, helpers)
│   ├── data_utils.py               # Data loading, OFI feature engineering, train/val/test splits
│   ├── data_overview.py            # Slide figure: schema card + event distribution + price series
│   ├── market_activity.py          # Slide figure: 4-panel market activity deep-dive
│   ├── mbo_explainer.py            # Slide figure: synthetic order lifecycle diagram
│   ├── ofi_analysis.py             # OFI quantification + univariate linear regression
│   ├── baseline_models.py          # OLS / Ridge regression + logistic classifiers (M1–M5)
│   ├── coefficient_story.py        # Standardized coefficient comparison (direction vs activity)
│   └── rf_classifier.py            # Random Forest price-direction classifier (3-class)
│
├── rl_analysis/
│   ├── market_maker/               # RL market-making environment + training pipeline
│   │   ├── env.py                  # Gymnasium environment (1-second steps, 25 actions)
│   │   ├── baseline.py             # Avellaneda-Stoikov analytical baseline + grid search
│   │   ├── data_preprocessor.py    # MBO .dbn.zst → 1-second feature snapshots
│   │   ├── train.py                # DQN / PPO training with stable-baselines3
│   │   ├── evaluate_multi.py       # Multi-episode statistical evaluation
│   │   ├── ablation.py             # Feature ablation study
│   │   └── run_pipeline.py         # End-to-end orchestration script
│   ├── models/                     # Trained agent checkpoints (gitignored)
│   ├── rl_results/                 # Result plots and CSVs (gitignored)
│   ├── snapshots/                  # Preprocessed 1-second MBO snapshots
│   └── rl_market_maker.ipynb       # RL analysis notebook
│
├── format_dataset.py               # Converts raw .dbn.zst files → parquet
├── data_exploration.ipynb          # Interactive EDA notebook
├── pyproject.toml
└── README.md
```

---

## Setup

We use **uv** for Python environments (Python 3.12).

```bash
uv venv --python 3.12
uv sync
```

Run any script from the project root with:

```bash
uv run python src/<script>.py
```

---

## Getting the Dataset

1. Download the Databento export zip and place it in `data/raw/`
2. Unzip:

```bash
unzip data/raw/XNAS-*.zip -d data/raw/
```

3. Convert to parquet:

```bash
uv run python format_dataset.py data/raw/XNAS-<ID> --out_dir data/formatted
```

---

## Analysis Pipeline

### 1. Data utilities (`src/data_utils.py`)

Core feature pipeline shared by all downstream scripts. Import directly:

```python
from data_utils import load_splits

train, val, test = load_splits(freq="1m", horizon=5)
```

| Function | Description |
|---|---|
| `load_raw()` | Lazy-scans the parquet, filters to regular session (9:30–4 PM ET), drops resets and zero-size rows |
| `build_ofi_features(df, freq)` | Aggregates into time buckets; computes OFI, trade OFI, VWAP, cancel features, order lifespan |
| `add_forward_returns(features, horizon)` | Appends `fwd_return = log(vwap[t+h] / vwap[t])` as the regression target |
| `split_by_date(df)` | Hard time-ordered split — no shuffling, no look-ahead leakage |

**Default splits** (Oct 2025 data):

| Split | Dates | ~Share |
|---|---|---|
| Train | Oct 1 – Oct 21 | 65% |
| Val | Oct 22 – Oct 26 | 15% |
| Test | Oct 27 – Oct 31 | 20% |

**OFI definition:**

```
ofi = (buy_vol − sell_vol) / (buy_vol + sell_vol)   ∈ [−1, +1]
```

`buy_vol` / `sell_vol` are summed sizes of limit-order Add events on the bid / ask side within each bucket.

---

### 2. OFI analysis (`src/ofi_analysis.py`)

Quantifies OFI and fits a univariate linear model. Outputs console stats and a 4-panel figure.

```bash
uv run python src/ofi_analysis.py
```

**Output — `data/output/ofi_analysis.png`:**
- **A** OFI distribution
- **B** OFI time series with 30-bucket rolling mean
- **C** Binned OFI vs mean forward return (bps) with OLS fit
- **D** Daily information coefficient (Pearson r per trading day)

---

### 3. Baseline models (`src/baseline_models.py`)

Fits five linear models on the full feature block and evaluates across all splits.

```bash
uv run python src/baseline_models.py
```

**Regression (predicting `fwd_return`):**
- M1 — OLS, univariate (OFI only)
- M2 — OLS, multivariate (OFI + trade-OFI + n_trades + cancel block + lifespan)
- M3 — Ridge (same features as M2, α = 1.0)

**Classification (predicting `active_next`):**
- M4 — Logistic regression
- M5 — Logistic regression + OFI lags (lag 1, lag 2)

**Output — `data/output/baseline_models.png`:** IC by model, coefficient comparison (M2 vs M3), AUC by model, ROC curves.

---

### 4. Coefficient story (`src/coefficient_story.py`)

Plots standardized coefficients from M3 (Ridge, direction) and M5 (Logistic, activity) side-by-side to show which features drive each problem.

```bash
uv run python src/coefficient_story.py
```

**Output — `data/output/coefficient_story.png`**

---

### 5. Random Forest classifier (`src/rf_classifier.py`)

Predicts price direction (Up / Flat / Down) from rolling event-level MBO features.

```bash
uv run python src/rf_classifier.py
```

**Features** (rolling windows of 10 / 30 / 100 events): price stats, volume, OFI, trade ratio, momentum, intraday time.

**Label:** 3-class direction 10 events ahead (threshold ± 0.01%).

**Model:** Random Forest (300 trees, `max_depth=12`, `class_weight="balanced"`).

**Outputs (`data/output/`):** `fig1_` through `fig12_` — label distribution, confusion matrix, feature importance, ROC/PR curves, CV scores, calibration, permutation importance, hyperparameter sensitivity, learning curve.

---

### 6. Slide figures (`src/`)

All figures use the shared theme in `src/theme.py` and save to `data/output/`.

```bash
uv run python src/data_overview.py
uv run python src/market_activity.py
uv run python src/mbo_explainer.py
```

| Script | Output | Description |
|---|---|---|
| `data_overview.py` | `data_overview.png` | Schema card, event-type donut, daily close price |
| `market_activity.py` | `market_activity.png` | Daily volume, intraday profile, order size distribution, action×side heatmap |
| `mbo_explainer.py` | `mbo_explainer.png` | Synthetic order lifecycle diagram |

---

## RL Market-Making (`rl_analysis/`)

A reinforcement learning agent trained to act as a market maker, posting bids and asks to earn the spread while managing inventory risk.

**Environment:** `market_maker/env.py` — Gymnasium env with 1-second decision steps and 25 discrete actions (combinations of bid/ask offsets). Reward is realized PnL minus an inventory penalty.

**Baseline:** Avellaneda-Stoikov analytical model (`market_maker/baseline.py`) with grid-search over risk-aversion and spread parameters.

**Agents:** DQN and PPO trained via stable-baselines3 (`market_maker/train.py`).

### Running the pipeline

The pipeline runs from within `rl_analysis/` so that relative paths (`models/`, `rl_results/`, `snapshots/`) resolve correctly.

```bash
cd rl_analysis
uv run python -m market_maker.run_pipeline   # full end-to-end
uv run python -m market_maker.train --algo dqn --timesteps 100000
uv run python -m market_maker.evaluate_multi
uv run python -m market_maker.ablation
```

Pre-trained agent checkpoints are in `rl_analysis/models/` (gitignored). Results and plots are written to `rl_analysis/rl_results/` (gitignored).

The interactive notebook `rl_analysis/rl_market_maker.ipynb` walks through the full analysis.

> **Note:** `run_pipeline.py` step 1 (preprocessing raw `.dbn.zst` → snapshots) expects the raw data directory to be present. Steps 2–4 (baseline, DQN, PPO evaluation) work with the pre-built snapshots in `rl_analysis/snapshots/`.

---

## Questions

Reach out in the group channel on Canvas if you run into issues with the data or environment.
