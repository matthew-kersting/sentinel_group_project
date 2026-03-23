# Sentinel Group Project — Theory of Machine Learning

Predicting short-term equity returns from NASDAQ ITCH market-by-order (MBO) data using order-flow imbalance (OFI) signals.

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
│   └── ofi_analysis.py             # OFI quantification + linear regression model
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

Run any script with:

```bash
cd src && uv run python <script>.py
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

## Pipeline

### 1. Data utilities (`src/data_utils.py`)

The core feature pipeline. Import and use directly:

```python
from data_utils import load_splits

train, val, test = load_splits(freq="1m", horizon=5)
```

**Steps:**

| Function | Description |
|---|---|
| `load_raw()` | Lazy-scans the parquet (5 columns only), filters to regular session (9:30–4 PM ET), drops resets and zero-size rows |
| `build_ofi_features(df, freq)` | Aggregates into time buckets; computes OFI, trade OFI, VWAP, cancel ratio |
| `add_forward_returns(features, horizon)` | Appends `fwd_return = log(vwap[t+h] / vwap[t])` as the prediction target |
| `split_by_date(df)` | Hard time-ordered split — no shuffling, no look-ahead leakage |

**Default splits** (based on Oct 2025 data):

| Split | Dates | ~Share |
|---|---|---|
| Train | Oct 1 – Oct 21 | 65% |
| Val | Oct 22 – Oct 26 | 15% |
| Test | Oct 27 – Oct 31 | 20% |

**OFI definition:**

```
ofi = (buy_vol − sell_vol) / (buy_vol + sell_vol)
```

Where `buy_vol` / `sell_vol` are the summed sizes of limit-order Add events on the bid / ask side within each time bucket. Bounded in [−1, +1].

---

### 2. OFI analysis (`src/ofi_analysis.py`)

Quantifies OFI and fits a simple linear model. Run from `src/`:

```bash
uv run python ofi_analysis.py
```

**Outputs:**
- Console: OFI summary stats + linear regression table (α, β, t-stat, p-value, R², RMSE, IC across all three splits)
- `data/output/ofi_analysis.png` — 4-panel figure:
  - **A** OFI distribution
  - **B** OFI time series with rolling mean
  - **C** Binned OFI vs mean forward return (bps), with OLS fit line
  - **D** Daily information coefficient (Pearson corr per day)

**Model:** `fwd_return = α + β·OFI` fitted on train, evaluated on val and test.

---

### 3. Slide figures

All figures use the shared theme in `src/theme.py` and save to `data/output/`.

| Script | Output | Description |
|---|---|---|
| `data_overview.py` | `data_overview.png` | Schema card, event-type donut, daily close price |
| `market_activity.py` | `market_activity.png` | Daily volume, intraday profile, order size distribution, action×side heatmap |
| `mbo_explainer.py` | `mbo_explainer.png` | Synthetic order lifecycle diagram (illustrative, no real data) |

---

## Questions

Reach out in the group channel on Canvas if you run into issues with the data or environment.
