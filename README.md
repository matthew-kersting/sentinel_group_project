# Sentinel Group Project — Theory of Machine Learning

This repository contains our full data analysis pipeline and modeling work for the Theory of Machine Learning course project.

We use NASDAQ ITCH (market-by-order) data from Databento and convert it into a format suitable for analysis and experiments.

---

## Project Structure

```
.
├── data/                  # Local datasets (ignored by git)
│   ├── raw/               # Downloaded zip + extracted files
│   └── formatted/         # Processed datasets (parquet/csv)
│
├── format_dataset.py      # Script to convert raw data → tabular dataset
├── notebooks/             # Analysis notebooks (optional)
├── src/                   # Models / utilities (optional)
├── README.md
└── .gitignore
```

The `data/` folder is included in the repo, but its contents are ignored so large files are not committed.

---

## Setup (Python Environment)

We use **uv** for Python environments.

Install dependencies:

```bash
uv venv --python 3.12
uv pip install databento pandas pyarrow zstandard
```

Run scripts with:

```bash
uv run python <script>.py
```

No activation needed.

---

## Getting the Dataset

1. Download the Databento export zip.
2. Place it inside `data/raw/`
3. Unzip:

```bash
unzip data/raw/XNAS-*.zip -d data/raw/
```

You should end up with a folder containing files like:

```
xnas-itch-YYYYMMDD.mbo.dbn.zst
```

Ignore the `__MACOSX` folder if present.

---

## Formatting the Data

Convert the raw files into a single dataset:

```bash
uv run python format_dataset.py data/raw/XNAS-XXXXXXXX --out_dir data/formatted
```

Output will be:

```
data/formatted/xnas_itch_mbo.parquet
```

Parquet is recommended because it loads much faster than CSV.

---

## Goal of This Repo

This repository will contain:

* Data ingestion and preprocessing
* Exploratory analysis
* Machine learning models
* Experiments and results for the course project

---

## Questions

If you run into issues setting up the environment or loading data, reach out in the group channel in Canvas.
