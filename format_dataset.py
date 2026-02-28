import os
import glob
import argparse
from pathlib import Path

import databento as db
import polars as pl


def find_files(data_dir: str) -> list[str]:
    # Only match the real payload files, not __MACOSX
    pattern = os.path.join(data_dir, "xnas-itch-*.mbo.dbn.zst")
    files = sorted(glob.glob(pattern))
    return files


def load_files(files: list[str]) -> pl.DataFrame:
    dfs = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Loading {f}")
        store = db.DBNStore.from_file(f)
        dfs.append(pl.from_pandas(store.to_df()))
    return pl.concat(dfs)


def main():
    parser = argparse.ArgumentParser(
        description="Load Databento DBN (.dbn.zst) files and save as Parquet/CSV"
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing xnas-itch-*.mbo.dbn.zst files (e.g. data/XNAS-20260227-QVD7UYV7GQ)",
    )
    parser.add_argument(
        "--out_dir",
        default="formatted",
        help="Where to save the output (default: formatted/)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format (default: parquet)",
    )
    parser.add_argument(
        "--name",
        default="xnas_itch_mbo",
        help="Base output filename without extension (default: xnas_itch_mbo)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    files = find_files(str(data_dir))
    if not files:
        raise FileNotFoundError(
            f"No files matched xnas-itch-*.mbo.dbn.zst in: {data_dir}\n"
            f"Tip: point data_dir at the extracted folder, not the .zip."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_files(files)

    out_path = out_dir / f"{args.name}.{args.format}"
    print(f"Saving {len(df):,} rows to {out_path}")

    if args.format == "parquet":
        df.write_parquet(out_path)
    else:
        df.write_csv(out_path)

    print("Done ✅")


if __name__ == "__main__":
    main()
