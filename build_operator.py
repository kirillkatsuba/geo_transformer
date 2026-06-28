from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .operators import operator_from_intersections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized sparse assay/block operator table from intersections."
    )
    parser.add_argument("--intersections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-col", required=True, help="Interval id or block id column")
    parser.add_argument("--node-col", default="node_id")
    parser.add_argument(
        "--measure-col",
        required=True,
        help="Intersection length/volume/weight column before normalization",
    )
    parser.add_argument("--operator-id-col", default="operator_id")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    intersections = read_table(args.intersections)
    operator_df = operator_from_intersections(
        intersections,
        operator_col=args.operator_col,
        node_col=args.node_col,
        length_col=args.measure_col,
        operator_id_col=args.operator_id_col,
    )
    write_table(operator_df, args.output)
    print(f"Saved normalized operator table: {args.output}")
    print(f"rows={len(operator_df)} operators={operator_df[args.operator_id_col].nunique()}")


if __name__ == "__main__":
    main()

