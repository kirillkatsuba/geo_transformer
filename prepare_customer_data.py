from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import (
    CustomerDataPaths,
    build_center_node_table,
    build_north_node_table,
    read_customer_data,
    summarize_loaded_data,
)
from .features import GeoFeatureBuilder, attach_feature_matrix
from .splits import add_train_eval_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and standardize customer assay/CEN/NTH datasets."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("geo_transformer/prepared"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = CustomerDataPaths(
        assays_xlsx=args.root / "Вся_химия+литология+Au_final_all_data.XLSX",
        center_blocks_csv=args.root / "md_nat250721_CEN_Отработано.csv",
        north_blocks_csv=args.root / "md_nat241227(Модель_ресурсов_ind,inf).csv",
    )
    data = read_customer_data(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data.assays.to_parquet(args.output_dir / "assays_standardized.parquet", index=False)
    data.center_blocks.to_parquet(args.output_dir / "center_blocks_standardized.parquet", index=False)
    data.north_blocks.to_parquet(args.output_dir / "north_blocks_standardized.parquet", index=False)

    center_nodes = build_center_node_table(data.center_blocks)
    north_nodes = build_north_node_table(data.north_blocks)
    center_nodes, north_nodes = add_train_eval_flags(center_nodes, north_nodes)

    # Fit the feature schema on all known inference-time covariates so target
    # domain categories such as NTH are represented. This does not use northern
    # target values as labels; it only fixes the covariate encoding space.
    feature_schema_nodes = pd.concat([center_nodes, north_nodes], ignore_index=True)
    feature_builder = GeoFeatureBuilder()
    feature_builder.fit(feature_schema_nodes)
    center_features = feature_builder.transform(center_nodes)
    north_features = feature_builder.transform(north_nodes)

    center_nodes = attach_feature_matrix(center_nodes, center_features)
    north_nodes = attach_feature_matrix(north_nodes, north_features)

    center_nodes.to_parquet(args.output_dir / "center_nodes.parquet", index=False)
    north_nodes.to_parquet(args.output_dir / "north_nodes.parquet", index=False)

    feature_columns = [f"feat_{col}" for col in feature_builder.output_columns_]
    (args.output_dir / "feature_columns.txt").write_text("\n".join(feature_columns) + "\n")

    summary = summarize_loaded_data(data)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nPrepared node features: {len(feature_columns)} columns")
    print(f"Center nodes: {len(center_nodes)}")
    print(f"North known/eval nodes: {int(north_nodes['has_targets'].sum())}")
    print(f"North unknown/prediction nodes: {int((~north_nodes['has_targets']).sum())}")


if __name__ == "__main__":
    main()
