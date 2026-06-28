from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import (
    CustomerDataPaths,
    add_pca_spatial_coordinates,
    attach_baselines,
    build_center_node_table,
    build_north_node_table,
    read_customer_data,
    summarize_loaded_data,
    transform_as_abs_zscore_div10,
)
from .features import GeoFeatureBuilder, attach_feature_matrix
from .knn_chemistry import KNNConfig, project_assay_chemistry_to_nodes
from .splits import add_train_eval_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and standardize customer assay/CEN/NTH datasets."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("geo_transformer/prepared"))
    parser.add_argument("--add-knn-chemistry", action="store_true")
    parser.add_argument("--knn-neighbors", type=int, default=16)
    parser.add_argument("--knn-power", type=float, default=2.0)
    parser.add_argument("--knn-batch-size", type=int, default=100000)
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Attach baseline prediction CSV by coordinates. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_baseline_specs(
    specs: list[str],
    root: Path,
    as_mean: float,
    as_std: float,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --baseline spec {spec!r}; expected NAME=PATH")
        name, path_text = spec.split("=", 1)
        path = Path(path_text)
        if not path.is_absolute():
            path = root / path
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        else:
            frame = pd.read_csv(path)
        rename = {}
        if {"EAST", "NORTH", "RL"}.issubset(frame.columns):
            rename.update({"EAST": "X", "NORTH": "Y", "RL": "Z"})
        if "CORG" in frame.columns and "CORG-1" not in frame.columns:
            rename["CORG"] = "CORG-1"
        frame = frame.rename(columns=rename)
        if "AS" in frame.columns:
            frame["AS_raw"] = pd.to_numeric(frame["AS"], errors="coerce")
            frame["AS"], _, _ = transform_as_abs_zscore_div10(
                frame["AS_raw"],
                mean=as_mean,
                std=as_std,
            )
        if name in frames:
            frames[name] = pd.concat([frames[name], frame], ignore_index=True)
        else:
            frames[name] = frame
    return frames


def main() -> None:
    args = parse_args()
    paths = CustomerDataPaths(
        assays_xlsx=args.root / "Вся_химия+литология+Au_final_all_data.XLSX",
        center_blocks_csv=args.root / "md_nat250721_CEN_Отработано.csv",
        north_blocks_csv=args.root / "md_nat241227(Модель_ресурсов_ind,inf).csv",
    )
    data = read_customer_data(paths)
    center_as_raw = pd.to_numeric(pd.read_csv(paths.center_blocks_csv, usecols=["AS"])["AS"], errors="coerce")
    _, as_mean, as_std = transform_as_abs_zscore_div10(center_as_raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data.assays.to_parquet(args.output_dir / "assays_standardized.parquet", index=False)
    data.center_blocks.to_parquet(args.output_dir / "center_blocks_standardized.parquet", index=False)
    data.north_blocks.to_parquet(args.output_dir / "north_blocks_standardized.parquet", index=False)

    baseline_frames = load_baseline_specs(args.baseline, args.root, as_mean=as_mean, as_std=as_std)
    center_nodes = build_center_node_table(data.center_blocks, baseline_frames=baseline_frames)
    north_nodes = build_north_node_table(data.north_blocks, baseline_frames=baseline_frames)
    center_nodes, north_nodes = add_pca_spatial_coordinates(center_nodes, north_nodes)
    center_nodes, north_nodes = add_train_eval_flags(center_nodes, north_nodes)

    if args.add_knn_chemistry:
        knn_config = KNNConfig(
            n_neighbors=args.knn_neighbors,
            power=args.knn_power,
            batch_size=args.knn_batch_size,
        )
        print(
            "Projecting assay chemistry to CEN/NTH nodes "
            f"(k={knn_config.n_neighbors}, power={knn_config.power})..."
        )
        center_knn = project_assay_chemistry_to_nodes(data.assays, center_nodes, config=knn_config)
        north_knn = project_assay_chemistry_to_nodes(data.assays, north_nodes, config=knn_config)
        center_nodes = pd.concat([center_nodes.reset_index(drop=True), center_knn], axis=1)
        north_nodes = pd.concat([north_nodes.reset_index(drop=True), north_knn], axis=1)
        center_knn.to_parquet(args.output_dir / "center_knn_chemistry.parquet", index=False)
        north_knn.to_parquet(args.output_dir / "north_knn_chemistry.parquet", index=False)

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
