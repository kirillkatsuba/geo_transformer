from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["AS", "S", "CORG-1", "CA", "FE"]
COORD_COLUMNS = ["X", "Y", "Z"]
BLOCK_SIZE_COLUMNS = ["_X", "_Y", "_Z"]


@dataclass(frozen=True)
class CustomerDataPaths:
    """Input files received from the customer."""

    assays_xlsx: Path = Path("Вся_химия+литология+Au_final_all_data.XLSX")
    center_blocks_csv: Path = Path("md_nat250721_CEN_Отработано.csv")
    north_blocks_csv: Path = Path("md_nat241227(Модель_ресурсов_ind,inf).csv")


@dataclass
class LoadedCustomerData:
    """Standardized customer datasets.

    `assays` are drillhole interval observations.
    `center_blocks` are the interpolation/supervised block domain.
    `north_blocks` are the extrapolation target domain.
    """

    assays: pd.DataFrame
    center_blocks: pd.DataFrame
    north_blocks: pd.DataFrame


def transform_as_abs_zscore_div10(
    values: pd.Series,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[pd.Series, float, float]:
    """Transform AS as abs(z-score) / 10.

    This follows the project convention:

    ```python
    AS = abs((AS - AS.mean()) / AS.var()**0.5) / 10
    ```

    `std` is computed with population variance (`ddof=0`) to match
    `numpy.var() ** 0.5`.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    if mean is None:
        mean = float(numeric.mean())
    if std is None:
        std = float(numeric.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        std = 1.0
    transformed = ((numeric - mean) / std).abs() / 10.0
    return transformed, mean, std


def _first_existing(columns: Iterable[str], df: pd.DataFrame) -> str | None:
    for col in columns:
        if col in df.columns:
            return col
    return None


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def read_customer_data(paths: CustomerDataPaths | None = None) -> LoadedCustomerData:
    """Read and standardize the three customer-provided datasets."""

    paths = paths or CustomerDataPaths()
    assays_raw = pd.read_excel(paths.assays_xlsx, sheet_name=0)
    center_raw = pd.read_csv(paths.center_blocks_csv)
    north_raw = pd.read_csv(paths.north_blocks_csv)

    center_as_source = pd.to_numeric(center_raw["AS"], errors="coerce")
    _, as_mean, as_std = transform_as_abs_zscore_div10(center_as_source)

    return LoadedCustomerData(
        assays=standardize_assays(assays_raw),
        center_blocks=standardize_block_model(
            center_raw,
            domain="CEN",
            as_mean=as_mean,
            as_std=as_std,
        ),
        north_blocks=standardize_block_model(
            north_raw,
            domain="NTH",
            as_mean=as_mean,
            as_std=as_std,
        ),
    )


def standardize_assays(
    df: pd.DataFrame,
    as_mean: float | None = None,
    as_std: float | None = None,
) -> pd.DataFrame:
    """Normalize drillhole interval table columns.

    `AS` is transformed with `abs(z-score) / 10` when `as_mean/as_std`
    are provided. Other targets are kept in assay-native percent units.
    """

    _require_columns(
        df,
        ["HOLE_ID", "DEPTH_FROM", "DEPTH_TO", "X", "Y", "Z", "Au_Final"],
        "assays",
    )

    target_map = {
        "AS": _first_existing(["As (ME-ICP61),ppm", "AS"], df),
        "S": _first_existing(["Sобщ (S-IR08),%", "S (ME-ICP61),%", "S"], df),
        "CORG-1": _first_existing(["C organic (C-IR06),%", "CORG-1", "CORG"], df),
        "CA": _first_existing(["Ca (ME-ICP61),%", "CA"], df),
        "FE": _first_existing(["Fe (ME-ICP61),%", "FE"], df),
    }
    missing_targets = [target for target, source in target_map.items() if source is None]
    if missing_targets:
        raise ValueError(f"assays is missing target columns for: {missing_targets}")

    out = pd.DataFrame(index=df.index)
    out["source"] = "assay"
    out["domain"] = "DRILLHOLE"
    out["HOLE_ID"] = df["HOLE_ID"].astype(str)
    out["SAMPLE_ID"] = df.get("SAMPLE_ID", pd.Series(index=df.index, dtype="object"))
    out["DEPTH_FROM"] = pd.to_numeric(df["DEPTH_FROM"], errors="coerce")
    out["DEPTH_TO"] = pd.to_numeric(df["DEPTH_TO"], errors="coerce")
    out["interval_length"] = (
        pd.to_numeric(df.get("LENGTH", out["DEPTH_TO"] - out["DEPTH_FROM"]), errors="coerce")
        .fillna(out["DEPTH_TO"] - out["DEPTH_FROM"])
    )
    out[COORD_COLUMNS] = df[COORD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out["Au_Final"] = pd.to_numeric(df["Au_Final"], errors="coerce")

    for target, source in target_map.items():
        out[target] = pd.to_numeric(df[source], errors="coerce")

    out["AS_raw"] = out["AS"]
    if as_mean is not None and as_std is not None:
        out["AS"], _, _ = transform_as_abs_zscore_div10(out["AS"], mean=as_mean, std=as_std)

    optional_feature_cols = [
        "DIST_Sandstone",
        "DIST_SP",
        "DIST_R",
        "DIST_P3om GR",
        "DIST_P3at TG",
        "DIST_JZ",
        "DIST_FZ",
        "FRAME_NAME",
        "IN_ORE",
        "LITH",
        "GTO_INT",
        "LITH_STRUCTURE",
        "LITH_TEXTURE",
        "LITH_COLOUR",
        "INCL_PERCENT",
    ]
    for col in optional_feature_cols:
        if col in df.columns:
            out[col] = df[col]

    out["has_targets"] = out[TARGET_COLUMNS].notna().all(axis=1)
    return out


def standardize_block_model(
    df: pd.DataFrame,
    domain: str,
    as_mean: float | None = None,
    as_std: float | None = None,
) -> pd.DataFrame:
    """Normalize block model columns to the same naming scheme.

    `AS` is transformed with `abs(z-score) / 10`. By default the caller
    should pass center-block statistics so center/north use the same scale.
    """

    _require_columns(
        df,
        ["EAST", "NORTH", "RL", "_EAST", "_NORTH", "_RL", "AU"],
        f"{domain} block model",
    )

    out = pd.DataFrame(index=df.index)
    out["source"] = "block"
    out["domain"] = domain
    out["block_id"] = np.arange(len(df), dtype=np.int64)
    out["IJK"] = df.get("IJK", pd.Series(index=df.index, dtype="float"))
    out["X"] = pd.to_numeric(df["EAST"], errors="coerce")
    out["Y"] = pd.to_numeric(df["NORTH"], errors="coerce")
    out["Z"] = pd.to_numeric(df["RL"], errors="coerce")
    out["_X"] = pd.to_numeric(df["_EAST"], errors="coerce")
    out["_Y"] = pd.to_numeric(df["_NORTH"], errors="coerce")
    out["_Z"] = pd.to_numeric(df["_RL"], errors="coerce")
    out["Au_Final"] = pd.to_numeric(df["AU"], errors="coerce")

    target_map = {
        "AS": "AS",
        "S": "S",
        "CORG-1": "CORG" if "CORG" in df.columns else "CORG-1",
        "CA": "CA",
        "FE": "FE",
    }
    for target, source in target_map.items():
        if source in df.columns:
            out[target] = pd.to_numeric(df[source], errors="coerce")
        else:
            out[target] = np.nan

    out["AS_raw"] = out["AS"]
    out["AS"], _, _ = transform_as_abs_zscore_div10(out["AS"], mean=as_mean, std=as_std)

    for col in ["DENSITY", "RESCAT", "MINED", "MODAREA", "ZONE", "PVALUE", "IND", "RESCAT_C"]:
        if col in df.columns:
            out[col] = df[col]

    out["volume"] = out["_X"] * out["_Y"] * out["_Z"]
    out["has_targets"] = out[TARGET_COLUMNS].notna().all(axis=1)
    return out


def build_center_node_table(
    center_blocks: pd.DataFrame,
    baseline_frames: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build a supervised node table from the central block model.

    The first version uses one token per block. Later we can replace this
    with quadrature nodes or microblocks while preserving the same interface.
    """

    nodes = center_blocks.copy()
    nodes["node_id"] = np.arange(len(nodes), dtype=np.int64)
    nodes["is_target_domain"] = False
    nodes["is_supervised_domain"] = True
    nodes["known_target_mask"] = nodes["has_targets"]
    nodes = attach_baselines(nodes, baseline_frames)
    return nodes


def build_north_node_table(
    north_blocks: pd.DataFrame,
    baseline_frames: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build an extrapolation node table from the northern block model."""

    nodes = north_blocks.copy()
    nodes["node_id"] = np.arange(len(nodes), dtype=np.int64)
    nodes["is_target_domain"] = True
    nodes["is_supervised_domain"] = False
    nodes["known_target_mask"] = nodes["has_targets"]
    nodes = attach_baselines(nodes, baseline_frames)
    return nodes


def attach_baselines(
    nodes: pd.DataFrame,
    baseline_frames: dict[str, pd.DataFrame] | None = None,
    coord_round: int = 6,
) -> pd.DataFrame:
    """Attach existing model predictions as baseline features.

    Baseline frames should contain `X`, `Y`, `Z` and target columns. Columns are
    added as `{name}_{target}`.
    """

    if not baseline_frames:
        return nodes

    out = nodes.copy()
    key = ["X", "Y", "Z"]
    out_key = out[key].round(coord_round)
    out["_join_key"] = list(map(tuple, out_key.to_numpy()))

    for name, frame in baseline_frames.items():
        _require_columns(frame, key, f"baseline {name}")
        base = frame.copy()
        base["_join_key"] = list(map(tuple, base[key].round(coord_round).to_numpy()))
        cols = ["_join_key"] + [col for col in TARGET_COLUMNS if col in base.columns]
        base = base[cols].drop_duplicates("_join_key")
        rename = {target: f"{name}_{target}" for target in TARGET_COLUMNS if target in base.columns}
        out = out.merge(base.rename(columns=rename), on="_join_key", how="left")

    return out.drop(columns=["_join_key"])


def summarize_loaded_data(data: LoadedCustomerData) -> pd.DataFrame:
    """Return compact diagnostics for the three standardized datasets."""

    rows = []
    for name, df in [
        ("assays", data.assays),
        ("center_blocks", data.center_blocks),
        ("north_blocks", data.north_blocks),
    ]:
        row = {
            "dataset": name,
            "rows": len(df),
            "has_targets_rows": int(df["has_targets"].sum()),
            "target_coverage": float(df["has_targets"].mean()),
            "x_min": df["X"].min(),
            "x_max": df["X"].max(),
            "y_min": df["Y"].min(),
            "y_max": df["Y"].max(),
            "z_min": df["Z"].min(),
            "z_max": df["Z"].max(),
        }
        for target in TARGET_COLUMNS:
            row[f"{target}_non_null"] = int(df[target].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)
