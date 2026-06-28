from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .data import COORD_COLUMNS


@dataclass(frozen=True)
class KNNConfig:
    n_neighbors: int = 16
    power: float = 2.0
    eps: float = 1e-6
    batch_size: int = 100_000


def chemistry_feature_columns(assays: pd.DataFrame) -> list[str]:
    """Return standardized assay chemistry columns available for KNN projection."""

    return [col for col in assays.columns if col.startswith("chem_")]


def _fit_assay_matrix(
    assays: pd.DataFrame,
    chem_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = assays[COORD_COLUMNS].notna().all(axis=1)
    valid &= assays[chem_cols].notna().any(axis=1)
    work = assays.loc[valid].copy()
    if work.empty:
        raise ValueError("No valid assay rows for KNN chemistry projection")

    coords = work[COORD_COLUMNS].to_numpy(dtype=np.float32)
    values = work[chem_cols].apply(pd.to_numeric, errors="coerce")
    medians = values.median(axis=0).fillna(0.0).to_numpy(dtype=np.float32)
    matrix = values.fillna(pd.Series(medians, index=chem_cols)).to_numpy(dtype=np.float32)
    return coords, matrix, medians


def project_assay_chemistry_to_nodes(
    assays: pd.DataFrame,
    nodes: pd.DataFrame,
    chem_cols: list[str] | None = None,
    config: KNNConfig | None = None,
    prefix: str = "knn_",
) -> pd.DataFrame:
    """Project assay chemistry to block/microblock nodes with IDW KNN.

    Output columns:
      - `{prefix}{chem_col}` for each assay chemistry column;
      - `{prefix}mean_distance`;
      - `{prefix}min_distance`;
      - `{prefix}max_distance`;
      - `{prefix}distance_std`;
    """

    config = config or KNNConfig()
    chem_cols = chem_cols or chemistry_feature_columns(assays)
    if not chem_cols:
        raise ValueError("No assay chemistry columns found")
    missing_nodes = [col for col in COORD_COLUMNS if col not in nodes.columns]
    if missing_nodes:
        raise ValueError(f"nodes is missing coordinate columns: {missing_nodes}")

    assay_coords, assay_values, _ = _fit_assay_matrix(assays, chem_cols)
    n_neighbors = min(config.n_neighbors, len(assay_coords))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(assay_coords)

    node_coords = nodes[COORD_COLUMNS].to_numpy(dtype=np.float32)
    feature_chunks = []

    for start in range(0, len(nodes), config.batch_size):
        stop = min(start + config.batch_size, len(nodes))
        distances, indices = nn.kneighbors(node_coords[start:stop], return_distance=True)
        weights = 1.0 / np.power(distances + config.eps, config.power)
        weights = weights / weights.sum(axis=1, keepdims=True)
        projected = np.einsum("bk,bkf->bf", weights, assay_values[indices])

        chunk = pd.DataFrame(projected, columns=[f"{prefix}{col}" for col in chem_cols])
        chunk[f"{prefix}mean_distance"] = distances.mean(axis=1)
        chunk[f"{prefix}min_distance"] = distances.min(axis=1)
        chunk[f"{prefix}max_distance"] = distances.max(axis=1)
        chunk[f"{prefix}distance_std"] = distances.std(axis=1)
        feature_chunks.append(chunk)

    return pd.concat(feature_chunks, ignore_index=True)

