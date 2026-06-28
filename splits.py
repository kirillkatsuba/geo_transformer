from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DomainMasks:
    center_known: np.ndarray
    north_known: np.ndarray
    north_unknown: np.ndarray


def build_domain_masks(
    center_nodes: pd.DataFrame,
    north_nodes: pd.DataFrame,
) -> DomainMasks:
    """Create the first practical masks for interpolation/extrapolation tests."""

    center_known = center_nodes["has_targets"].to_numpy(dtype=bool)
    north_known = north_nodes["has_targets"].to_numpy(dtype=bool)
    north_unknown = ~north_known
    return DomainMasks(
        center_known=center_known,
        north_known=north_known,
        north_unknown=north_unknown,
    )


def add_train_eval_flags(
    center_nodes: pd.DataFrame,
    north_nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark rows for the initial experiment protocol."""

    center = center_nodes.copy()
    north = north_nodes.copy()

    center["experiment_role"] = np.where(center["has_targets"], "train_center_known", "center_missing")
    north["experiment_role"] = np.where(north["has_targets"], "eval_north_known", "predict_north_unknown")
    return center, north

