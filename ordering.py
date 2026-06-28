from __future__ import annotations

import numpy as np
import pandas as pd


def order_by_strike(
    nodes: pd.DataFrame,
    strike_col: str = "coord_strike",
    cross_col: str = "coord_cross",
    z_col: str = "Z",
    ascending: bool = True,
) -> np.ndarray:
    """Order nodes along geological strike, then cross-strike, then vertical."""

    required = [strike_col, cross_col, z_col]
    missing = [col for col in required if col not in nodes.columns]
    if missing:
        raise ValueError(f"nodes is missing required columns: {missing}")
    return nodes.sort_values(required, ascending=ascending).index.to_numpy()


def order_by_distance_to_data(
    nodes: pd.DataFrame,
    distance_col: str = "nearest_distance_to_train",
    tie_cols: tuple[str, ...] = ("coord_strike", "coord_cross", "Z"),
) -> np.ndarray:
    """Order nodes from best-supported regions toward extrapolation."""

    cols = [distance_col] + [col for col in tie_cols if col in nodes.columns]
    if distance_col not in nodes.columns:
        raise ValueError(f"nodes is missing distance column {distance_col!r}")
    return nodes.sort_values(cols).index.to_numpy()


def order_by_domain_then_strike(
    nodes: pd.DataFrame,
    domain_col: str = "domain_id",
    strike_col: str = "coord_strike",
    cross_col: str = "coord_cross",
    z_col: str = "Z",
) -> np.ndarray:
    """Order nodes inside domains before moving across structural domains."""

    cols = [col for col in [domain_col, strike_col, cross_col, z_col] if col in nodes.columns]
    if not cols:
        raise ValueError("nodes does not contain any usable ordering columns")
    return nodes.sort_values(cols).index.to_numpy()


def random_order(n_nodes: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n_nodes)


def invert_order(order: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return inverse

