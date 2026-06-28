from __future__ import annotations

import pandas as pd
import torch


def build_sparse_operator(
    operator_df: pd.DataFrame,
    n_operators: int,
    n_nodes: int,
    operator_col: str = "operator_id",
    node_col: str = "node_id",
    weight_col: str = "weight",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a sparse operator matrix from a long table.

    The returned matrix has shape `[n_operators, n_nodes]`.
    """

    required = {operator_col, node_col, weight_col}
    missing = required.difference(operator_df.columns)
    if missing:
        raise ValueError(f"operator_df is missing required columns: {sorted(missing)}")

    indices = torch.tensor(
        operator_df[[operator_col, node_col]].to_numpy().T,
        dtype=torch.long,
        device=device,
    )
    values = torch.tensor(
        operator_df[weight_col].to_numpy(),
        dtype=torch.float32,
        device=device,
    )
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(n_operators, n_nodes),
        device=device,
    ).coalesce()


def normalize_operator_rows(operator: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize sparse operator rows so row weights sum to one."""

    operator = operator.coalesce()
    row_sum = torch.sparse.sum(operator, dim=1).to_dense().clamp_min(eps)
    row_idx = operator.indices()[0]
    values = operator.values() / row_sum[row_idx]
    return torch.sparse_coo_tensor(
        operator.indices(), values, operator.shape, device=operator.device
    ).coalesce()


def apply_operator(operator: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Apply a sparse support operator to node values.

    Args:
        operator: sparse tensor [n_operators, n_nodes]
        values: dense tensor [n_nodes, target_dim] or [batch, n_nodes, target_dim]
    """

    if values.dim() == 2:
        return torch.sparse.mm(operator, values)
    if values.dim() == 3:
        outputs = [torch.sparse.mm(operator, sample) for sample in values]
        return torch.stack(outputs, dim=0)
    raise ValueError("values must have shape [nodes, targets] or [batch, nodes, targets]")


def operator_from_intersections(
    intersections: pd.DataFrame,
    operator_col: str,
    node_col: str,
    length_col: str,
    operator_id_col: str = "operator_id",
) -> pd.DataFrame:
    """Convert interval/node intersections to normalized operator weights."""

    required = {operator_col, node_col, length_col}
    missing = required.difference(intersections.columns)
    if missing:
        raise ValueError(f"intersections is missing required columns: {sorted(missing)}")

    out = intersections[[operator_col, node_col, length_col]].copy()
    out = out.rename(columns={operator_col: operator_id_col, length_col: "weight"})
    denom = out.groupby(operator_id_col)["weight"].transform("sum")
    out["weight"] = out["weight"] / denom
    return out[[operator_id_col, node_col, "weight"]]

