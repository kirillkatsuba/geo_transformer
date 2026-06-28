from __future__ import annotations

import torch

from .model import GeoTransformer


@torch.no_grad()
def generate_autoregressive(
    model: GeoTransformer,
    conditions: torch.Tensor,
    order: torch.Tensor,
    assay_tokens: torch.Tensor | None = None,
    baseline: torch.Tensor | None = None,
    sample: bool = True,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generate targets in a given order.

    Args:
        conditions: [n_nodes, condition_dim]
        order: [n_nodes] permutation of node indices
        baseline: optional [n_nodes, target_dim], added outside if model is residual.

    Returns:
        generated targets in original node order: [n_nodes, target_dim]
    """

    device = next(model.parameters()).device
    conditions = conditions.to(device)
    order = order.to(device)
    ordered_conditions = conditions[order].unsqueeze(0)

    n_nodes = ordered_conditions.size(1)
    target_dim = model.config.target_dim
    generated_ordered = torch.zeros(1, n_nodes, target_dim, device=device)
    attention_mask = torch.ones(1, n_nodes, dtype=torch.bool, device=device)
    assay_batch = assay_tokens.to(device).unsqueeze(0) if assay_tokens is not None else None

    for pos in range(n_nodes):
        mu, log_sigma = model(
            conditions=ordered_conditions[:, : pos + 1],
            prev_targets=generated_ordered[:, : pos + 1],
            assay_tokens=assay_batch,
            attention_mask=attention_mask[:, : pos + 1],
        )
        mu_i = mu[:, -1]
        if log_sigma is not None and sample:
            eps = torch.randn_like(mu_i)
            y_i = mu_i + torch.exp(log_sigma[:, -1]) * eps * temperature
        else:
            y_i = mu_i
        generated_ordered[:, pos] = y_i

    generated = torch.zeros_like(generated_ordered)
    generated[:, order] = generated_ordered
    generated = generated.squeeze(0)
    if baseline is not None:
        generated = generated + baseline.to(device)
    return generated


@torch.no_grad()
def generate_multiple_realizations(
    model: GeoTransformer,
    conditions: torch.Tensor,
    orders: list[torch.Tensor],
    assay_tokens: torch.Tensor | None = None,
    baseline: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generate one realization per order."""

    realizations = [
        generate_autoregressive(
            model=model,
            conditions=conditions,
            order=order,
            assay_tokens=assay_tokens,
            baseline=baseline,
            sample=True,
            temperature=temperature,
        )
        for order in orders
    ]
    return torch.stack(realizations, dim=0)

