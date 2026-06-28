from __future__ import annotations

import torch
from torch.nn import functional as F

from .operators import apply_operator


def gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diagonal Gaussian negative log likelihood."""

    sigma2 = torch.exp(2.0 * log_sigma)
    loss = 0.5 * ((target - mu) ** 2 / sigma2 + 2.0 * log_sigma)
    if mask is not None:
        loss = loss * mask.unsqueeze(-1).to(loss.dtype)
        denom = mask.sum().clamp_min(1).to(loss.dtype) * target.size(-1)
        return loss.sum() / denom
    return loss.mean()


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    delta: float = 1.0,
) -> torch.Tensor:
    loss = F.huber_loss(pred, target, delta=delta, reduction="none")
    if mask is not None:
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        loss = loss * mask.to(loss.dtype)
        return loss.sum() / mask.sum().clamp_min(1).to(loss.dtype)
    return loss.mean()


def assay_consistency_loss(
    generated_nodes: torch.Tensor,
    observed_assays: torch.Tensor,
    assay_operator: torch.Tensor,
    mask: torch.Tensor | None = None,
    delta: float = 1.0,
) -> torch.Tensor:
    """Compare generated node field with observed assay interval averages."""

    pred_assays = apply_operator(assay_operator, generated_nodes)
    return masked_huber(pred_assays, observed_assays, mask=mask, delta=delta)


def prior_residual_loss(
    residual: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize unnecessary departures from the baseline field."""

    loss = residual**2
    if confidence is not None:
        while confidence.dim() < loss.dim():
            confidence = confidence.unsqueeze(-1)
        loss = loss * confidence.to(loss.dtype)
    return loss.mean()

