from __future__ import annotations

import torch

from .config import TrainingConfig
from .losses import assay_consistency_loss, gaussian_nll, prior_residual_loss
from .model import GeoTransformer, shift_targets_right


def training_step(
    model: GeoTransformer,
    batch: dict[str, torch.Tensor],
    config: TrainingConfig,
    assay_operator: torch.Tensor | None = None,
    observed_assays: torch.Tensor | None = None,
    assay_tokens: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Single teacher-forced training step.

    The model output is interpreted as residual targets. `batch["baseline"]`
    is added when computing assay consistency.
    """

    prev_targets = batch["prev_targets"]
    if config.context_dropout > 0:
        keep = torch.rand(prev_targets.shape[:2], device=prev_targets.device) >= config.context_dropout
        prev_targets = prev_targets * keep.unsqueeze(-1).to(prev_targets.dtype)

    if config.scheduled_sampling_prob > 0:
        with torch.no_grad():
            warm_mu, _ = model(
                conditions=batch["conditions"],
                prev_targets=prev_targets,
                assay_tokens=assay_tokens,
                attention_mask=batch["attention_mask"],
            )
            generated_targets = batch["baseline"] + warm_mu
            generated_prev = shift_targets_right(generated_targets)
            use_generated = (
                torch.rand(prev_targets.shape[:2], device=prev_targets.device)
                < config.scheduled_sampling_prob
            )
            use_generated[:, 0] = False
            prev_targets = torch.where(
                use_generated.unsqueeze(-1),
                generated_prev,
                prev_targets,
            )

    mu, log_sigma = model(
        conditions=batch["conditions"],
        prev_targets=prev_targets,
        assay_tokens=assay_tokens,
        attention_mask=batch["attention_mask"],
    )

    target_residual = batch["targets"] - batch["baseline"]
    if log_sigma is not None:
        nll = gaussian_nll(
            target=target_residual,
            mu=mu,
            log_sigma=log_sigma,
            mask=batch["attention_mask"],
        )
    else:
        nll = torch.nn.functional.mse_loss(mu, target_residual)

    prior = prior_residual_loss(mu)
    total = config.lambda_nll * nll + config.lambda_prior * prior
    losses = {"loss": total, "nll": nll.detach(), "prior": prior.detach()}

    if assay_operator is not None and observed_assays is not None:
        generated_nodes = batch["baseline"] + mu
        assay_loss = assay_consistency_loss(
            generated_nodes=generated_nodes,
            observed_assays=observed_assays,
            assay_operator=assay_operator,
        )
        total = total + config.lambda_assay * assay_loss
        losses["loss"] = total
        losses["assay"] = assay_loss.detach()

    return losses
