from __future__ import annotations

import math

import torch
from torch import nn

from .config import GeoTransformerConfig


class SinusoidalPositionEncoding(nn.Module):
    """Standard sinusoidal sequence position encoding."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class GeoTransformer(nn.Module):
    """Causal Transformer for autoregressive residual-field generation.

    Inputs are sequences of spatial nodes. During training, `prev_targets`
    is teacher-forced and shifted by the caller so token i only receives
    target information from tokens `< i`.

    Shapes:
        conditions: [batch, seq, condition_dim]
        prev_targets: [batch, seq, target_dim]
        assay_tokens: optional [batch, n_assays, assay_dim]
        attention_mask: optional bool [batch, seq], True for valid tokens
    """

    def __init__(self, config: GeoTransformerConfig):
        super().__init__()
        self.config = config

        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(config.target_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        self.position_encoding = SinusoidalPositionEncoding(
            config.d_model, config.max_sequence_length
        )

        if config.use_assay_cross_attention:
            self.assay_encoder = nn.Sequential(
                nn.Linear(config.assay_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model),
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, config.n_layers)
            self.null_memory = nn.Parameter(torch.zeros(1, 1, config.d_model))
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, config.n_layers)

        out_dim = config.target_dim * (2 if config.predict_log_sigma else 1)
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, out_dim),
        )

    def forward(
        self,
        conditions: torch.Tensor,
        prev_targets: torch.Tensor,
        assay_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = conditions.shape
        if seq_len > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}"
            )

        x = self.condition_encoder(conditions) + self.target_encoder(prev_targets)
        x = self.position_encoding(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        if self.config.use_assay_cross_attention:
            if assay_tokens is None:
                memory = self.null_memory.expand(batch_size, 1, -1)
            else:
                memory = self.assay_encoder(assay_tokens)
            hidden = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=key_padding_mask,
            )
        else:
            hidden = self.encoder(
                x,
                mask=causal_mask,
                src_key_padding_mask=key_padding_mask,
            )

        out = self.output_head(hidden)
        if self.config.predict_log_sigma:
            mu, log_sigma = out.chunk(2, dim=-1)
            log_sigma = torch.clamp(log_sigma, min=-7.0, max=4.0)
            return mu, log_sigma
        return out, None


def shift_targets_right(targets: torch.Tensor, bos_value: float = 0.0) -> torch.Tensor:
    """Create teacher-forcing inputs so position i sees target i-1."""

    shifted = torch.zeros_like(targets)
    shifted[:, 0, :] = bos_value
    shifted[:, 1:, :] = targets[:, :-1, :]
    return shifted

