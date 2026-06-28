from dataclasses import dataclass, field


@dataclass
class GeoTransformerConfig:
    """Model configuration for the causal geochemical Transformer."""

    condition_dim: int
    target_dim: int = 5
    assay_dim: int = 5
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_sequence_length: int = 4096
    use_assay_cross_attention: bool = True
    predict_log_sigma: bool = True


@dataclass
class TrainingConfig:
    """Training defaults for teacher-forced autoregressive learning."""

    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    gradient_clip_norm: float = 1.0
    lambda_nll: float = 1.0
    lambda_assay: float = 1.0
    lambda_prior: float = 0.05
    lambda_spatial: float = 0.0
    scheduled_sampling_prob: float = 0.0
    context_dropout: float = 0.0
    target_columns: list[str] = field(
        default_factory=lambda: ["AS", "S", "CORG-1", "CA", "FE"]
    )

