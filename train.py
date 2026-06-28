from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import GeoTransformerConfig, TrainingConfig
from .data import TARGET_COLUMNS
from .dataset import GeoSequenceDataset, collate_sequences
from .losses import gaussian_nll
from .model import GeoTransformer
from .ordering import order_by_domain_then_strike, order_by_distance_to_data, order_by_strike, random_order
from .train_step import training_step


class TargetStandardizer:
    """Mean/std target scaler saved with the checkpoint."""

    def __init__(self) -> None:
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "TargetStandardizer":
        for col in columns:
            values = pd.to_numeric(df[col], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            self.mean[col] = mean
            self.std[col] = std if np.isfinite(std) and std > 1e-12 else 1.0
        return self

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in columns:
            out[f"{col}_scaled"] = (pd.to_numeric(df[col], errors="coerce") - self.mean[col]) / self.std[col]
        return out

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"mean": self.mean, "std": self.std}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first GeoTransformer prototype.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("geo_transformer/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("geo_transformer/runs/first"))
    parser.add_argument("--order", choices=["strike", "distance", "domain_strike", "random"], default="domain_strike")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--max-sequences", type=int, default=0, help="0 means use all sequences")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_feature_columns(prepared_dir: Path) -> list[str]:
    feature_file = prepared_dir / "feature_columns.txt"
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing feature column file: {feature_file}")
    return [line.strip() for line in feature_file.read_text().splitlines() if line.strip()]


def make_order(nodes: pd.DataFrame, order_name: str, seed: int) -> np.ndarray:
    if order_name == "strike":
        return order_by_strike(nodes)
    if order_name == "distance":
        if "nearest_distance_to_train" in nodes.columns:
            return order_by_distance_to_data(nodes)
        return order_by_strike(nodes)
    if order_name == "domain_strike":
        return order_by_domain_then_strike(nodes)
    if order_name == "random":
        return random_order(len(nodes), seed=seed)
    raise ValueError(f"Unknown order {order_name!r}")


def chunk_order(order: np.ndarray, sequence_length: int) -> list[np.ndarray]:
    return [
        order[start : start + sequence_length]
        for start in range(0, len(order), sequence_length)
        if len(order[start : start + sequence_length]) > 1
    ]


def evaluate(
    model: GeoTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            mu, log_sigma = model(
                conditions=batch["conditions"],
                prev_targets=batch["prev_targets"],
                attention_mask=batch["attention_mask"],
            )
            target_residual = batch["targets"] - batch["baseline"]
            if log_sigma is not None:
                loss = gaussian_nll(target_residual, mu, log_sigma, mask=batch["attention_mask"])
            else:
                loss = torch.nn.functional.mse_loss(mu, target_residual)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    center = pd.read_parquet(args.prepared_dir / "center_nodes.parquet")
    center = center.loc[center["experiment_role"] == "train_center_known"].reset_index(drop=True)
    feature_columns = load_feature_columns(args.prepared_dir)

    scaler = TargetStandardizer().fit(center, TARGET_COLUMNS)
    scaled = scaler.transform(center, TARGET_COLUMNS)
    scaled_targets = [f"{target}_scaled" for target in TARGET_COLUMNS]
    center = pd.concat([center, scaled], axis=1)
    for col in scaled_targets:
        center[f"baseline_{col}"] = 0.0

    order = make_order(center, args.order, args.seed)
    sequences = chunk_order(order, args.sequence_length)
    if args.max_sequences and args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    rng = np.random.default_rng(args.seed)
    seq_idx = np.arange(len(sequences))
    rng.shuffle(seq_idx)
    val_count = max(1, int(len(seq_idx) * args.val_fraction))
    val_indices = set(seq_idx[:val_count].tolist())
    train_sequences = [seq for idx, seq in enumerate(sequences) if idx not in val_indices]
    val_sequences = [seq for idx, seq in enumerate(sequences) if idx in val_indices]

    train_dataset = GeoSequenceDataset(
        center,
        condition_columns=feature_columns,
        target_columns=scaled_targets,
        baseline_columns=[f"baseline_{col}" for col in scaled_targets],
        orders=train_sequences,
    )
    val_dataset = GeoSequenceDataset(
        center,
        condition_columns=feature_columns,
        target_columns=scaled_targets,
        baseline_columns=[f"baseline_{col}" for col in scaled_targets],
        orders=val_sequences,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )

    model_config = GeoTransformerConfig(
        condition_dim=len(feature_columns),
        target_dim=len(TARGET_COLUMNS),
        assay_dim=len(TARGET_COLUMNS),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_sequence_length=args.sequence_length,
        use_assay_cross_attention=False,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        target_columns=TARGET_COLUMNS,
    )

    model = GeoTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    metrics = []
    best_val = float("inf")
    best_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            losses = training_step(model, batch, training_config)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(losses["loss"].detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        metrics.append(row)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "training_config": asdict(training_config),
                    "feature_columns": feature_columns,
                    "target_columns": TARGET_COLUMNS,
                    "scaled_target_columns": scaled_targets,
                    "target_scaler": scaler.to_dict(),
                    "order": args.order,
                    "sequence_length": args.sequence_length,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )

    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(serializable_args, ensure_ascii=False, indent=2)
    )
    print(f"\nSaved best checkpoint: {best_path}")
    print(f"Saved metrics: {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
