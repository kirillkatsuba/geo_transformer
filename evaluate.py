from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-geo-transformer")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import TARGET_COLUMNS
from .dataset import GeoSequenceDataset, collate_sequences
from .inference import generate_autoregressive
from .model import GeoTransformer
from .ordering import order_by_domain_then_strike, order_by_distance_to_data, order_by_strike, random_order
from .target_baseline import add_scaled_target_baselines

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - cluster fallback
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GeoTransformer on CEN/NTH known blocks.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("geo_transformer/prepared"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("geo_transformer/eval/center_v1"))
    parser.add_argument("--domain", choices=["center", "north", "both"], default="both")
    parser.add_argument(
        "--mode",
        choices=["teacher_forced", "autoregressive"],
        default="teacher_forced",
        help="teacher_forced is fast and optimistic; autoregressive is slower and closer to inference.",
    )
    parser.add_argument("--order", choices=["checkpoint", "strike", "distance", "domain_strike", "random"], default="checkpoint")
    parser.add_argument(
        "--ensemble-orders",
        default="",
        help="Comma-separated orders for AR ensemble, e.g. domain_strike,strike,random. Only for autoregressive mode.",
    )
    parser.add_argument("--sequence-length", type=int, default=0, help="0 uses checkpoint sequence_length")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=0, help="0 means all sequences")
    parser.add_argument(
        "--sample-sequences",
        action="store_true",
        help="When max-sequences is set, sample chunks randomly instead of taking the first chunks.",
    )
    parser.add_argument("--max-plot-points", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument(
        "--token-progress",
        action="store_true",
        help="Show inner token progress for autoregressive generation. Verbose but useful for long single chunks.",
    )
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


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def restore_model(checkpoint: dict, device: torch.device) -> GeoTransformer:
    from .config import GeoTransformerConfig

    config = GeoTransformerConfig(**checkpoint["model_config"])
    model = GeoTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def make_order(nodes: pd.DataFrame, order_name: str, seed: int) -> np.ndarray:
    if order_name == "strike":
        return order_by_strike(nodes)
    if order_name == "distance":
        if "nearest_distance_to_train" in nodes.columns:
            return order_by_distance_to_data(nodes)
        return order_by_domain_then_strike(nodes)
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


def inverse_targets(values: np.ndarray, checkpoint: dict) -> pd.DataFrame:
    scaler = checkpoint["target_scaler"]
    out = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        out[target] = values[:, idx] * scaler["std"][target] + scaler["mean"][target]
    return pd.DataFrame(out)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        y = df[f"true_{target}"].to_numpy(dtype=float)
        pred = df[f"pred_{target}"].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(pred)
        y = y[mask]
        pred = pred[mask]
        if len(y) == 0:
            continue
        err = pred - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        denom = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - np.sum(err**2) / denom) if denom > 1e-12 else np.nan
        nonzero = np.abs(y) > 1e-12
        mape = float(np.mean(np.abs(err[nonzero] / y[nonzero])) * 100.0) if np.any(nonzero) else np.nan
        bias = float(np.mean(err))
        rows.append(
            {
                "target": target,
                "n": int(len(y)),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE_%": mape,
                "bias": bias,
                "true_mean": float(np.mean(y)),
                "pred_mean": float(np.mean(pred)),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_teacher_forced(
    model: GeoTransformer,
    nodes: pd.DataFrame,
    checkpoint: dict,
    sequences: list[np.ndarray],
    device: torch.device,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    feature_columns = checkpoint["feature_columns"]
    scaled_targets = checkpoint["scaled_target_columns"]
    work = nodes.copy()
    scaler = checkpoint["target_scaler"]
    for target in TARGET_COLUMNS:
        work[f"{target}_scaled"] = (pd.to_numeric(work[target], errors="coerce") - scaler["mean"][target]) / scaler["std"][target]
    work, _ = add_scaled_target_baselines(
        work,
        target_columns=TARGET_COLUMNS,
        scaler=scaler,
        mode=checkpoint.get("target_baseline", "zero"),
        allowed_columns=checkpoint.get("target_baseline_columns"),
    )

    dataset = GeoSequenceDataset(
        work,
        condition_columns=feature_columns,
        target_columns=scaled_targets,
        baseline_columns=[f"baseline_{col}" for col in scaled_targets],
        orders=sequences,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    pred_scaled = np.full((len(nodes), len(TARGET_COLUMNS)), np.nan, dtype=np.float32)

    batch_iter = loader
    if tqdm is not None and show_progress:
        batch_iter = tqdm(loader, desc="teacher-forced batches", leave=False)
    for batch in batch_iter:
        order = batch["order"].cpu().numpy()
        batch = {key: value.to(device) for key, value in batch.items() if key != "order"}
        mu, _ = model(
            conditions=batch["conditions"],
            prev_targets=batch["prev_targets"],
            attention_mask=batch["attention_mask"],
        )
        mu_np = mu.detach().cpu().numpy()
        pred_np = mu_np + batch["baseline"].detach().cpu().numpy()
        valid_np = batch["attention_mask"].detach().cpu().numpy()
        for b in range(order.shape[0]):
            valid = valid_np[b]
            pred_scaled[order[b, valid]] = pred_np[b, valid]
    return pred_scaled


@torch.no_grad()
def predict_autoregressive(
    model: GeoTransformer,
    nodes: pd.DataFrame,
    checkpoint: dict,
    sequences: list[np.ndarray],
    device: torch.device,
    show_progress: bool,
    token_progress: bool,
) -> np.ndarray:
    feature_columns = checkpoint["feature_columns"]
    work, _ = add_scaled_target_baselines(
        nodes,
        target_columns=TARGET_COLUMNS,
        scaler=checkpoint["target_scaler"],
        mode=checkpoint.get("target_baseline", "zero"),
        allowed_columns=checkpoint.get("target_baseline_columns"),
    )
    conditions = torch.tensor(work[feature_columns].to_numpy(dtype=np.float32), device=device)
    baseline_columns = [f"baseline_{target}_scaled" for target in TARGET_COLUMNS]
    baseline = torch.tensor(work[baseline_columns].to_numpy(dtype=np.float32), device=device)
    pred_scaled = np.full((len(nodes), len(TARGET_COLUMNS)), np.nan, dtype=np.float32)
    seq_iter = sequences
    if tqdm is not None and show_progress:
        seq_iter = tqdm(sequences, desc="autoregressive chunks")
    for seq_idx, seq in enumerate(seq_iter):
        local_conditions = conditions[seq].detach().cpu()
        local_order = torch.arange(len(seq), dtype=torch.long)
        generated = generate_autoregressive(
            model=model,
            conditions=local_conditions,
            order=local_order,
            baseline=baseline[seq].detach().cpu(),
            sample=False,
            progress=token_progress,
            progress_desc=f"chunk {seq_idx + 1}/{len(sequences)} tokens",
        )
        pred_scaled[seq] = generated.detach().cpu().numpy()
    return pred_scaled


def build_eval_frame(nodes: pd.DataFrame, pred_scaled: np.ndarray, checkpoint: dict, domain_name: str) -> pd.DataFrame:
    pred = inverse_targets(pred_scaled, checkpoint)
    out = nodes[["X", "Y", "Z", "domain", "experiment_role"]].copy()
    out["eval_domain"] = domain_name
    for target in TARGET_COLUMNS:
        out[f"true_{target}"] = nodes[target].to_numpy()
        out[f"pred_{target}"] = pred[target].to_numpy()
        out[f"error_{target}"] = out[f"pred_{target}"] - out[f"true_{target}"]
    return out


def plot_xy_maps(df: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path, max_points: int, seed: int) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if len(df) > max_points:
        plot_df = df.iloc[rng.choice(len(df), size=max_points, replace=False)].copy()
    else:
        plot_df = df.copy()

    for target in TARGET_COLUMNS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        columns = [f"true_{target}", f"pred_{target}", f"error_{target}"]
        titles = [f"{target} true", f"{target} pred", f"{target} error"]
        for ax, col, title in zip(axes, columns, titles):
            values = plot_df[col].to_numpy(dtype=float)
            if "error" in col:
                vmax = np.nanpercentile(np.abs(values), 98)
                vmin = -vmax
                cmap = "coolwarm"
            else:
                vmin = np.nanpercentile(values, 2)
                vmax = np.nanpercentile(values, 98)
                cmap = "viridis"
            sc = ax.scatter(
                plot_df["X"],
                plot_df["Y"],
                c=values,
                s=2,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
            )
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        metric_row = metrics.loc[metrics["target"] == target]
        if not metric_row.empty:
            row = metric_row.iloc[0]
            fig.suptitle(
                f"{target}: MAE={row['MAE']:.4g}, RMSE={row['RMSE']:.4g}, R2={row['R2']:.4g}",
                fontsize=12,
            )
        fig.savefig(plot_dir / f"xy_{target}.png", dpi=180)
        plt.close(fig)


def evaluate_domain(
    domain_name: str,
    nodes: pd.DataFrame,
    model: GeoTransformer,
    checkpoint: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    nodes = nodes.loc[nodes["has_targets"]].reset_index(drop=True)
    if len(nodes) == 0:
        print(f"{domain_name}: no known target rows, skipping")
        return

    sequence_length = args.sequence_length or int(checkpoint.get("sequence_length", 512))
    order_name = checkpoint.get("order", "domain_strike") if args.order == "checkpoint" else args.order

    def make_sequences_for_order(name: str, seed_offset: int = 0) -> list[np.ndarray]:
        order = make_order(nodes, name, args.seed + seed_offset)
        seqs = chunk_order(order, sequence_length)
        if args.max_sequences > 0:
            if args.sample_sequences and len(seqs) > args.max_sequences:
                rng = np.random.default_rng(args.seed + seed_offset)
                selected = rng.choice(len(seqs), size=args.max_sequences, replace=False)
                seqs = [seqs[int(idx)] for idx in selected]
            else:
                seqs = seqs[: args.max_sequences]
        return seqs

    sequences = make_sequences_for_order(order_name)

    print(
        f"{domain_name}: rows={len(nodes)}, sequences={len(sequences)}, "
        f"sequence_length={sequence_length}, mode={args.mode}, order={order_name}"
    )
    if args.mode == "teacher_forced":
        pred_scaled = predict_teacher_forced(
            model=model,
            nodes=nodes,
            checkpoint=checkpoint,
            sequences=sequences,
            device=device,
            batch_size=args.batch_size,
            show_progress=not args.no_progress,
        )
    else:
        ensemble_order_names = [name.strip() for name in args.ensemble_orders.split(",") if name.strip()]
        if ensemble_order_names:
            preds = []
            for idx, ens_order in enumerate(ensemble_order_names):
                ens_sequences = make_sequences_for_order(ens_order, seed_offset=idx)
                print(f"AR ensemble order {idx + 1}/{len(ensemble_order_names)}: {ens_order}")
                preds.append(
                    predict_autoregressive(
                        model=model,
                        nodes=nodes,
                        checkpoint=checkpoint,
                        sequences=ens_sequences,
                        device=device,
                        show_progress=not args.no_progress,
                        token_progress=args.token_progress and not args.no_progress,
                    )
                )
            stacked = np.stack(preds, axis=0)
            finite = np.isfinite(stacked)
            counts = finite.sum(axis=0)
            sums = np.where(finite, stacked, 0.0).sum(axis=0)
            pred_scaled = np.divide(
                sums,
                counts,
                out=np.full_like(sums, np.nan, dtype=np.float32),
                where=counts > 0,
            )
        else:
            pred_scaled = predict_autoregressive(
                model=model,
                nodes=nodes,
                checkpoint=checkpoint,
                sequences=sequences,
                device=device,
                show_progress=not args.no_progress,
                token_progress=args.token_progress and not args.no_progress,
            )

    valid = np.isfinite(pred_scaled).all(axis=1)
    eval_frame = build_eval_frame(nodes.loc[valid].reset_index(drop=True), pred_scaled[valid], checkpoint, domain_name)
    domain_dir = args.output_dir / f"{domain_name}_{args.mode}"
    domain_dir.mkdir(parents=True, exist_ok=True)
    eval_frame.to_csv(domain_dir / "predictions.csv", index=False)

    metrics = compute_metrics(eval_frame)
    metrics.to_csv(domain_dir / "metrics.csv", index=False)
    print(f"\n{domain_name} metrics ({args.mode}):")
    print(metrics.to_string(index=False))

    plot_xy_maps(eval_frame, metrics, domain_dir, args.max_plot_points, args.seed)
    print(f"Saved predictions/metrics/plots to: {domain_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = restore_model(checkpoint, device)

    domains = []
    if args.domain in {"center", "both"}:
        center = pd.read_parquet(args.prepared_dir / "center_nodes.parquet")
        center = center.loc[center["experiment_role"] == "train_center_known"].reset_index(drop=True)
        domains.append(("center", center))
    if args.domain in {"north", "both"}:
        north = pd.read_parquet(args.prepared_dir / "north_nodes.parquet")
        north = north.loc[north["experiment_role"] == "eval_north_known"].reset_index(drop=True)
        domains.append(("north_known", north))

    for domain_name, nodes in domains:
        evaluate_domain(domain_name, nodes, model, checkpoint, args, device)


if __name__ == "__main__":
    main()
