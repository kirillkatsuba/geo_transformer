from __future__ import annotations

import numpy as np
import pandas as pd


def candidate_baseline_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Find raw baseline prediction columns for one target.

    `prepare_customer_data --baseline v1=...` creates columns like `v1_AS`.
    The model can use their mean as a deterministic trend and learn an
    autoregressive residual field on top.
    """

    suffix = f"_{target}"
    excluded = {
        target,
        f"{target}_scaled",
        f"baseline_{target}_scaled",
        f"true_{target}",
        f"pred_{target}",
        f"error_{target}",
    }
    cols = []
    allowed_prefixes = ("v1_", "v2_", "dnn_", "gp_", "ensemble_", "baseline_")
    for col in df.columns:
        if col in excluded:
            continue
        if not col.startswith(allowed_prefixes):
            continue
        if col == f"baseline_{target}" or col.endswith(suffix):
            cols.append(col)
    return cols


def add_scaled_target_baselines(
    df: pd.DataFrame,
    target_columns: list[str],
    scaler: dict[str, dict[str, float]],
    mode: str = "zero",
    allowed_columns: dict[str, list[str]] | None = None,
    scale_guard: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add `baseline_{target}_scaled` columns used by residual training/eval."""

    out = df.copy()
    used: dict[str, list[str]] = {}
    for target in target_columns:
        out_col = f"baseline_{target}_scaled"
        if mode == "zero":
            out[out_col] = 0.0
            used[target] = []
            continue
        if mode != "mean_baselines":
            raise ValueError(f"Unknown target baseline mode: {mode!r}")

        if allowed_columns is not None:
            cols = [col for col in allowed_columns.get(target, []) if col in out.columns]
        else:
            cols = candidate_baseline_columns(out, target)
            if scale_guard and target in out.columns:
                target_values = pd.to_numeric(out[target], errors="coerce").to_numpy(dtype=float)
                target_scale = np.nanpercentile(np.abs(target_values), 95)
                if np.isfinite(target_scale):
                    compatible_cols = []
                    for col in cols:
                        col_values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
                        col_scale = np.nanpercentile(np.abs(col_values), 95)
                        if np.isfinite(col_scale) and col_scale <= max(10.0 * target_scale, target_scale + 1e-6):
                            compatible_cols.append(col)
                    cols = compatible_cols
        used[target] = cols
        if not cols:
            out[out_col] = 0.0
            continue

        raw = out[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        raw = raw.fillna(float(scaler["mean"][target]))
        out[out_col] = (raw - float(scaler["mean"][target])) / float(scaler["std"][target])

    return out, used
