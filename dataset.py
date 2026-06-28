from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .model import shift_targets_right


@dataclass
class SequenceBatch:
    conditions: torch.Tensor
    targets: torch.Tensor
    prev_targets: torch.Tensor
    baseline: torch.Tensor
    attention_mask: torch.Tensor
    order: torch.Tensor


class GeoSequenceDataset(Dataset):
    """Table-backed teacher-forcing dataset for causal field generation.

    Each item returns one ordered sequence. For large deposits, pass precomputed
    patch node ids as `sequences`.
    """

    def __init__(
        self,
        nodes: pd.DataFrame,
        condition_columns: Sequence[str],
        target_columns: Sequence[str],
        baseline_columns: Sequence[str] | None = None,
        sequences: Sequence[Sequence[int]] | None = None,
        orders: Sequence[Sequence[int]] | None = None,
    ):
        self.nodes = nodes.reset_index(drop=True)
        self.condition_columns = list(condition_columns)
        self.target_columns = list(target_columns)
        self.baseline_columns = list(baseline_columns or [])
        self.sequences = list(sequences) if sequences is not None else [np.arange(len(nodes))]
        self.orders = list(orders) if orders is not None else self.sequences

        for cols, name in [
            (self.condition_columns, "condition"),
            (self.target_columns, "target"),
            (self.baseline_columns, "baseline"),
        ]:
            missing = [col for col in cols if col not in self.nodes.columns]
            if missing:
                raise ValueError(f"Missing {name} columns: {missing}")

    def __len__(self) -> int:
        return len(self.orders)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        order = np.asarray(self.orders[idx], dtype=np.int64)
        df = self.nodes.iloc[order]

        conditions = torch.tensor(
            df[self.condition_columns].to_numpy(),
            dtype=torch.float32,
        )
        targets = torch.tensor(
            df[self.target_columns].to_numpy(),
            dtype=torch.float32,
        )
        if self.baseline_columns:
            baseline = torch.tensor(
                df[self.baseline_columns].to_numpy(),
                dtype=torch.float32,
            )
        else:
            baseline = torch.zeros_like(targets)

        prev_targets = shift_targets_right(targets.unsqueeze(0)).squeeze(0)
        attention_mask = torch.ones(len(order), dtype=torch.bool)
        return {
            "conditions": conditions,
            "targets": targets,
            "prev_targets": prev_targets,
            "baseline": baseline,
            "attention_mask": attention_mask,
            "order": torch.tensor(order, dtype=torch.long),
        }


def collate_sequences(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences for batching."""

    max_len = max(item["conditions"].size(0) for item in items)
    batch = {}
    for key in ["conditions", "targets", "prev_targets", "baseline"]:
        dim = items[0][key].size(-1)
        out = torch.zeros(len(items), max_len, dim, dtype=items[0][key].dtype)
        for i, item in enumerate(items):
            out[i, : item[key].size(0)] = item[key]
        batch[key] = out

    mask = torch.zeros(len(items), max_len, dtype=torch.bool)
    order = torch.full((len(items), max_len), fill_value=-1, dtype=torch.long)
    for i, item in enumerate(items):
        n = item["attention_mask"].size(0)
        mask[i, :n] = item["attention_mask"]
        order[i, :n] = item["order"]
    batch["attention_mask"] = mask
    batch["order"] = order
    return batch

