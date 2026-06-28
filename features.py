from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import BLOCK_SIZE_COLUMNS, COORD_COLUMNS, TARGET_COLUMNS


DEFAULT_NUMERIC_FEATURES = [
    *COORD_COLUMNS,
    "coord_strike",
    "coord_cross",
    *BLOCK_SIZE_COLUMNS,
    "Au_Final",
    "volume",
    "DENSITY",
    "RESCAT",
    "ZONE",
    "PVALUE",
    "IND",
    "RESCAT_C",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "domain",
    "MODAREA",
    "MINED",
]


@dataclass
class GeoFeatureBuilder:
    """Simple table feature builder for first Transformer experiments.

    This intentionally avoids a heavy sklearn dependency. It keeps medians,
    means, standard deviations and one-hot categorical levels learned on the
    training table, then applies the same schema to extrapolation tables.
    """

    numeric_features: list[str] = field(default_factory=lambda: list(DEFAULT_NUMERIC_FEATURES))
    categorical_features: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORICAL_FEATURES))
    include_target_availability: bool = True
    medians_: dict[str, float] = field(default_factory=dict)
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)
    categories_: dict[str, list[str]] = field(default_factory=dict)
    output_columns_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "GeoFeatureBuilder":
        numeric_feature_set = set(self.numeric_features)
        numeric_feature_set.update(
            col
            for col in df.columns
            if col.startswith("knn_")
            or col.startswith("baseline_")
            or col.startswith("v1_")
            or col.startswith("v2_")
            or col.startswith("dnn_")
            or col.startswith("gp_")
            or col.startswith("ensemble_")
        )
        numeric_cols = [col for col in df.columns if col in numeric_feature_set]
        categorical_cols = [col for col in self.categorical_features if col in df.columns]

        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.medians_[col] = median
            self.means_[col] = mean
            self.stds_[col] = std if std > 1e-12 else 1.0

        for col in categorical_cols:
            values = df[col].astype("string").fillna("__MISSING__")
            self.categories_[col] = sorted(values.unique().tolist())

        transformed = self.transform(df)
        self.output_columns_ = transformed.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []

        for col, median in self.medians_.items():
            values = pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype="float64")), errors="coerce")
            standardized = (values.fillna(median) - self.means_[col]) / self.stds_[col]
            parts.append(pd.DataFrame({col: standardized}, index=df.index))

        for col, categories in self.categories_.items():
            values = df.get(col, pd.Series(index=df.index, dtype="object")).astype("string").fillna("__MISSING__")
            encoded = {
                f"{col}={category}": (values == category).astype(float)
                for category in categories
            }
            parts.append(pd.DataFrame(encoded, index=df.index))

        if self.include_target_availability:
            availability = {
                f"known_{target}": df[target].notna().astype(float)
                for target in TARGET_COLUMNS
                if target in df.columns
            }
            if "has_targets" in df.columns:
                availability["has_targets"] = df["has_targets"].astype(float)
            parts.append(pd.DataFrame(availability, index=df.index))

        if not parts:
            return pd.DataFrame(index=df.index)

        out = pd.concat(parts, axis=1)
        if self.output_columns_:
            for col in self.output_columns_:
                if col not in out.columns:
                    out[col] = 0.0
            out = out[self.output_columns_]
        return out.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def attach_feature_matrix(
    nodes: pd.DataFrame,
    features: pd.DataFrame,
    prefix: str = "feat_",
) -> pd.DataFrame:
    """Attach feature matrix columns to a node table."""

    out = nodes.copy()
    renamed = features.rename(columns={col: f"{prefix}{col}" for col in features.columns})
    return pd.concat([out.reset_index(drop=True), renamed.reset_index(drop=True)], axis=1)


def target_matrix(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """Return targets in standard order, filling missing values for tensors."""

    return df[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(fill_value)
