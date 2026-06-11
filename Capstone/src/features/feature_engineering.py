from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "payment_amount_USD",
    "charge_amount_USD",
    "length_of_stay_days",
    "payment_ratio",
    "unpaid_amount_USD",
]
CATEGORICAL_FEATURES = [
    "department",
    "diagnosis",
    "visit_reason",
    "is_emergency",
    "claim_status",
    "visit_type",
    "gender",
]


@dataclass(frozen=True)
class FeatureConfig:
    numeric_features: list[str]
    categorical_features: list[str]


DEFAULT_FEATURE_CONFIG = FeatureConfig(NUMERIC_FEATURES, CATEGORICAL_FEATURES)


def build_feature_pipeline(config: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> Pipeline:
    """Create the preprocessing pipeline used before Isolation Forest."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), config.numeric_features),
            ("categorical", encoder, config.categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline([("features", transformer)])


def select_model_columns(
    df: pd.DataFrame, config: FeatureConfig = DEFAULT_FEATURE_CONFIG
) -> pd.DataFrame:
    required = config.numeric_features + config.categorical_features
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required model columns: {missing}")
    return df[required]
