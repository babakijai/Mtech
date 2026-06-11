from __future__ import annotations

from pathlib import Path

import joblib
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.features.feature_engineering import build_feature_pipeline, select_model_columns
from src.ingestion.load_data import load_csv
from src.preprocessing.preprocess import preprocess_billing_data, save_processed_data


def train_isolation_forest(
    raw_data_path: str | Path = "data/raw/Hospital_billing.csv",
    model_path: str | Path = "src/models/model.pkl",
    processed_path: str | Path = "data/processed/billing_cleaned.csv",
    contamination: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    """Train and persist the Isolation Forest anomaly detection pipeline."""
    raw = load_csv(raw_data_path)
    cleaned = preprocess_billing_data(raw)
    save_processed_data(cleaned, processed_path)

    pipeline = Pipeline(
        [
            ("feature_pipeline", build_feature_pipeline()),
            (
                "model",
                IsolationForest(
                    contamination=contamination,
                    random_state=random_state,
                    n_estimators=200,
                ),
            ),
        ]
    )
    pipeline.fit(select_model_columns(cleaned))

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return pipeline


def train_from_config(config_path: str | Path = "config.yaml") -> Pipeline:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return train_isolation_forest(
        raw_data_path=config["data"]["raw_path"],
        model_path=config["model"]["artifact_path"],
        processed_path=config["data"]["processed_path"],
        contamination=float(config["model"]["contamination"]),
        random_state=int(config["model"]["random_state"]),
    )


if __name__ == "__main__":
    train_from_config()
    print("Saved model artifact")
