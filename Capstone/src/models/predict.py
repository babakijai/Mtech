from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_engineering import select_model_columns
from src.preprocessing.preprocess import preprocess_billing_data


def load_model(model_path: str | Path = "src/models/model.pkl"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def predict_anomalies(df: pd.DataFrame, model_path: str | Path = "src/models/model.pkl") -> pd.DataFrame:
    """Attach anomaly scores and labels to raw or cleaned billing records."""
    model = load_model(model_path)
    scored = preprocess_billing_data(df)
    model_input = select_model_columns(scored)
    predictions = model.predict(model_input)
    scored["anomaly_score"] = model.decision_function(model_input)
    scored["anomaly_flag"] = predictions == -1
    scored["anomaly_label"] = scored["anomaly_flag"].map({True: "anomaly", False: "normal"})
    return scored


if __name__ == "__main__":
    from src.ingestion.load_data import load_csv

    raw = load_csv("data/raw/Hospital_billing.csv")
    result = predict_anomalies(raw)
    result.to_csv("data/processed/billing_scored.csv", index=False)
    print(f"Detected {int(result['anomaly_flag'].sum()):,} anomalies")
