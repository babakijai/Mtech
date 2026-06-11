from __future__ import annotations

from pathlib import Path

import pandas as pd


DATE_COLUMNS = ["visit_date", "admission_date", "discharge_date"]
NUMERIC_COLUMNS = ["age", "charge_amount_USD", "payment_amount_USD"]
TEXT_COLUMNS = [
    "gender",
    "department",
    "diagnosis",
    "visit_type",
    "visit_reason",
    "appointment_id",
    "is_emergency",
    "insurance_id",
    "payer_name",
    "payer_type",
    "claim_status",
]


def _mode_or_missing(series: pd.Series) -> str:
    modes = series.dropna().mode()
    if modes.empty:
        return "missing"
    return str(modes.iloc[0])


def preprocess_billing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw billing data using the notebook's imputation strategy."""
    cleaned = df.copy()

    for column in DATE_COLUMNS:
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce", dayfirst=True)

    for column in TEXT_COLUMNS:
        if column == "insurance_id":
            cleaned[column] = cleaned[column].fillna("missing")
        else:
            cleaned[column] = cleaned[column].fillna(_mode_or_missing(cleaned[column]))
        cleaned[column] = cleaned[column].astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned["length_of_stay_days"] = (
        cleaned["discharge_date"] - cleaned["admission_date"]
    ).dt.days.fillna(0)
    cleaned["length_of_stay_days"] = cleaned["length_of_stay_days"].clip(lower=0)
    cleaned["payment_ratio"] = (
        cleaned["payment_amount_USD"] / cleaned["charge_amount_USD"].replace(0, pd.NA)
    ).fillna(0)
    cleaned["unpaid_amount_USD"] = (
        cleaned["charge_amount_USD"] - cleaned["payment_amount_USD"]
    ).clip(lower=0)

    return cleaned


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    from src.ingestion.load_data import load_csv

    raw = load_csv("data/raw/Hospital_billing.csv")
    processed = preprocess_billing_data(raw)
    save_processed_data(processed, "data/processed/billing_cleaned.csv")
    print(f"Saved cleaned data with {len(processed):,} rows")
