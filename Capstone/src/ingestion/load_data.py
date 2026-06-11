from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_COLUMNS = [
    "patient_id",
    "gender",
    "age",
    "visit_date",
    "department",
    "physician_id",
    "diagnosis",
    "visit_type",
    "visit_reason",
    "appointment_id",
    "is_emergency",
    "insurance_id",
    "payer_name",
    "payer_type",
    "claim_status",
    "charge_amount_USD",
    "payment_amount_USD",
    "admission_date",
    "discharge_date",
]


def validate_schema(df: pd.DataFrame, expected_columns: Iterable[str] = EXPECTED_COLUMNS) -> None:
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a hospital billing CSV and validate its schema."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    validate_schema(df)
    return df


if __name__ == "__main__":
    frame = load_csv("data/raw/Hospital_billing.csv")
    print(f"Loaded {len(frame):,} rows and {len(frame.columns):,} columns")
