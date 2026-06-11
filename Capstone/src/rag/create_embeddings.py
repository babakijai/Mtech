from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.anomaly.anomaly_explainer import explain_anomalies


def report_to_document(report: dict) -> str:
    return "\n".join(
        [
            f"Patient {report['patient_id']}",
            f"Department: {report['department']}",
            f"Diagnosis: {report['diagnosis']}",
            f"Charge Amount: {report['charge_amount_USD']:.2f}",
            f"Department Average: {report['department_avg_charge']:.2f}",
            f"Anomaly Score: {report['anomaly_score']:.6f}",
            f"Reason: {report['reason']}",
        ]
    )


def build_investigation_documents(scored_df: pd.DataFrame) -> list[dict]:
    documents = []
    for report in explain_anomalies(scored_df):
        documents.append(
            {
                "id": f"patient-{report['patient_id']}",
                "text": report_to_document(report),
                "metadata": report,
            }
        )
    return documents


def save_documents(documents: list[dict], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document) + "\n")


def load_documents(path: str | Path) -> list[dict]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    with input_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    from src.ingestion.load_data import load_csv
    from src.models.predict import predict_anomalies

    raw = load_csv("data/raw/Hospital_billing.csv")
    scored = predict_anomalies(raw)
    docs = build_investigation_documents(scored)
    save_documents(docs, "data/processed/anomaly_reports.jsonl")
    print(f"Generated {len(docs):,} anomaly reports")
