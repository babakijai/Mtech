from __future__ import annotations

import pandas as pd


def department_baselines(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("department", dropna=False)
        .agg(
            department_avg_charge=("charge_amount_USD", "mean"),
            department_median_charge=("charge_amount_USD", "median"),
            department_avg_payment=("payment_amount_USD", "mean"),
            department_count=("patient_id", "count"),
        )
        .reset_index()
    )


def explain_record(record: pd.Series, baselines: pd.DataFrame) -> dict:
    department = record["department"]
    baseline = baselines[baselines["department"] == department].iloc[0]
    avg_charge = float(baseline["department_avg_charge"])
    charge = float(record["charge_amount_USD"])
    payment = float(record["payment_amount_USD"])
    percent_delta = ((charge - avg_charge) / avg_charge * 100) if avg_charge else 0.0

    reasons: list[str] = []
    if percent_delta > 100:
        reasons.append(f"charge is {percent_delta:.1f}% above the department average")
    elif percent_delta > 50:
        reasons.append(f"charge is materially above the department average by {percent_delta:.1f}%")
    if payment == 0 and charge > 0:
        reasons.append("no payment has been recorded against a positive charge")
    if str(record.get("claim_status", "")).lower() in {"denied", "rejected"}:
        reasons.append(f"claim status is {record['claim_status']}")
    if not reasons:
        reasons.append("combined feature pattern is unusual compared with peer records")

    return {
        "patient_id": str(record["patient_id"]),
        "department": department,
        "diagnosis": record["diagnosis"],
        "charge_amount_USD": charge,
        "payment_amount_USD": payment,
        "department_avg_charge": avg_charge,
        "anomaly_score": float(record.get("anomaly_score", 0)),
        "reason": "; ".join(reasons),
        "summary": (
            f"Patient {record['patient_id']} was flagged in {department}. "
            f"Charge amount ${charge:,.2f}; department average ${avg_charge:,.2f}. "
            f"Reason: {'; '.join(reasons)}."
        ),
    }


def explain_anomalies(scored_df: pd.DataFrame) -> list[dict]:
    baselines = department_baselines(scored_df)
    anomalies = scored_df[scored_df["anomaly_flag"]].copy()
    anomalies = anomalies.sort_values("anomaly_score")
    return [explain_record(row, baselines) for _, row in anomalies.iterrows()]
