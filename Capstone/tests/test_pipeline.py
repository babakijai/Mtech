from __future__ import annotations

import pytest

from src.anomaly.anomaly_explainer import explain_anomalies
from src.evaluation.validation import create_review_sample, evaluate_labeled_predictions
from src.ingestion.load_data import load_csv
from src.models.predict import predict_anomalies
from src.models.train_isolation_forest import train_isolation_forest


def test_training_prediction_and_explanations(tmp_path):
    model_path = tmp_path / "model.pkl"
    processed_path = tmp_path / "billing_cleaned.csv"

    train_isolation_forest(
        raw_data_path="data/raw/Hospital_billing.csv",
        model_path=model_path,
        processed_path=processed_path,
        contamination=0.05,
    )
    scored = predict_anomalies(load_csv("data/raw/Hospital_billing.csv"), model_path=model_path)

    assert "anomaly_score" in scored.columns
    assert "anomaly_flag" in scored.columns
    assert scored["anomaly_flag"].sum() > 0
    assert explain_anomalies(scored)


def test_review_sample_and_validation_metrics():
    scored = load_csv("data/raw/Hospital_billing.csv").head(8)
    scored["anomaly_score"] = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
    scored["anomaly_flag"] = [True, True, False, False, True, False, False, False]
    scored["anomaly_label"] = scored["anomaly_flag"].map({True: "anomaly", False: "normal"})

    sample = create_review_sample(
        scored,
        flagged_count=2,
        normal_count=2,
        random_count=2,
        random_state=1,
    )

    assert {"reviewer_label", "reviewer_notes", "_review_bucket"}.issubset(sample.columns)
    assert len(sample) == 6

    labeled = scored.copy()
    labeled["anomaly_flag"] = labeled["anomaly_flag"].astype(str)
    labeled["reviewer_label"] = [
        "true_anomaly",
        "not_anomaly",
        "not_anomaly",
        "true_anomaly",
        "true_anomaly",
        "not_anomaly",
        "needs_review",
        "needs_review",
    ]

    metrics = evaluate_labeled_predictions(labeled)

    assert metrics.reviewed_records == 6
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 2
    assert metrics.false_negatives == 1
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.false_positive_rate == pytest.approx(1 / 3)
