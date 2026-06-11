from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


POSITIVE_LABELS = {"true_anomaly", "anomaly", "fraud", "billing_error", "yes", "1", "true"}
NEGATIVE_LABELS = {"not_anomaly", "normal", "no_issue", "no", "0", "false"}
UNREVIEWED_LABELS = {"needs_review", "unknown", "unreviewed", ""}


@dataclass(frozen=True)
class ValidationMetrics:
    reviewed_records: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float


def normalize_reviewer_label(label: object) -> bool | None:
    """Convert reviewer labels into binary ground truth values."""
    normalized = str(label).strip().lower()
    if normalized in POSITIVE_LABELS:
        return True
    if normalized in NEGATIVE_LABELS:
        return False
    if normalized in UNREVIEWED_LABELS:
        return None
    raise ValueError(
        "Unsupported reviewer label "
        f"{label!r}. Use true_anomaly, not_anomaly, or needs_review."
    )


def normalize_prediction(value: object) -> bool:
    """Convert saved anomaly predictions into booleans."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "anomaly"}:
        return True
    if normalized in {"false", "0", "no", "normal"}:
        return False
    raise ValueError(f"Unsupported prediction value {value!r}. Expected boolean-like values.")


def create_review_sample(
    scored_df: pd.DataFrame,
    flagged_count: int = 50,
    normal_count: int = 50,
    random_count: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a balanced review sample from flagged, normal, and random records."""
    required = {"anomaly_flag", "anomaly_score"}
    missing = required.difference(scored_df.columns)
    if missing:
        raise ValueError(f"Scored data is missing required columns: {sorted(missing)}")

    scored = scored_df.copy()
    scored["_review_bucket"] = "random"

    flagged = (
        scored[scored["anomaly_flag"]]
        .sort_values("anomaly_score")
        .head(flagged_count)
        .assign(_review_bucket="flagged")
    )
    normal = (
        scored[~scored["anomaly_flag"]]
        .sort_values("anomaly_score", ascending=False)
        .head(normal_count)
        .assign(_review_bucket="normal")
    )

    used_index = flagged.index.union(normal.index)
    remaining = scored.drop(index=used_index)
    sample_size = min(random_count, len(remaining))
    random = remaining.sample(n=sample_size, random_state=random_state) if sample_size else remaining

    review_sample = pd.concat([flagged, normal, random], ignore_index=True)
    review_sample["reviewer_label"] = "needs_review"
    review_sample["reviewer_notes"] = ""

    preferred_columns = [
        "patient_id",
        "department",
        "diagnosis",
        "claim_status",
        "charge_amount_USD",
        "payment_amount_USD",
        "anomaly_score",
        "anomaly_flag",
        "anomaly_label",
        "_review_bucket",
        "reviewer_label",
        "reviewer_notes",
    ]
    ordered_columns = [column for column in preferred_columns if column in review_sample.columns]
    remaining_columns = [column for column in review_sample.columns if column not in ordered_columns]
    return review_sample[ordered_columns + remaining_columns]


def evaluate_labeled_predictions(
    labeled_df: pd.DataFrame,
    label_column: str = "reviewer_label",
    prediction_column: str = "anomaly_flag",
) -> ValidationMetrics:
    """Calculate metrics by comparing anomaly predictions with reviewer labels."""
    required = {label_column, prediction_column}
    missing = required.difference(labeled_df.columns)
    if missing:
        raise ValueError(f"Labeled data is missing required columns: {sorted(missing)}")

    evaluated = labeled_df.copy()
    evaluated["_ground_truth"] = evaluated[label_column].map(normalize_reviewer_label)
    evaluated = evaluated[evaluated["_ground_truth"].notna()]
    if evaluated.empty:
        raise ValueError("No reviewed true_anomaly or not_anomaly labels were found.")

    y_true = evaluated["_ground_truth"].astype(bool)
    y_pred = evaluated[prediction_column].map(normalize_prediction)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return ValidationMetrics(
        reviewed_records=int(len(evaluated)),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        false_positive_rate=float(false_positive_rate),
    )


def save_review_sample(sample: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or evaluate anomaly review labels.")
    parser.add_argument("--scored-path", default="data/processed/billing_scored.csv")
    parser.add_argument("--labels-path", default="data/processed/anomaly_review_labels.csv")
    parser.add_argument("--mode", choices=["sample", "evaluate"], default="sample")
    parser.add_argument("--flagged-count", type=int, default=50)
    parser.add_argument("--normal-count", type=int, default=50)
    parser.add_argument("--random-count", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "sample":
        scored = pd.read_csv(args.scored_path)
        sample = create_review_sample(
            scored,
            flagged_count=args.flagged_count,
            normal_count=args.normal_count,
            random_count=args.random_count,
        )
        save_review_sample(sample, args.labels_path)
        print(f"Saved review sample to {args.labels_path} with {len(sample):,} records")
        return

    labeled = pd.read_csv(args.labels_path)
    metrics = evaluate_labeled_predictions(labeled)
    for key, value in asdict(metrics).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
