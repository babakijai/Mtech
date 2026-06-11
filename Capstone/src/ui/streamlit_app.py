from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import yaml
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.graph import InvestigationGraph
from src.features.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, select_model_columns


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "billing_scored.csv"
REPORTS_PATH = PROJECT_ROOT / "data" / "processed" / "anomaly_reports.jsonl"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "model.pkl"


st.set_page_config(
    page_title="Hospital Billing Dashboard",
    page_icon="HB",
    layout="wide",
)


def inject_bootstrap_theme() -> None:
    st.markdown(
        """
        <link
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
          rel="stylesheet"
        >
        <style>
            :root {
                --hb-ink: #1d2733;
                --hb-muted: #64748b;
                --hb-line: #d9e2ec;
                --hb-bg: #f6f8fb;
                --hb-panel: #ffffff;
                --hb-blue: #2563eb;
                --hb-teal: #0f766e;
                --hb-amber: #b45309;
                --hb-red: #b91c1c;
            }

            .stApp {
                background:
                    linear-gradient(180deg, #f8fbff 0%, var(--hb-bg) 38%, #eef4f7 100%);
                color: var(--hb-ink);
            }

            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--hb-line);
            }

            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1480px;
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            div[data-testid="stMetric"] {
                background: var(--hb-panel);
                border: 1px solid var(--hb-line);
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 10px 26px rgba(29, 39, 51, 0.06);
            }

            div[data-testid="stMetric"] label {
                color: var(--hb-muted);
            }

            .hb-hero {
                background: linear-gradient(135deg, #123c69 0%, #116466 62%, #4f6f52 100%);
                color: #ffffff;
                border-radius: 8px;
                padding: 1.3rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 16px 34px rgba(18, 60, 105, 0.18);
            }

            .hb-hero h1 {
                font-size: clamp(1.6rem, 3vw, 2.35rem);
                line-height: 1.15;
                margin: 0 0 .4rem 0;
                font-weight: 750;
            }

            .hb-hero p {
                max-width: 880px;
                margin: 0;
                color: rgba(255, 255, 255, .86);
                font-size: 1rem;
            }

            .hb-section-title {
                color: var(--hb-ink);
                font-size: 1.05rem;
                font-weight: 750;
                margin: .35rem 0 .7rem;
            }

            .hb-card {
                background: var(--hb-panel);
                border: 1px solid var(--hb-line);
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 10px 26px rgba(29, 39, 51, 0.06);
            }

            .hb-alert {
                border-left: 4px solid var(--hb-red);
                background: #fff7f7;
                border-radius: 8px;
                padding: .85rem 1rem;
                margin: .35rem 0 .85rem;
                color: #7f1d1d;
            }

            .hb-note {
                border-left: 4px solid var(--hb-teal);
                background: #f0fdfa;
                border-radius: 8px;
                padding: .85rem 1rem;
                margin: .35rem 0 .85rem;
                color: #134e4a;
            }

            .hb-table-small {
                font-size: .9rem;
            }

            .stChatMessage {
                border: 1px solid var(--hb-line);
                border-radius: 8px;
                background: #ffffff;
            }

            .hb-rag-title {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: .75rem;
                margin-bottom: .35rem;
            }

            .hb-rag-title strong {
                color: var(--hb-ink);
                font-size: 1rem;
            }

            .hb-rag-title span {
                background: #dcfce7;
                color: #166534;
                border-radius: 999px;
                font-size: .72rem;
                font-weight: 700;
                padding: .2rem .55rem;
                white-space: nowrap;
            }

            .st-key-rag_chat_toggle button {
                min-height: 2rem;
                width: 2rem;
                padding: .2rem;
                border-radius: 999px;
                border: 1px solid var(--hb-line);
                background: #f8fafc;
                color: var(--hb-ink);
                font-size: 1.15rem;
                line-height: 1;
                font-weight: 700;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }

            .hb-rag-hint {
                color: var(--hb-muted);
                font-size: .82rem;
                line-height: 1.35;
                margin-bottom: .65rem;
            }

            .st-key-floating_rag_chat {
                background: #ffffff;
                border: 1px solid var(--hb-line);
                border-radius: 8px;
                box-shadow: 0 18px 44px rgba(15, 23, 42, .18);
                padding: .95rem;
            }

            .st-key-floating_rag_chat .stChatMessage {
                padding: .35rem .55rem;
                margin-bottom: .5rem;
            }

            .st-key-floating_rag_chat textarea,
            .st-key-floating_rag_chat input {
                font-size: .9rem;
            }

            @media (min-width: 980px) {
                .st-key-floating_rag_chat {
                    position: fixed;
                    right: 1.25rem;
                    bottom: 1.25rem;
                    width: min(390px, calc(100vw - 2.5rem));
                    max-height: 76vh;
                    overflow-y: auto;
                    z-index: 1000;
                }

                .main .block-container {
                    padding-bottom: 21rem;
                }
            }

            @media (max-width: 979px) {
                .st-key-floating_rag_chat {
                    margin-top: 1rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_billing_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    df["anomaly_flag"] = df["anomaly_flag"].astype(bool)
    return df


@st.cache_data
def load_reports() -> list[dict]:
    if not REPORTS_PATH.exists():
        return []

    reports: list[dict] = []
    with REPORTS_PATH.open("r", encoding="utf-8") as report_file:
        for line in report_file:
            if line.strip():
                reports.append(json.loads(line))
    return reports


@st.cache_data
def load_project_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


@st.cache_data
def calculate_unsupervised_metrics(df: pd.DataFrame) -> dict[str, float | str]:
    if not MODEL_PATH.exists():
        return {"error": "Model artifact is missing."}

    labels = df["anomaly_flag"].astype(int)
    if labels.nunique() < 2:
        return {"error": "At least two prediction groups are required for cluster metrics."}
    if labels.nunique() >= len(df):
        return {"error": "Cluster metrics require fewer groups than records."}

    model = joblib.load(MODEL_PATH)
    feature_matrix = model.named_steps["feature_pipeline"].transform(select_model_columns(df))
    if hasattr(feature_matrix, "toarray"):
        feature_matrix = feature_matrix.toarray()

    return {
        "silhouette": float(silhouette_score(feature_matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(feature_matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(feature_matrix, labels)),
    }


def money(value: float) -> str:
    return f"${value:,.0f}"


def percentage(value: float) -> str:
    return f"{value:.1f}%"


def build_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")

        min_date = df["visit_date"].min().date()
        max_date = df["visit_date"].max().date()
        date_range = st.date_input(
            "Visit date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        departments = st.multiselect(
            "Department",
            sorted(df["department"].dropna().unique()),
            default=sorted(df["department"].dropna().unique()),
        )
        payers = st.multiselect(
            "Payer type",
            sorted(df["payer_type"].dropna().unique()),
            default=sorted(df["payer_type"].dropna().unique()),
        )
        claim_statuses = st.multiselect(
            "Claim status",
            sorted(df["claim_status"].dropna().unique()),
            default=sorted(df["claim_status"].dropna().unique()),
        )
        show_anomalies_only = st.toggle("Flagged records only", value=False)

    filtered = df.copy()
    if len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[
            filtered["visit_date"].between(start, end + pd.Timedelta(days=1), inclusive="left")
        ]
    if departments:
        filtered = filtered[filtered["department"].isin(departments)]
    if payers:
        filtered = filtered[filtered["payer_type"].isin(payers)]
    if claim_statuses:
        filtered = filtered[filtered["claim_status"].isin(claim_statuses)]
    if show_anomalies_only:
        filtered = filtered[filtered["anomaly_flag"]]

    return filtered


def render_header(df: pd.DataFrame) -> None:
    start = df["visit_date"].min().strftime("%b %d, %Y")
    end = df["visit_date"].max().strftime("%b %d, %Y")
    st.markdown(
        f"""
        <div class="hb-hero">
            <h1>Hospital Billing Investigation Dashboard</h1>
            <p>
                Review anomaly risk, charge variance, unpaid exposure, and payer patterns
                across {len(df):,} billing records from {start} to {end}.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame) -> None:
    total_records = len(df)
    flagged = int(df["anomaly_flag"].sum())
    anomaly_rate = (flagged / total_records * 100) if total_records else 0
    total_charges = df["charge_amount_USD"].sum()
    unpaid = df["unpaid_amount_USD"].sum()
    payment_ratio = df["payment_ratio"].mean() * 100 if total_records else 0

    cols = st.columns(5)
    cols[0].metric("Records", f"{total_records:,}")
    cols[1].metric("Flagged", f"{flagged:,}", percentage(anomaly_rate))
    cols[2].metric("Charges", money(total_charges))
    cols[3].metric("Unpaid", money(unpaid))
    cols[4].metric("Avg paid", percentage(payment_ratio))


def render_charts(df: pd.DataFrame) -> None:
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="hb-section-title">Anomaly Rate by Department</div>', unsafe_allow_html=True)
        department = (
            df.groupby("department", as_index=False)
            .agg(records=("patient_id", "count"), flagged=("anomaly_flag", "sum"))
            .assign(anomaly_rate=lambda x: x["flagged"] / x["records"] * 100)
            .sort_values("anomaly_rate", ascending=False)
        )
        st.bar_chart(department, x="department", y="anomaly_rate", color="#b91c1c")

    with right:
        st.markdown('<div class="hb-section-title">Claim Status Mix</div>', unsafe_allow_html=True)
        status = (
            df["claim_status"]
            .value_counts()
            .rename_axis("claim_status")
            .reset_index(name="records")
        )
        st.bar_chart(status, x="claim_status", y="records", color="#2563eb")

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="hb-section-title">Charge Trend</div>', unsafe_allow_html=True)
        monthly = (
            df.dropna(subset=["visit_date"])
            .set_index("visit_date")
            .resample("ME")
            .agg(
                charge_amount_USD=("charge_amount_USD", "sum"),
                unpaid_amount_USD=("unpaid_amount_USD", "sum"),
            )
        )
        st.line_chart(monthly)

    with right:
        st.markdown('<div class="hb-section-title">Top Diagnoses by Flagged Volume</div>', unsafe_allow_html=True)
        diagnoses = (
            df[df["anomaly_flag"]]
            .groupby("diagnosis", as_index=False)
            .agg(flagged=("patient_id", "count"), avg_charge=("charge_amount_USD", "mean"))
            .sort_values("flagged", ascending=False)
            .head(10)
        )
        st.bar_chart(diagnoses, x="diagnosis", y="flagged", color="#0f766e")


def render_tables(df: pd.DataFrame, reports: list[dict]) -> None:
    st.markdown('<div class="hb-section-title">Priority Records</div>', unsafe_allow_html=True)

    priority_columns = [
        "patient_id",
        "department",
        "diagnosis",
        "claim_status",
        "charge_amount_USD",
        "payment_amount_USD",
        "unpaid_amount_USD",
        "anomaly_score",
    ]
    priority = (
        df[df["anomaly_flag"]]
        .sort_values(["anomaly_score", "charge_amount_USD"], ascending=[True, False])
        .head(25)
        .loc[:, priority_columns]
    )
    st.dataframe(
        priority,
        use_container_width=True,
        hide_index=True,
        column_config={
            "patient_id": "Patient",
            "charge_amount_USD": st.column_config.NumberColumn("Charge", format="$%d"),
            "payment_amount_USD": st.column_config.NumberColumn("Paid", format="$%d"),
            "unpaid_amount_USD": st.column_config.NumberColumn("Unpaid", format="$%d"),
            "anomaly_score": st.column_config.NumberColumn("Score", format="%.4f"),
        },
    )

    if reports:
        st.markdown('<div class="hb-section-title">Recent Investigation Notes</div>', unsafe_allow_html=True)
        for report in reports[:3]:
            metadata = report.get("metadata", {})
            st.markdown(
                f"""
                <div class="hb-note">
                    <strong>Patient {metadata.get("patient_id", "Unknown")}</strong>
                    &middot; {metadata.get("department", "Unknown department")}
                    &middot; {money(float(metadata.get("charge_amount_USD", 0)))}
                    <br>{metadata.get("reason", "No reason available.")}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_ml_metrics(df: pd.DataFrame) -> None:
    st.markdown('<div class="hb-section-title">Model Performance and Diagnostics</div>', unsafe_allow_html=True)

    config = load_project_config()
    model_config = config.get("model", {})
    expected_contamination = float(model_config.get("contamination", 0.05))
    flagged = int(df["anomaly_flag"].sum())
    total_records = len(df)
    observed_rate = flagged / total_records if total_records else 0
    score_threshold = df.loc[df["anomaly_flag"], "anomaly_score"].max() if flagged else 0

    cols = st.columns(5)
    cols[0].metric("Model", "Isolation Forest")
    cols[1].metric("Training target", percentage(expected_contamination * 100))
    cols[2].metric("Observed flagged", percentage(observed_rate * 100), f"{flagged:,} records")
    cols[3].metric("Score cutoff", f"{score_threshold:.4f}")
    cols[4].metric("Features", f"{len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}")

    unsupervised = calculate_unsupervised_metrics(df)
    if "error" not in unsupervised:
        st.markdown('<div class="hb-section-title">Unsupervised Separation Metrics</div>', unsafe_allow_html=True)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Silhouette Score", f"{unsupervised['silhouette']:.4f}")
        metric_cols[1].metric("Davies-Bouldin Index", f"{unsupervised['davies_bouldin']:.4f}")
        metric_cols[2].metric("Calinski-Harabasz Index", f"{unsupervised['calinski_harabasz']:.2f}")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Metric": "Silhouette Score",
                        "What it indicates": "Separation between anomaly and normal groups",
                        "Preferred direction": "Higher is better",
                    },
                    {
                        "Metric": "Davies-Bouldin Index",
                        "What it indicates": "Average similarity between groups",
                        "Preferred direction": "Lower is better",
                    },
                    {
                        "Metric": "Calinski-Harabasz Index",
                        "What it indicates": "Between-group dispersion versus within-group dispersion",
                        "Preferred direction": "Higher is better",
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown(
            f"""
            <div class="hb-note">
                <strong>Unsupervised metrics are unavailable for this filter.</strong>
                {unsupervised["error"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="hb-section-title">Anomaly Score Distribution</div>', unsafe_allow_html=True)
        score_bins = pd.cut(df["anomaly_score"], bins=20)
        score_distribution = (
            df.assign(score_bucket=score_bins.astype(str))
            .groupby("score_bucket", as_index=False, observed=False)
            .agg(records=("patient_id", "count"))
        )
        st.bar_chart(score_distribution, x="score_bucket", y="records", color="#2563eb")

    with right:
        st.markdown('<div class="hb-section-title">Prediction Breakdown</div>', unsafe_allow_html=True)
        prediction_mix = (
            df["anomaly_label"]
            .value_counts()
            .rename_axis("prediction")
            .reset_index(name="records")
        )
        st.bar_chart(prediction_mix, x="prediction", y="records", color="#0f766e")

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="hb-section-title">Model Inputs</div>', unsafe_allow_html=True)
        feature_rows = (
            [{"Feature type": "Numeric", "Feature": feature} for feature in NUMERIC_FEATURES]
            + [{"Feature type": "Categorical", "Feature": feature} for feature in CATEGORICAL_FEATURES]
        )
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)

    with right:
        st.markdown('<div class="hb-section-title">Artifact Status</div>', unsafe_allow_html=True)
        artifact_status = pd.DataFrame(
            [
                {
                    "Artifact": "Model pipeline",
                    "Path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
                    "Status": "Available" if MODEL_PATH.exists() else "Missing",
                },
                {
                    "Artifact": "Scored billing data",
                    "Path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
                    "Status": "Available" if DATA_PATH.exists() else "Missing",
                },
                {
                    "Artifact": "RAG reports",
                    "Path": str(REPORTS_PATH.relative_to(PROJECT_ROOT)),
                    "Status": "Available" if REPORTS_PATH.exists() else "Missing",
                },
                {
                    "Artifact": "Project config",
                    "Path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
                    "Status": "Available" if CONFIG_PATH.exists() else "Missing",
                },
            ]
        )
        st.dataframe(artifact_status, use_container_width=True, hide_index=True)


def ensure_chat_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "I can help investigate billing anomalies. Try asking which Oncology patients have unusually high charges.",
            }
        ]


def ask_rag(question: str) -> str:
    result = InvestigationGraph().run(question)
    answer = result["answer"]
    recommendations = result.get("recommendations", [])
    if recommendations:
        answer = f"{answer}\n\n**Recommended next steps**\n" + "\n".join(
            f"- {recommendation}" for recommendation in recommendations
        )
    return answer


def render_floating_rag_chat() -> None:
    ensure_chat_history()
    with st.container(key="floating_rag_chat"):
        if "rag_chat_minimized" not in st.session_state:
            st.session_state.rag_chat_minimized = False

        title_col, button_col = st.columns([0.82, 0.18], vertical_alignment="center")
        with title_col:
            st.markdown(
                """
                <div class="hb-rag-title">
                    <strong>RAG Assistant</strong>
                    <span>Live</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with button_col:
            button_label = "＋" if st.session_state.rag_chat_minimized else "−"
            button_help = "Open RAG Assistant" if st.session_state.rag_chat_minimized else "Minimize RAG Assistant"
            if st.button(button_label, key="rag_chat_toggle", help=button_help):
                st.session_state.rag_chat_minimized = not st.session_state.rag_chat_minimized
                st.rerun()

        if st.session_state.rag_chat_minimized:
            st.caption("RAG chat is minimized.")
            return

        st.markdown(
            """
            <div class="hb-rag-hint">
                Ask about flagged patients, high charges, departments, diagnoses, or payer patterns.
            </div>
            """,
            unsafe_allow_html=True,
        )

        for message in st.session_state.messages[-4:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.form("rag_chat_form", clear_on_submit=True):
            question = st.text_area(
                "Question",
                placeholder="Ask a billing investigation question...",
                label_visibility="collapsed",
                height=76,
            )
            submitted = st.form_submit_button("Ask RAG", use_container_width=True)

        if submitted and question.strip():
            question = question.strip()
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Retrieving anomaly context..."):
                answer = ask_rag(question)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()


def main() -> None:
    inject_bootstrap_theme()
    df = load_billing_data()
    reports = load_reports()

    if df.empty:
        st.error("No scored billing data found. Run `python run_project.py pipeline` first.")
        return

    filtered = build_filters(df)
    render_header(filtered if not filtered.empty else df)

    if filtered.empty:
        st.markdown(
            '<div class="hb-alert">No records match the selected filters.</div>',
            unsafe_allow_html=True,
        )
        return

    overview_tab, risk_tab, ml_tab = st.tabs(["Overview", "Risk Review", "ML Metrics"])

    with overview_tab:
        render_kpis(filtered)
        render_charts(filtered)

    with risk_tab:
        render_tables(filtered, reports)

    with ml_tab:
        render_ml_metrics(filtered)

    render_floating_rag_chat()


if __name__ == "__main__":
    main()
