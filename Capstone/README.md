# Hospital Billing AI Investigation Assistant

This project turns an exploratory hospital billing anomaly detection notebook into a reusable investigation assistant. It loads billing data, preprocesses records, trains an Isolation Forest anomaly model, generates anomaly explanations, builds lightweight RAG reports, and exposes the workflow through FastAPI and Streamlit.

The original notebook remains available as a research artifact. The runnable project code lives under `src/`.

## Project Features

- CSV ingestion with schema validation
- Data cleaning and feature engineering
- Isolation Forest anomaly detection
- Anomaly scoring and explanation generation
- RAG-style retrieval over anomaly reports
- Retrieval, investigation, and recommendation agents
- FastAPI `/chat` endpoint
- Streamlit chat UI
- Reviewer-label workflow for precision, recall, F1, and false-positive-rate evaluation

## Project Structure

```text
.
├── data/
│   ├── raw/Hospital_billing.csv
│   └── processed/
├── notebooks/
│   └── anomaly_detection.ipynb
├── src/
│   ├── agents/
│   ├── anomaly/
│   ├── api/
│   ├── evaluation/
│   ├── features/
│   ├── ingestion/
│   ├── models/
│   ├── preprocessing/
│   ├── rag/
│   └── ui/
├── tests/
├── config.yaml
├── project.md
├── run_project.py
├── requirements.txt
└── README.md
```

## Setup

From the project root:

```bash
cd /path/to/Capstone
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## One-Command Start

Use `run_project.py` to run the pipeline and start the project without manually entering each command.

Start the FastAPI app:

```bash
python run_project.py api
```

This runs:

```text
model training -> prediction/scoring -> RAG report generation -> FastAPI server
```

Then open:

```text
http://127.0.0.1:8001/docs
```

Start the Streamlit chat UI:

```bash
python run_project.py ui
```

Only run the data/model/RAG pipeline:

```bash
python run_project.py pipeline
```

Restart the API without rerunning the pipeline:

```bash
python run_project.py api --skip-pipeline
```

Use a different API port:

```bash
python run_project.py api --port 8002
```

Create a reviewer-label sample:

```bash
python run_project.py review-sample
```

Evaluate completed reviewer labels:

```bash
python run_project.py evaluate
```

## Run the Full Pipeline

Train the anomaly detection model and save cleaned data:

```bash
python -m src.models.train_isolation_forest
```

This creates or updates:

```text
src/models/model.pkl
data/processed/billing_cleaned.csv
```

Score the billing records:

```bash
python -m src.models.predict
```

This creates or updates:

```text
data/processed/billing_scored.csv
```

Generate anomaly investigation reports for RAG:

```bash
python -m src.rag.create_embeddings
```

This creates or updates:

```text
data/processed/anomaly_reports.jsonl
```

## Run the FastAPI Server

Start the API:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8001
```

Open the API docs:

```text
http://127.0.0.1:8001/docs
```

Useful endpoints:

```text
GET  /
GET  /health
POST /chat
```

Example `/chat` request:

```json
{
  "question": "Why was patient 151073 flagged?"
}
```

Example curl command:

```bash
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Why was patient 151073 flagged?"}'
```

Note: opening `/chat` directly in a browser will not work because it is a `POST` endpoint. Use `/docs` to test it interactively.

## Run the Streamlit Chat UI

Start the UI:

```bash
streamlit run src/ui/streamlit_app.py
```

Streamlit usually opens a browser automatically. If not, use the local URL printed in the terminal, commonly:

```text
http://localhost:8501
```

Try questions such as:

```text
Why was patient 151073 flagged?
Show me high charge Oncology anomalies.
Which denied claims look suspicious?
Find cases with no payment recorded.
```

## RAG Workflow

The RAG layer searches anomaly reports in:

```text
data/processed/anomaly_reports.jsonl
```

By default, retrieval uses a local TF-IDF vector store, so it works without an external LLM or embedding API. Optional ChromaDB support is available in the code, but the default local retrieval is enough for the demo.

To refresh RAG after data, model, or scoring changes:

```bash
python -m src.models.predict
python -m src.rag.create_embeddings
```

For patient-specific questions, include the patient ID in the question:

```text
Why was patient 151073 flagged?
```

The retriever checks exact patient IDs first, then falls back to similarity search.

## Evaluation Workflow

Because the dataset does not include confirmed ground-truth anomaly labels, create a reviewer sample:

```bash
python -m src.evaluation.validation --mode sample
```

This creates:

```text
data/processed/anomaly_review_labels.csv
```

Reviewers should update the `reviewer_label` column using:

```text
true_anomaly
not_anomaly
needs_review
```

After labels are reviewed, calculate metrics:

```bash
python -m src.evaluation.validation --mode evaluate
```

The evaluation reports:

```text
reviewed_records
true_positives
false_positives
true_negatives
false_negatives
precision
recall
f1
false_positive_rate
```

## Run Tests

```bash
pytest
```

If `pytest` crashes in a local Anaconda Python 3.13 environment, use a clean virtual environment with a stable Python version such as Python 3.11 or 3.12, then reinstall dependencies.

## Configuration

Main configuration is in:

```text
config.yaml
```

Current configurable paths and model settings include:

```yaml
data:
  raw_path: data/raw/Hospital_billing.csv
  processed_path: data/processed/billing_cleaned.csv

model:
  artifact_path: src/models/model.pkl
  contamination: 0.05
  random_state: 42

rag:
  persist_directory: data/vector_store
  reports_path: data/processed/anomaly_reports.jsonl
  top_k: 5

evaluation:
  labels_path: data/processed/anomaly_review_labels.csv
```

## Common Troubleshooting

If `http://127.0.0.1:8001/chat` says method not allowed, use `/docs` and send a `POST` request.

If port `8001` is already in use, start the API on another port:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8002
```

If `/chat` returns no matching anomaly reports, regenerate the scored data and reports:

```bash
python -m src.models.predict
python -m src.rag.create_embeddings
```

If patient-specific questions return broad results, make sure the question includes the numeric patient ID exactly as it appears in the data.

## Recommended First Run

For a fresh run, execute:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_project.py api
```

Then open:

```text
http://127.0.0.1:8001/docs
```
