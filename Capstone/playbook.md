# Hospital Billing AI Investigation Assistant - Project Playbook

## Overview

This project transforms the existing exploratory anomaly detection notebook into a production-ready AI-powered investigation platform.

### Objectives

* Preserve the notebook as a research artifact.
* Refactor notebook logic into reusable Python modules.
* Implement anomaly detection using Isolation Forest.
* Build explainability around detected anomalies.
* Implement Retrieval Augmented Generation (RAG).
* Build AI agents for investigation workflows.
* Expose functionality through APIs.
* Create a user-friendly chat interface.

---

# Current State

The notebook currently contains:

* Data Loading
* Data Cleaning
* Missing Value Handling
* Feature Engineering
* Exploratory Data Analysis
* Isolation Forest Training
* Anomaly Detection
* Manual Investigation

The notebook should remain unchanged and act only as a reference implementation.

---

# Target Architecture

```text
Hospital Billing Dataset
        ↓
Data Processing Layer
        ↓
Feature Engineering
        ↓
Isolation Forest
        ↓
Anomaly Findings
        ↓
Explanation Engine
        ↓
Vector Database (ChromaDB)
        ↓
RAG Layer
        ↓
AI Investigation Agents
        ↓
FastAPI Services
        ↓
Chat UI
```

---

# Project Structure

```text
hospital-billing-ai/
│
├── notebooks/
│   └── anomaly_detection.ipynb
│
├── data/
│   ├── raw/
│   │   └── Hospital_billing.csv
│   │
│   └── processed/
│       └── billing_cleaned.csv
│
├── src/
│   │
│   ├── ingestion/
│   │   └── load_data.py
│   │
│   ├── preprocessing/
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train_isolation_forest.py
│   │   ├── predict.py
│   │   └── model.pkl
│   │
│   ├── anomaly/
│   │   └── anomaly_explainer.py
│   │
│   ├── rag/
│   │   ├── create_embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── agents/
│   │   ├── retrieval_agent.py
│   │   ├── investigation_agent.py
│   │   ├── recommendation_agent.py
│   │   └── graph.py
│   │
│   ├── api/
│   │   └── app.py
│   │
│   └── utils/
│       └── logger.py
│
├── tests/
│
├── requirements.txt
├── config.yaml
├── README.md
└── playbook.md
```

---

# Phase 1 - Refactor Notebook

## Module 1: Data Ingestion

File:

```text
src/ingestion/load_data.py
```

Responsibilities:

* Load CSV files
* Validate schema
* Return DataFrame

---

## Module 2: Data Preprocessing

File:

```text
src/preprocessing/preprocess.py
```

Responsibilities:

* Missing value handling
* Date conversions
* Data normalization
* Data validation

---

## Module 3: Feature Engineering

File:

```text
src/features/feature_engineering.py
```

Responsibilities:

* One-hot encoding
* Numerical feature preparation
* Model feature generation

---

# Phase 2 - Anomaly Detection Layer

## Model Training

File:

```text
src/models/train_isolation_forest.py
```

Responsibilities:

* Train Isolation Forest
* Save model
* Version model artifacts

Outputs:

```text
model.pkl
```

---

## Prediction Module

File:

```text
src/models/predict.py
```

Responsibilities:

* Load trained model
* Generate anomaly scores
* Label anomalies

Outputs:

```text
anomaly_score
anomaly_flag
```

---

# Phase 3 - Explainability Layer

## Anomaly Explainer

File:

```text
src/anomaly/anomaly_explainer.py
```

Responsibilities:

* Explain anomaly causes
* Compare against department averages
* Generate human-readable findings

Example:

```text
Patient P1001 was flagged.

Charge Amount: $15,000

Department Average: $5,000

Reason:
Charge exceeds department average by 200%.
```

---

# Phase 4 - RAG Implementation

## Generate Investigation Documents

File:

```text
src/rag/create_embeddings.py
```

Responsibilities:

* Convert anomaly records into investigation reports
* Create semantic documents for retrieval

Example Document:

```text
Patient P1001

Department: Cardiology

Charge Amount: 15000

Department Average: 5000

Reason:
Charge exceeds department average.
```

---

## Vector Store

File:

```text
src/rag/vector_store.py
```

Technology:

```text
ChromaDB
```

Responsibilities:

* Store embeddings
* Store metadata
* Enable semantic search

---

## Retriever

File:

```text
src/rag/retriever.py
```

Responsibilities:

* Accept user queries
* Retrieve relevant anomaly documents
* Return context for LLMs

---

# Phase 5 - Multi-Agent System

## Retrieval Agent

File:

```text
src/agents/retrieval_agent.py
```

Responsibilities:

* Query vector database
* Retrieve anomaly reports

---

## Investigation Agent

File:

```text
src/agents/investigation_agent.py
```

Responsibilities:

* Analyze retrieved findings
* Explain anomaly causes
* Generate investigation summary

---

## Recommendation Agent

File:

```text
src/agents/recommendation_agent.py
```

Responsibilities:

* Suggest corrective actions
* Recommend manual reviews
* Identify potential fraud indicators

Example:

```text
Recommended Actions:

1. Verify insurance claim
2. Review physician notes
3. Check duplicate billing
```

---

## LangGraph Workflow

File:

```text
src/agents/graph.py
```

Workflow:

```text
START
 ↓
Retrieval Agent
 ↓
Investigation Agent
 ↓
Recommendation Agent
 ↓
END
```

---

# Phase 6 - API Layer

## FastAPI Application

File:

```text
src/api/app.py
```

Endpoints:

### Chat Endpoint

```http
POST /chat
```

Request:

```json
{
  "question": "Why was patient P1001 flagged?"
}
```

Response:

```json
{
  "answer": "Patient P1001 was flagged because..."
}
```

---

# Phase 7 - Frontend

Options:

## Streamlit

Pros:

* Fastest implementation
* Ideal for demos
* Suitable for hackathons

## React

Pros:

* Production ready
* Better user experience
* Easier future expansion

---

# Development Roadmap

## Week 1

* Create project structure
* Extract notebook code
* Build preprocessing pipeline

Deliverables:

* load_data.py
* preprocess.py
* feature_engineering.py

---

## Week 2

* Train Isolation Forest
* Save model
* Generate anomaly scores

Deliverables:

* train_isolation_forest.py
* predict.py

---

## Week 3

* Build explanation engine
* Generate investigation reports

Deliverables:

* anomaly_explainer.py

---

## Week 4

* Implement ChromaDB
* Create embeddings
* Build retrieval pipeline

Deliverables:

* create_embeddings.py
* vector_store.py
* retriever.py

---

## Week 5

* Build LangGraph workflow
* Create agents

Deliverables:

* retrieval_agent.py
* investigation_agent.py
* recommendation_agent.py
* graph.py

---

## Week 6

* FastAPI integration
* Chat endpoint
* Streamlit UI

Deliverables:

* app.py
* Chat UI

---

# Recommended Technology Stack

## Machine Learning

* Python
* Pandas
* Scikit-Learn
* Isolation Forest

## RAG

* LangChain
* ChromaDB
* Sentence Transformers

## Agents

* LangGraph

## API

* FastAPI

## Frontend

* Streamlit (Phase 1)
* React (Phase 2)

## Deployment

* Docker
* Kubernetes (optional)
* AWS / Azure

---

# Final Outcome

The final system should support:

* Billing anomaly detection
* Explainable anomaly reports
* Semantic search on findings
* AI-powered investigation assistant
* Recommended actions for auditors
* Conversational chat interface
* Production-ready API architecture

This transforms the notebook from an exploratory analysis into a scalable AI Investigation Platform suitable for enterprise and hackathon demonstrations.
