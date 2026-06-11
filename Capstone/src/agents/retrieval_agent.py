from __future__ import annotations

from src.rag.retriever import AnomalyRetriever


class RetrievalAgent:
    def __init__(self, retriever: AnomalyRetriever | None = None):
        self.retriever = retriever or AnomalyRetriever()

    def run(self, question: str) -> list[dict]:
        return self.retriever.retrieve(question)
