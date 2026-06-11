from __future__ import annotations

import re

from src.rag.create_embeddings import load_documents
from src.rag.vector_store import build_vector_store


PATIENT_ID_PATTERN = re.compile(r"\b\d{5,}\b")


class AnomalyRetriever:
    def __init__(
        self,
        reports_path: str = "data/processed/anomaly_reports.jsonl",
        top_k: int = 5,
        prefer_chroma: bool = False,
    ):
        self.documents = load_documents(reports_path)
        self.top_k = top_k
        self.store = build_vector_store(self.documents, prefer_chroma=prefer_chroma)

    def _retrieve_by_patient_id(self, query: str) -> list[dict]:
        patient_ids = set(PATIENT_ID_PATTERN.findall(query))
        if not patient_ids:
            return []
        return [
            document
            for document in self.documents
            if str(document.get("metadata", {}).get("patient_id")) in patient_ids
        ]

    def retrieve(self, query: str) -> list[dict]:
        exact_patient_matches = self._retrieve_by_patient_id(query)
        if exact_patient_matches:
            return exact_patient_matches

        if hasattr(self.store, "search"):
            return self.store.search(query, self.top_k)
        result = self.store.query(query_texts=[query], n_results=self.top_k)
        return [
            {"id": doc_id, "text": text, "metadata": metadata}
            for doc_id, text, metadata in zip(
                result.get("ids", [[]])[0],
                result.get("documents", [[]])[0],
                result.get("metadatas", [[]])[0],
            )
        ]
