from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LocalVectorStore:
    """Small TF-IDF retrieval fallback for demos and tests."""

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([doc["text"] for doc in documents]) if documents else None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.documents or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1][:top_k]
        results = []
        for index in ranked_indices:
            document = dict(self.documents[index])
            document["score"] = float(scores[index])
            results.append(document)
        return results


def build_chroma_store(documents: list[dict], persist_directory: str | Path = "data/vector_store"):
    """Build a Chroma collection when chromadb is installed."""
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_or_create_collection("billing_anomaly_reports")
    if documents:
        collection.upsert(
            ids=[doc["id"] for doc in documents],
            documents=[doc["text"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents],
        )
    return collection


def build_vector_store(
    documents: list[dict],
    persist_directory: str | Path = "data/vector_store",
    prefer_chroma: bool = False,
):
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    if prefer_chroma:
        try:
            return build_chroma_store(documents, persist_directory)
        except Exception:
            pass
    return LocalVectorStore(documents)
