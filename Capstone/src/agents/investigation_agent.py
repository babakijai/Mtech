from __future__ import annotations


class InvestigationAgent:
    def run(self, question: str, documents: list[dict]) -> str:
        if not documents:
            return "No matching anomaly reports were found. Train and score the model first, then generate reports."

        summaries = []
        for document in documents[:3]:
            metadata = document.get("metadata", {})
            summaries.append(metadata.get("summary") or document["text"].replace("\n", " "))
        return "\n".join(summaries)
