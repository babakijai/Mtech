from __future__ import annotations

from src.agents.investigation_agent import InvestigationAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.retrieval_agent import RetrievalAgent


class InvestigationGraph:
    """Sequential retrieval -> investigation -> recommendation workflow."""

    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.investigation_agent = InvestigationAgent()
        self.recommendation_agent = RecommendationAgent()

    def run(self, question: str) -> dict:
        documents = self.retrieval_agent.run(question)
        summary = self.investigation_agent.run(question, documents)
        recommendations = self.recommendation_agent.run(documents)
        return {
            "question": question,
            "answer": f"{summary}\n\nRecommended actions:\n" + "\n".join(f"- {item}" for item in recommendations),
            "documents": documents,
            "recommendations": recommendations,
        }
