from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.graph import InvestigationGraph


app = FastAPI(title="Hospital Billing AI Investigation Assistant")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    recommendations: list[str]


@app.get("/")
def root() -> dict:
    return {
        "name": "Hospital Billing AI Investigation Assistant",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = InvestigationGraph().run(request.question)
    return ChatResponse(answer=result["answer"], recommendations=result["recommendations"])
