"""FastAPI layer over the decision pipeline (README §14).

Run locally:
    uvicorn api.main:app --reload

The API owns no logic of its own: it validates input, calls `decide()`, and maps
pipeline errors onto status codes. Everything else - data access, context
assembly, the Anthropic key - stays server-side in the pipeline modules.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.context_builder import PlayerNotFoundError
from pipeline.decision_engine import REQUIRED_PLAYERS, DecisionError, decide
from pipeline.entity_extraction import known_player_names

# The frontend is a separate origin in dev (Vite on 5173) and in prod (Cloudflare
# Pages), so the allowed origins have to be configurable per deployment.
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

load_dotenv()

app = FastAPI(title="Fantasy Football Wizard", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", DEFAULT_ORIGINS).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RecommendationRequest(BaseModel):
    players: list[str] = Field(
        min_length=REQUIRED_PLAYERS,
        max_length=REQUIRED_PLAYERS,
        description="The two players to compare",
    )
    week: int | None = Field(
        default=None, description="Defaults to the week the processed data describes"
    )
    question: str | None = Field(default=None, description="Optional free-text question")


class RecommendationResponse(BaseModel):
    start: str
    bench: str
    confidence: float
    key_factors: list[str]
    risk_factors: list[str]
    week: int
    players: list[str]
    # The assembled context travels with the answer so the UI can show what the
    # recommendation was actually based on (README §10's debug view).
    context: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe - also what Cloud Run hits before serving traffic."""
    return {"status": "ok"}


@app.get("/players")
def players() -> list[str]:
    """Every player the processed data knows about, for the frontend's picker."""
    return known_player_names()


@app.post("/recommendation", response_model=RecommendationResponse)
def recommendation(request: RecommendationRequest) -> RecommendationResponse:
    """Run a two-player start/sit decision end to end."""
    question = request.question or (
        f"Who should I start, {request.players[0]} or {request.players[1]}?"
    )
    try:
        decision = decide(question, players=request.players, week=request.week)
    except PlayerNotFoundError as exc:
        # Unknown player or a week the data doesn't cover: the caller's input is
        # wrong, not the server - 404 rather than a 500 stack trace.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DecisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rec = decision.recommendation
    return RecommendationResponse(
        start=rec.start,
        bench=rec.bench,
        confidence=rec.confidence,
        key_factors=rec.key_factors,
        risk_factors=rec.risk_factors,
        week=decision.week,
        players=decision.players,
        context=decision.context,
    )
