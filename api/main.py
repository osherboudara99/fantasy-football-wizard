"""FastAPI layer over the chat pipeline (README §14, chat design spec).

Run locally:
    uvicorn api.main:app --reload

The API owns no logic of its own: it validates input, calls `chat()`, and maps
pipeline errors onto status codes. Everything else - data access, context
assembly, the Anthropic key - stays server-side in the pipeline modules.
"""
from __future__ import annotations

import os

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from pipeline.chat_engine import NoPlayersFoundError, chat
from pipeline.context_builder import PlayerNotFoundError
from pipeline.decision_engine import DataUnavailableError, DecisionError
from pipeline.entity_extraction import known_player_names
from pipeline.gcs_sync import sync_from_gcs

# The frontend is a separate origin in dev (Vite on 5173) and in prod (Cloudflare
# Pages), so the allowed origins have to be configurable per deployment.
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

# Every request that reaches chat() costs a paid LLM call whose price scales with
# the input. Cap the message, each history turn, each player name, and the number
# of mentions/history turns so a caller can't turn one request into a six-figure-
# token bill.
MAX_QUESTION_LENGTH = 500
MAX_PLAYER_NAME_LENGTH = 100
MAX_MENTIONED_PLAYERS = 10
MAX_HISTORY_TURNS = 20

DATA_UNAVAILABLE_DETAIL = "Player data is unavailable; try again later."

# CORS is not a defense against abuse - it only constrains browsers, and curl
# ignores it entirely. This caps how many paid LLM calls one caller can trigger
# per hour (README §14). In-process/per-instance counters, not shared across
# replicas - the Cloud Run service MUST be deployed with --max-instances=1 or
# a second instance resets this caller's quota to another 30/hour. See the
# "Deploy-time requirement" note under README §14's rate limiting section.
CHAT_RATE_LIMIT = "30/hour"

load_dotenv()


def _client_ip(request: Request) -> str:
    """Key the rate limiter by the real caller, not Cloud Run's ingress proxy.

    Cloud Run terminates the connection and forwards to the container from an
    internal proxy address, so every request's raw socket peer
    (get_remote_address's source) would be identical - collapsing every real
    caller into one shared 30/hour bucket instead of 30/hour each.

    Cloud Run's GFE is a single trusted proxy hop directly in front of the
    container: like any reverse proxy, it appends the address it observed to
    the *end* of any X-Forwarded-For header already on the request, so the
    rightmost entry is the one GFE itself attests to. Everything to its left -
    including the entire header, if a caller sends one - is attacker-supplied
    input; trusting the leftmost entry would let a caller dodge the limit by
    sending a fresh fake value on every request. Local dev (nothing in front
    of uvicorn) never sets this header, so this falls back to the socket
    address there.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[-1].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)


def _docs_config() -> dict[str, str | None]:
    """Disable /docs, /redoc, /openapi.json in prod - they advertise the paid
    endpoint's schema. Cloud Run sets GCS_BUCKET; local dev leaves it unset.
    """
    if os.getenv("GCS_BUCKET"):
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {}


def _maybe_sync_from_gcs() -> None:
    """Download processed data + the Chroma index before serving traffic.

    Cloud Run sets GCS_BUCKET; local dev leaves it unset, so this is a no-op
    and local dev keeps reading data/ and embeddings/ straight off disk.
    """
    bucket = os.getenv("GCS_BUCKET")
    if bucket:
        sync_from_gcs(bucket)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _maybe_sync_from_gcs()
    yield


app = FastAPI(
    title="Fantasy Football Wizard", version="0.1.0", lifespan=lifespan, **_docs_config()
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
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


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(max_length=MAX_QUESTION_LENGTH)]


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(max_length=MAX_QUESTION_LENGTH)]
    mentioned_players: list[Annotated[str, StringConstraints(max_length=MAX_PLAYER_NAME_LENGTH)]] = (
        Field(default_factory=list, max_length=MAX_MENTIONED_PLAYERS)
    )
    week: int | None = Field(
        default=None, description="Defaults to the week the processed data describes"
    )
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class SourceItem(BaseModel):
    title: str
    link: str
    source: str
    published_at: str


class RecommendationPayload(BaseModel):
    start: str
    bench: str
    confidence: float
    key_factors: list[str]
    risk_factors: list[str]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    players_discussed: list[str]
    recommendation: RecommendationPayload | None
    week: int
    # The assembled context travels with the answer so the UI can show what the
    # answer was actually based on (README §10's debug view).
    context: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe - also what Cloud Run hits before serving traffic."""
    return {"status": "ok"}


@app.get("/players")
def players() -> list[str]:
    """Every player the processed data knows about, for the frontend's @-mention search."""
    try:
        return known_player_names()
    except FileNotFoundError as exc:
        # Missing local file (GCS sync failed, or hasn't run yet) - same outage
        # signal as DataUnavailableError, just from a different failure mode.
        raise HTTPException(status_code=503, detail=DATA_UNAVAILABLE_DETAIL) from exc


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(CHAT_RATE_LIMIT)
def chat_endpoint(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """Answer any fantasy-football question about the mentioned/named players.

    Takes the raw `Request` too (unused directly) because slowapi's `@limiter.limit`
    reads the caller's address off it - the parameter name `request` is what
    `get_remote_address` expects to find.
    """
    try:
        result = chat(
            chat_request.message,
            mentioned_players=chat_request.mentioned_players,
            week=chat_request.week,
            history=[turn.model_dump() for turn in chat_request.history],
        )
    except NoPlayersFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PlayerNotFoundError as exc:
        # Unknown player or a week the data doesn't cover: the caller's input is
        # wrong, not the server - 404 rather than a 500 stack trace.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        # The refresh job hasn't populated data/ - nothing the caller can fix, and
        # the underlying message names server paths, so don't echo it back.
        raise HTTPException(status_code=503, detail=DATA_UNAVAILABLE_DETAIL) from exc
    except FileNotFoundError as exc:
        # Missing local file (GCS sync failed, or hasn't run yet) - same outage
        # signal as DataUnavailableError, just from a different failure mode.
        raise HTTPException(status_code=503, detail=DATA_UNAVAILABLE_DETAIL) from exc
    except DecisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceItem(title=s.title, link=s.link, source=s.source, published_at=s.published_at)
            for s in result.sources
        ],
        players_discussed=result.players_discussed,
        recommendation=(
            RecommendationPayload(**result.recommendation.model_dump())
            if result.recommendation is not None
            else None
        ),
        week=result.week,
        context=result.context,
    )
