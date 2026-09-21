from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pipeline.scoring import ScoringRules

DEFAULT_MODEL = "claude-haiku-4-5"

RECENCY_WEIGHTING_GUIDANCE = (
    "The context separates a player's this-season stats from their last-season stats — "
    "never average or blend the two yourself. This season's games, injury reports, and "
    "news (depth-chart moves, a new starting QB, a scheme or coaching change, a trade) "
    "are the primary signal; treat them as far more predictive of what happens next than "
    "how the player performed last season. Last season's numbers are lower-confidence "
    "background, useful mainly when this season's sample is small (0-2 games) or absent — "
    "lean on them only to the extent this season's data doesn't yet answer the question, "
    "and say so explicitly (e.g. 'with only one game of current-season data, last season's "
    "usage is still informative here') rather than citing both seasons as if equally strong "
    "evidence. If news describes a change that would make last season's stats stop applying "
    "(new team, new role, new starter under center), say so and discount them accordingly. "
    "If the context includes a note that historical projections/news aren't available for a "
    "past week, state that limitation plainly in your answer rather than ignoring it or "
    "answering as if you had the full picture for that week. If the context instead notes "
    "that a future week hasn't been prepared yet (no projection exists for it), you may "
    "still reason from current-form trends, but say plainly that you don't have a specific "
    "projection for that week rather than presenting the trend as if it were one."
)

SYSTEM_PROMPT = (
    "You are a fantasy football analyst. "
    "Use only the provided context — do not rely on outside knowledge of players. "
    "Explain your reasoning clearly and explicitly assess risk. "
    + RECENCY_WEIGHTING_GUIDANCE
)


class Recommendation(BaseModel):
    start: str = Field(description="The player to start")
    bench: str = Field(description="The player to bench")
    confidence: float = Field(ge=0, le=1, description="Confidence between 0 and 1")
    key_factors: list[str] = Field(description="Main reasons for the recommendation")
    risk_factors: list[str] = Field(description="Risks that could invalidate it")


class ChatAnswer(BaseModel):
    answer: str = Field(description="Prose answer to the user's fantasy football question")
    recommendation: Recommendation | None = Field(
        default=None,
        description=(
            "A start/bench recommendation, populated only when the question is a "
            "start/sit or flex-style comparison; left unset for any other question."
        ),
    )


CHAT_SYSTEM_PROMPT = (
    "You are a fantasy football analyst answering questions in a chat interface. "
    "Use only the provided context — do not rely on outside knowledge of players. "
    "Explain your reasoning clearly. If, and only if, the question is a start/sit "
    "or flex-style comparison between players, also fill in `recommendation` with "
    "a start/bench pick, a confidence between 0 and 1, and key/risk factors; for "
    "any other question (a single player's outlook, a trade evaluation, or a "
    "general question), leave `recommendation` unset and answer only in `answer`. "
    + RECENCY_WEIGHTING_GUIDANCE
)


def run_chat_llm(
    context: str,
    message: str,
    history: list[dict[str, str]] | None = None,
) -> ChatAnswer:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = anthropic.Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    messages = [dict(turn) for turn in (history or [])]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"})

    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=CHAT_SYSTEM_PROMPT,
        messages=messages,
        output_format=ChatAnswer,
    )
    return response.parsed_output


def run_llm(context: str, question: str | None = None) -> Recommendation:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = anthropic.Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    user_input = (
        f"Context:\n{context}\n\n"
        f"Question: {question or 'Provide a start/sit recommendation.'}"
    )

    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_input}],
        output_format=Recommendation,
    )
    return response.parsed_output


def parse_custom_scoring_rules(base: ScoringRules, description: str) -> ScoringRules:
    """Turn a free-text description of league-scoring modifications into a
    full ScoringRules, seeded from `base` (docs/superpowers/specs/2026-09-19-
    custom-league-scoring-design.md). Runs once, at settings-save time
    (POST /scoring-rules) - never per chat message.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = anthropic.Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    prompt = (
        "A fantasy football league's base scoring rules are (points per unit):\n"
        f"{base.model_dump_json(indent=2)}\n\n"
        "The user describes how their league's rules differ from this base:\n"
        f'"{description}"\n\n'
        "Return the full set of scoring rules with only the described "
        "categories changed - every field the description doesn't mention "
        "must keep its base value exactly."
    )
    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=(
            "You configure fantasy football scoring rules from a user's "
            "description. Only change what the user actually describes."
        ),
        messages=[{"role": "user", "content": prompt}],
        output_format=ScoringRules,
    )
    return response.parsed_output


if __name__ == "__main__":
    load_dotenv()
    rec = run_llm("Player A: 18 pts avg. Player B: 12 pts avg.", "Who should I start?")
    print(rec)
