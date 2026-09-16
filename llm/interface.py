from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are a fantasy football analyst. "
    "Use only the provided context — do not rely on outside knowledge of players. "
    "Explain your reasoning clearly and explicitly assess risk."
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
    "general question), leave `recommendation` unset and answer only in `answer`."
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


if __name__ == "__main__":
    load_dotenv()
    rec = run_llm("Player A: 18 pts avg. Player B: 12 pts avg.", "Who should I start?")
    print(rec)
