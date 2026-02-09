from __future__ import annotations
from dataclasses import dataclass
import json
import os
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv


@dataclass
class LLMResponse:
    start: str
    bench: str
    confidence: float
    key_factors: list[str]
    risk_factors: list[str]


def run_llm(context: str, question: str | None = None) -> LLMResponse:
    return run_openai(context, question=question)


def run_openai(context: str, question: str | None = None) -> LLMResponse:

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set. Example: OPENAI_MODEL=gpt-4o-mini")
    
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI key not set!")

    client = OpenAI(api_key=key)

    instructions = (
        "You are a fantasy football analyst. "
        "Use only the provided context. "
        "Return ONLY valid JSON with keys: "
        "start, bench, confidence, key_factors, risk_factors. "
        "confidence must be between 0 and 1."
    )
    user_input = f"Context:\n{context}\n\nQuestion: {question or 'Provide a start/sit recommendation.'}"

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
    )

    return _parse_json_response(response.output_text)


def _parse_json_response(text: str) -> LLMResponse:
    data: dict[str, Any]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Best-effort fallback to keep the pipeline running.
        return LLMResponse(
            start="Unknown",
            bench="Unknown",
            confidence=0.0,
            key_factors=["model output was not valid JSON"],
            risk_factors=["check LLM prompt/response formatting"],
        )

    return LLMResponse(
        start=str(data.get("start", "Unknown")),
        bench=str(data.get("bench", "Unknown")),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        key_factors=data.get("key_factors", []),
        risk_factors=data.get("risk_factors", []),
    )


if __name__ == '__main__':
    load_dotenv()
    resp = run_llm("Player A: 18 pts. Player B: 12 pts.", "Who should I start?")
    print(resp)