"""Phase 3 decision engine: free-text question -> structured context -> LLM ->
validated `Recommendation` (README §8-§9).

Usage:
    python -m pipeline.decision_engine "Should I start Derrick Henry or Bijan Robinson in week 18?"
    python -m pipeline.decision_engine --players "Derrick Henry" "Bijan Robinson" --week 18

This module owns orchestration only: entity extraction and context assembly stay
deterministic (pipeline/), the LLM call stays in llm/interface.py, and the LLM
never fetches data itself.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from llm.interface import Recommendation, run_llm
from pipeline.context_builder import build_context
from pipeline.entity_extraction import extract_players, extract_week

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# The Recommendation schema is a single start/bench pair, so a decision compares
# exactly two players. Flex-from-N is a later extension, not v1 (README §0).
REQUIRED_PLAYERS = 2


class DecisionError(ValueError):
    """Raised when a question can't be turned into a well-formed two-player decision."""


@dataclass
class Decision:
    """A recommendation plus everything that produced it, for debugging and display."""

    question: str
    players: list[str]
    week: int
    context: str
    recommendation: Recommendation


def log(msg: str) -> None:
    """Print a progress message prefixed with the module name."""
    print(f"[decision_engine] {msg}")


def latest_week() -> int:
    """The week the processed data currently holds - the default when none is asked for.

    `refresh_stats.py` writes one week at a time, so the max here is the week the
    processed tables describe; asking for anything else would find no rows.
    """
    weeks = pl.read_parquet(PROCESSED_DIR / "player_stats.parquet", columns=["week"])
    if weeks.height == 0:
        raise DecisionError(
            "data/processed/player_stats.parquet is empty - run python scripts/refresh_stats.py"
        )
    return int(weeks.get_column("week").max())


def resolve_players(question: str, players: list[str] | None) -> list[str]:
    """Use explicitly passed players, else pull them out of the question text."""
    resolved = players if players is not None else extract_players(question)
    if len(resolved) != REQUIRED_PLAYERS:
        raise DecisionError(
            f"expected {REQUIRED_PLAYERS} known players, found {len(resolved)}: {resolved}. "
            "Name both players as they appear in the processed stats."
        )
    return resolved


def resolve_week(question: str, week: int | None) -> int:
    """Explicit week, else the week named in the question, else the data's week.

    `or` would be wrong here: week 0 is falsy, so a question asking about "week 0"
    would silently be answered for a different week instead of failing loudly in
    build_context.
    """
    if week is not None:
        return week
    from_question = extract_week(question)
    return from_question if from_question is not None else latest_week()


def _check_recommendation(recommendation: Recommendation, players: list[str]) -> None:
    """Deterministic guard on the LLM's answer.

    Structured outputs guarantee the response's *shape*, not that it named the two
    players actually asked about - a hallucinated or repeated name would otherwise
    render as a confident recommendation.
    """
    answered = {recommendation.start.strip().casefold(), recommendation.bench.strip().casefold()}
    if answered != {name.strip().casefold() for name in players}:
        raise DecisionError(
            f"LLM answered about {sorted(answered)}, not the requested {sorted(players)}"
        )


def decide(
    question: str,
    players: list[str] | None = None,
    week: int | None = None,
    tables: dict[str, pl.DataFrame] | None = None,
) -> Decision:
    """Answer a two-player start/sit question end to end.

    `players`/`week` override what the question text says; `tables` lets tests
    inject fixture DataFrames instead of reading data/processed/.
    """
    resolved_players = resolve_players(question, players)
    resolved_week = resolve_week(question, week)
    context = build_context(resolved_players, resolved_week, tables=tables)
    recommendation = run_llm(context, question)
    _check_recommendation(recommendation, resolved_players)
    return Decision(
        question=question,
        players=resolved_players,
        week=resolved_week,
        context=context,
        recommendation=recommendation,
    )


def format_decision(decision: Decision) -> str:
    """Human-readable terminal rendering of a Decision."""
    rec = decision.recommendation
    lines = [
        f"Week {decision.week}: {' vs '.join(decision.players)}",
        "",
        f"START: {rec.start}",
        f"BENCH: {rec.bench}",
        f"Confidence: {rec.confidence:.0%}",
        "",
        "Key factors:",
        *[f"  - {factor}" for factor in rec.key_factors],
        "",
        "Risk factors:",
        *[f"  - {factor}" for factor in rec.risk_factors],
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint: question (or explicit players) in, recommendation out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=None, help="free-text start/sit question")
    parser.add_argument(
        "--players", nargs=REQUIRED_PLAYERS, default=None,
        help="the two player names, bypassing entity extraction",
    )
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--show-context", action="store_true", help="print the LLM context too")
    args = parser.parse_args()

    if not args.question and not args.players:
        parser.error("pass a question, or --players with two names")

    load_dotenv()
    question = args.question or (
        f"Who should I start, {args.players[0]} or {args.players[1]}?"
    )
    decision = decide(question, players=args.players, week=args.week)

    if args.show_context:
        log("context sent to the LLM:")
        print(decision.context)
        print()
    print(format_decision(decision))


if __name__ == "__main__":
    main()
