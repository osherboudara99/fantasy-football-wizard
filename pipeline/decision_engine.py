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
from typing import Callable

import polars as pl
from dotenv import load_dotenv

from llm.interface import Recommendation, run_llm
from pipeline.context_builder import build_context
from pipeline.entity_extraction import extract_players, extract_week
from pipeline.scoring import ScoringRules
from retrieval.news_retriever import retrieve_news

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# The Recommendation schema is a single start/bench pair, so a decision compares
# exactly two players. Flex-from-N is a later extension, not v1 (README §0).
REQUIRED_PLAYERS = 2


class DecisionError(ValueError):
    """Raised when a question can't be turned into a well-formed two-player decision."""


class DataUnavailableError(DecisionError):
    """Raised when the processed data needed to answer anything is missing or empty.

    Distinct from a bad question: nothing the caller sends can fix it, so callers
    (the API) can report it as a server-side outage rather than user error.
    """


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


def target_season_week() -> tuple[int, int]:
    """The (season, week) the current processed data targets - the decision
    in progress. Read from data/processed/meta.parquet, written by
    refresh_stats.py's build_processed().

    player_stats.parquet no longer carries a single target season/week per
    row (it holds many historical per-game rows across two real seasons), so
    this dedicated meta table is the only place it survives.
    """
    meta_path = PROCESSED_DIR / "meta.parquet"
    if not meta_path.exists():
        raise DataUnavailableError(
            "data/processed/meta.parquet is missing - run python scripts/refresh_stats.py"
        )
    df = pl.read_parquet(meta_path)
    if df.height == 0:
        raise DataUnavailableError(
            "data/processed/meta.parquet is empty - run python scripts/refresh_stats.py"
        )
    row = df.row(0, named=True)
    return row["season"], row["week"]


def target_week() -> int:
    """The week half of target_season_week() - most callers only override week."""
    return target_season_week()[1]


def is_historical_query(season: int, week: int) -> bool | None:
    """Whether (season, week) is strictly before the real target the
    processed data describes - the authoritative signal build_context needs
    to know a query is historical even for a player whose own most recent
    game IS the asked-about week (no later row for them exists yet, so the
    per-player fallback heuristic in pipeline/player_form.py would otherwise
    miss this common case). `None` when the target itself is unavailable
    (e.g. a fresh checkout with no data refreshed yet) - build_context then
    falls back to its own per-player heuristic rather than crashing a
    request that doesn't strictly need this comparison.
    """
    try:
        return (season, week) < target_season_week()
    except DataUnavailableError:
        return None


def is_future_query(season: int, week: int) -> bool | None:
    """Whether (season, week) is strictly after the real target the
    processed data describes - the app has no prepared projection that far
    out (only the target week's Sleeper snapshot is ever fetched), so
    build_context needs this to disclose that a distant-future question is
    answered from current-form trends only, not a week-specific projection.
    `None` when the target itself is unavailable, matching is_historical_query.
    """
    try:
        return (season, week) > target_season_week()
    except DataUnavailableError:
        return None


def resolve_season(season: int | None) -> int:
    """Explicit season, else whatever the current processed data targets.

    Unlike week, there's no free-text season extraction - nobody types
    "season 2026" in a start/sit question - so the only override is passing
    `season` directly.
    """
    if season is not None:
        return season
    return target_season_week()[0]


def resolve_players(question: str, players: list[str] | None) -> list[str]:
    """Use explicitly passed players, else pull them out of the question text."""
    resolved = players if players is not None else extract_players(question)
    if len(resolved) != REQUIRED_PLAYERS:
        raise DecisionError(
            f"expected {REQUIRED_PLAYERS} known players, found {len(resolved)}: {resolved}. "
            "Name both players as they appear in the processed stats."
        )
    # Comparing a player to himself is never a real question, and it would slip
    # past the set-based check in _check_recommendation (a one-element set matches
    # a one-element set, so start == bench would look valid).
    if len({name.strip().casefold() for name in resolved}) != REQUIRED_PLAYERS:
        raise DecisionError(f"expected two different players, got {resolved}")
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
    return from_question if from_question is not None else target_week()


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
    season: int | None = None,
    week: int | None = None,
    tables: dict[str, pl.DataFrame] | None = None,
    news_fn: Callable[[str | None, str], list[str]] | None = None,
    scoring_rules: ScoringRules | None = None,
) -> Decision:
    """Answer a two-player start/sit question end to end.

    `players`/`season`/`week` override what the question text says (season
    has no text-extraction path - see resolve_season); `tables` lets tests
    inject fixture DataFrames instead of reading data/processed/.
    `scoring_rules` defaults to full PPR when omitted (build_context's own
    default) - see docs/superpowers/specs/2026-09-19-custom-league-scoring-design.md.
    """
    resolved_players = resolve_players(question, players)
    resolved_week = resolve_week(question, week)
    resolved_season = resolve_season(season)
    news_fn = news_fn if news_fn is not None else retrieve_news
    context = build_context(
        resolved_players, resolved_season, resolved_week,
        tables=tables, news_fn=news_fn, scoring_rules=scoring_rules,
        is_historical=is_historical_query(resolved_season, resolved_week),
        is_future=is_future_query(resolved_season, resolved_week),
    )
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
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--show-context", action="store_true", help="print the LLM context too")
    args = parser.parse_args()

    if not args.question and not args.players:
        parser.error("pass a question, or --players with two names")

    load_dotenv()
    question = args.question or (
        f"Who should I start, {args.players[0]} or {args.players[1]}?"
    )
    decision = decide(question, players=args.players, season=args.season, week=args.week)

    if args.show_context:
        log("context sent to the LLM:")
        print(decision.context)
        print()
    print(format_decision(decision))


if __name__ == "__main__":
    main()
