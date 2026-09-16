"""Chat engine: generalizes decision_engine.decide() beyond a fixed two-player
start/sit question (docs/superpowers/specs/2026-09-09-chat-assistant-design.md).

Any number of @-mentioned or free-text-named players; the response only
carries a start/bench recommendation when the LLM judges the question to be
shaped that way, and always carries the news sources it drew on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl

from llm.interface import ChatAnswer, Recommendation, run_chat_llm
from pipeline.context_builder import build_context
from pipeline.decision_engine import DecisionError, resolve_week
from pipeline.entity_extraction import extract_players
from retrieval.news_retriever import NewsItem, retrieve_news


class NoPlayersFoundError(ValueError):
    """Raised when neither @-mentions nor free text name any known player."""


@dataclass
class ChatResult:
    """A chat answer plus everything that produced it, for the API/UI to render."""

    answer: str
    sources: list[NewsItem]
    players_discussed: list[str]
    recommendation: Recommendation | None
    week: int
    context: str


def resolve_chat_players(message: str, mentioned_players: list[str]) -> list[str]:
    """@-mentions first (reliable, no fuzzy matching), then any additional names
    found in the free text, de-duplicated case-insensitively with mentions first.
    """
    seen: set[str] = set()
    resolved: list[str] = []
    for name in [*mentioned_players, *extract_players(message)]:
        key = name.strip().casefold()
        if key not in seen:
            seen.add(key)
            resolved.append(name)
    if not resolved:
        raise NoPlayersFoundError(
            "No known players mentioned - @-mention a player or name them in your message."
        )
    return resolved


def _check_recommendation(recommendation: Recommendation | None, players: list[str]) -> None:
    """A recommendation may only name players actually discussed this turn.

    Subset, not exact-set equality (unlike decision_engine's two-player check):
    chat can discuss 3+ players and still recommend a start/bench pair from
    among them.
    """
    if recommendation is None:
        return
    start = recommendation.start.strip().casefold()
    bench = recommendation.bench.strip().casefold()
    if start == bench:
        raise DecisionError(f"LLM recommended the same player to start and bench: {recommendation.start}")
    known = {name.strip().casefold() for name in players}
    if not {start, bench} <= known:
        raise DecisionError(
            f"LLM recommended players not among those discussed: {sorted({start, bench} - known)}"
        )


def chat(
    message: str,
    mentioned_players: list[str] | None = None,
    week: int | None = None,
    history: list[dict[str, str]] | None = None,
    tables: dict[str, pl.DataFrame] | None = None,
    news_fn: Callable[[str | None, str], list[NewsItem]] | None = None,
) -> ChatResult:
    """Answer any fantasy-football question about the resolved players.

    `tables`/`news_fn` are injectable for tests, matching decision_engine.decide().
    `news_fn` defaults to the real Chroma-backed retriever; every NewsItem it
    returns is collected (de-duplicated by link) into the response's sources.
    """
    mentioned_players = mentioned_players or []
    history = history or []
    players = resolve_chat_players(message, mentioned_players)
    resolved_week = resolve_week(message, week)

    news_fn = news_fn if news_fn is not None else retrieve_news
    collected: list[NewsItem] = []

    def _tracking_news_fn(player_id: str | None, player_name: str) -> list[NewsItem]:
        items = news_fn(player_id, player_name)
        collected.extend(items)
        return items

    context = build_context(players, resolved_week, tables=tables, news_fn=_tracking_news_fn)
    answer = run_chat_llm(context, message, history)
    _check_recommendation(answer.recommendation, players)

    seen_links: set[str] = set()
    sources: list[NewsItem] = []
    for item in collected:
        if item.link not in seen_links:
            seen_links.add(item.link)
            sources.append(item)

    return ChatResult(
        answer=answer.answer,
        sources=sources,
        players_discussed=players,
        recommendation=answer.recommendation,
        week=resolved_week,
        context=context,
    )
