import polars as pl
import pytest

from llm.interface import ChatAnswer, Recommendation
from pipeline.chat_engine import NoPlayersFoundError, chat, resolve_chat_players
from pipeline.decision_engine import DecisionError
from retrieval.news_retriever import NewsItem


def _fixture_tables():
    player_stats = pl.DataFrame({
        "player_name": ["Jordan Love", "Jared Goff", "Bo Nix"],
        "week": [5, 5, 5],
        "games_played_this_season": [4, 4, 4],
        "avg_fantasy_points_ppr_season": [17.9, 11.8, 14.5],
        "avg_fantasy_points_ppr_last3": [18.4, 12.1, 15.0],
        "prior_season_games_played": [0, 0, 0],
    })
    projections = pl.DataFrame({
        "player_name": ["Jordan Love", "Jared Goff", "Bo Nix"],
        "week": [5, 5, 5],
        "projected_points": [17.1, 14.3, 16.0],
    })
    injuries = pl.DataFrame({
        "player_name": ["Jordan Love"],
        "status": ["Questionable"],
        "practice_level": ["Full practice Friday"],
    })
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def _news_item(name, link="https://example.com/a"):
    return NewsItem(
        title=f"{name} update", snippet=f"{name} update\nDetails.",
        link=link, source="ESPN", published_at="2026-09-01T00:00:00+00:00",
    )


@pytest.fixture
def stub_run_chat_llm(monkeypatch):
    """Capture what chat() sends to the LLM without spending a real API call."""
    calls = {}

    def fake(context, message, history=None):
        calls.update(context=context, message=message, history=history)
        return ChatAnswer(answer="Start Jordan Love, he has the better matchup.")

    monkeypatch.setattr("pipeline.chat_engine.run_chat_llm", fake)
    return calls


def test_resolve_chat_players_prefers_mentions_then_extracted_text(monkeypatch):
    monkeypatch.setattr("pipeline.chat_engine.extract_players", lambda message: ["Bo Nix"])
    result = resolve_chat_players("Should I start him over Bo Nix?", ["Jordan Love"])
    assert result == ["Jordan Love", "Bo Nix"]


def test_resolve_chat_players_deduplicates_case_insensitively(monkeypatch):
    monkeypatch.setattr("pipeline.chat_engine.extract_players", lambda message: ["jordan love"])
    result = resolve_chat_players("Jordan Love?", ["Jordan Love"])
    assert result == ["Jordan Love"]


def test_resolve_chat_players_raises_when_nothing_found(monkeypatch):
    monkeypatch.setattr("pipeline.chat_engine.extract_players", lambda message: [])
    with pytest.raises(NoPlayersFoundError):
        resolve_chat_players("How's the waiver wire looking?", [])


def test_chat_builds_context_for_all_resolved_players(stub_run_chat_llm):
    result = chat(
        "Should I start Jordan Love or Jared Goff?",
        mentioned_players=["Jordan Love", "Jared Goff"],
        week=5,
        tables=_fixture_tables(),
        news_fn=lambda *_: [],
    )
    assert "Jordan Love:" in stub_run_chat_llm["context"]
    assert "Jared Goff:" in stub_run_chat_llm["context"]
    assert result.players_discussed == ["Jordan Love", "Jared Goff"]
    assert result.week == 5
    assert result.answer == "Start Jordan Love, he has the better matchup."


def test_chat_passes_history_through_to_the_llm(stub_run_chat_llm):
    history = [{"role": "user", "content": "Who's playing this week?"}]
    chat(
        "What about Bo Nix?",
        mentioned_players=["Bo Nix"],
        week=5,
        history=history,
        tables=_fixture_tables(),
        news_fn=lambda *_: [],
    )
    assert stub_run_chat_llm["history"] == history


def test_chat_collects_and_deduplicates_sources_across_players(stub_run_chat_llm):
    shared = _news_item("shared", link="https://example.com/shared")

    def news_fn(player_id, player_name):
        return [shared] if player_name in ("Jordan Love", "Jared Goff") else []

    result = chat(
        "Should I start Jordan Love or Jared Goff?",
        mentioned_players=["Jordan Love", "Jared Goff"],
        week=5,
        tables=_fixture_tables(),
        news_fn=news_fn,
    )
    assert result.sources == [shared]


def test_chat_passes_through_a_recommendation_when_the_llm_returns_one(monkeypatch):
    rec = Recommendation(
        start="Jordan Love", bench="Jared Goff", confidence=0.8,
        key_factors=["Better matchup"], risk_factors=[],
    )
    monkeypatch.setattr(
        "pipeline.chat_engine.run_chat_llm",
        lambda *_a, **_k: ChatAnswer(answer="Start Jordan Love.", recommendation=rec),
    )
    result = chat(
        "Should I start Jordan Love or Jared Goff?",
        mentioned_players=["Jordan Love", "Jared Goff"],
        week=5,
        tables=_fixture_tables(),
        news_fn=lambda *_: [],
    )
    assert result.recommendation == rec


def test_chat_leaves_recommendation_none_for_a_general_question(stub_run_chat_llm):
    result = chat(
        "How many points will Jordan Love score?",
        mentioned_players=["Jordan Love"],
        week=5,
        tables=_fixture_tables(),
        news_fn=lambda *_: [],
    )
    assert result.recommendation is None


def test_chat_rejects_a_recommendation_naming_an_undiscussed_player(monkeypatch):
    rec = Recommendation(
        start="Somebody Else", bench="Jared Goff", confidence=0.8,
        key_factors=[], risk_factors=[],
    )
    monkeypatch.setattr(
        "pipeline.chat_engine.run_chat_llm",
        lambda *_a, **_k: ChatAnswer(answer="x", recommendation=rec),
    )
    with pytest.raises(DecisionError):
        chat(
            "Should I start Jordan Love or Jared Goff?",
            mentioned_players=["Jordan Love", "Jared Goff"],
            week=5,
            tables=_fixture_tables(),
            news_fn=lambda *_: [],
        )


def test_chat_rejects_a_recommendation_starting_and_benching_the_same_player(monkeypatch):
    """A degenerate start==bench recommendation must not slip past the subset check
    just because a one-element set is trivially a subset of the discussed players.
    """
    rec = Recommendation(
        start="Jordan Love", bench="jordan love", confidence=0.8,
        key_factors=[], risk_factors=[],
    )
    monkeypatch.setattr(
        "pipeline.chat_engine.run_chat_llm",
        lambda *_a, **_k: ChatAnswer(answer="x", recommendation=rec),
    )
    with pytest.raises(DecisionError):
        chat(
            "Should I start Jordan Love or Jared Goff?",
            mentioned_players=["Jordan Love", "Jared Goff"],
            week=5,
            tables=_fixture_tables(),
            news_fn=lambda *_: [],
        )
