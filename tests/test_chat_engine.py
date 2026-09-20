import polars as pl
import pytest

from llm.interface import ChatAnswer, Recommendation
from pipeline.chat_engine import NoPlayersFoundError, chat, resolve_chat_players
from pipeline.decision_engine import DecisionError
from retrieval.news_retriever import NewsItem


def _fixture_tables():
    """Round per-game rates so the math is easy to hand-check: Love = 100
    rec_yards + 5 receptions/game -> PPR 15.0/game. Goff = 50 rec_yards + 5
    receptions -> PPR 10.0. Nix = 70 rec_yards + 5 receptions -> PPR 12.0.
    """
    player_stats = pl.DataFrame([
        {"player_id": pid, "player_name": name, "season": 2026, "week": w,
         "pass_yards": 0, "pass_tds": 0, "pass_interceptions": 0, "pass_2pt": 0,
         "rush_yards": 0, "rush_tds": 0, "rush_2pt": 0, "rush_attempts": 0,
         "receptions": 5, "rec_yards": rec_yards, "rec_tds": 0, "rec_2pt": 0, "fumbles_lost": 0}
        for pid, name, rec_yards in [
            ("00-love", "Jordan Love", 100), ("00-goff", "Jared Goff", 50), ("00-nix", "Bo Nix", 70),
        ]
        for w in range(1, 5)
    ])
    projections = pl.DataFrame({
        "player_id": ["00-love", "00-goff", "00-nix"],
        "player_name": ["Jordan Love", "Jared Goff", "Bo Nix"],
        "season": [2026, 2026, 2026], "week": [5, 5, 5],
        "pass_yards": [0, 0, 0], "pass_tds": [0, 0, 0], "pass_interceptions": [0, 0, 0],
        "pass_2pt": [0, 0, 0], "rush_yards": [0, 0, 0], "rush_tds": [0, 0, 0],
        "rush_2pt": [0, 0, 0], "rush_attempts": [0, 0, 0],
        "receptions": [5, 5, 5], "rec_yards": [80, 60, 70], "rec_tds": [0, 0, 0],
        "rec_2pt": [0, 0, 0], "fumbles_lost": [0, 0, 0],
    })
    injuries = pl.DataFrame({
        "player_id": ["00-love"], "player_name": ["Jordan Love"],
        "status": ["Questionable"], "practice_level": ["Full practice Friday"],
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
        season=2026, week=5,
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
        season=2026, week=5,
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
        season=2026, week=5,
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
        season=2026, week=5,
        tables=_fixture_tables(),
        news_fn=lambda *_: [],
    )
    assert result.recommendation == rec


def test_chat_leaves_recommendation_none_for_a_general_question(stub_run_chat_llm):
    result = chat(
        "How many points will Jordan Love score?",
        mentioned_players=["Jordan Love"],
        season=2026, week=5,
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
            season=2026, week=5,
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
            season=2026, week=5,
            tables=_fixture_tables(),
            news_fn=lambda *_: [],
        )


def test_chat_passes_scoring_rules_through_to_context(stub_run_chat_llm):
    from pipeline.scoring import PRESET_STANDARD

    chat(
        "Should I start Jordan Love or Jared Goff?",
        mentioned_players=["Jordan Love", "Jared Goff"],
        season=2026, week=5,
        tables=_fixture_tables(), news_fn=lambda *_: [],
        scoring_rules=PRESET_STANDARD,
    )
    # Love's PPR season avg would be 15.0 (10.0 rec_yards + 5.0 reception credit).
    # Standard drops the reception credit entirely - if scoring_rules were being
    # silently ignored, "15.0" would still show up; it must not.
    assert "15.0" not in stub_run_chat_llm["context"]
    assert "10.0 season avg" in stub_run_chat_llm["context"]
