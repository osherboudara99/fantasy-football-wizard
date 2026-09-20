import polars as pl
import pytest
from pydantic import ValidationError

from llm.interface import Recommendation
from pipeline.decision_engine import (
    DataUnavailableError,
    Decision,
    DecisionError,
    decide,
    format_decision,
    resolve_players,
    resolve_season,
    target_season_week,
    target_week,
)


def _fixture_tables():
    player_stats = pl.DataFrame([
        {"player_id": "00-love", "player_name": "Jordan Love", "season": 2026, "week": w,
         "pass_yards": 0, "pass_tds": 0, "pass_interceptions": 0, "pass_2pt": 0,
         "rush_yards": 0, "rush_tds": 0, "rush_2pt": 0, "rush_attempts": 0,
         "receptions": 1, "rec_yards": 195, "rec_tds": 0, "rec_2pt": 0, "fumbles_lost": 0}
        for w in range(1, 5)
    ] + [
        {"player_id": "00-goff", "player_name": "Jared Goff", "season": 2026, "week": w,
         "pass_yards": 0, "pass_tds": 0, "pass_interceptions": 0, "pass_2pt": 0,
         "rush_yards": 0, "rush_tds": 0, "rush_2pt": 0, "rush_attempts": 0,
         "receptions": 1, "rec_yards": 125, "rec_tds": 0, "rec_2pt": 0, "fumbles_lost": 0}
        for w in range(1, 5)
    ])
    projections = pl.DataFrame({
        "player_id": ["00-love", "00-goff"], "player_name": ["Jordan Love", "Jared Goff"],
        "season": [2026, 2026], "week": [5, 5],
        "pass_yards": [0, 0], "pass_tds": [0, 0], "pass_interceptions": [0, 0], "pass_2pt": [0, 0],
        "rush_yards": [0, 0], "rush_tds": [0, 0], "rush_2pt": [0, 0], "rush_attempts": [0, 0],
        "receptions": [1, 1], "rec_yards": [161, 133], "rec_tds": [0, 0], "rec_2pt": [0, 0],
        "fumbles_lost": [0, 0],
    })
    injuries = pl.DataFrame({
        "player_id": ["00-love"], "player_name": ["Jordan Love"],
        "status": ["Questionable"], "practice_level": ["Full practice Friday"],
    })
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def _no_news(*_args, **_kwargs):
    """Stub news_fn: keeps these tests hermetic, no real Chroma/model touch."""
    return []


def _fixture_recommendation():
    return Recommendation(
        start="Jordan Love",
        bench="Jared Goff",
        confidence=0.72,
        key_factors=["Higher recent average"],
        risk_factors=["Questionable injury tag"],
    )


@pytest.fixture
def stub_llm(monkeypatch):
    """Capture what decide() sends to the LLM without spending a real API call."""
    calls = {}

    def fake_run_llm(context, question=None):
        calls["context"] = context
        calls["question"] = question
        return _fixture_recommendation()

    monkeypatch.setattr("pipeline.decision_engine.run_llm", fake_run_llm)
    return calls


def test_decide_passes_built_context_and_question_to_the_llm(stub_llm):
    question = "Should I start Jordan Love or Jared Goff in week 5?"
    decision = decide(
        question, tables=_fixture_tables(), players=["Jordan Love", "Jared Goff"],
        season=2026, news_fn=_no_news,
    )

    assert isinstance(decision, Decision)
    assert decision.week == 5
    assert stub_llm["question"] == question
    assert stub_llm["context"].startswith("PLAYER COMPARISON\n\n")
    assert "Jordan Love:" in stub_llm["context"] and "Jared Goff:" in stub_llm["context"]
    assert decision.recommendation.start == "Jordan Love"


def test_decide_extracts_players_and_week_from_the_question(monkeypatch, stub_llm):
    """No explicit players/week: both come from the question text via entity extraction."""
    monkeypatch.setattr(
        "pipeline.entity_extraction.known_player_names",
        lambda: ["Jordan Love", "Jared Goff"],
    )
    decision = decide(
        "Who do I start in week 5, Jared Goff or Jordan Love?",
        tables=_fixture_tables(),
        season=2026,
        news_fn=_no_news,
    )

    assert decision.players == ["Jared Goff", "Jordan Love"]
    assert decision.week == 5


def test_decide_falls_back_to_target_week_when_question_has_none(monkeypatch, stub_llm):
    """A question without "week N" uses whatever week the processed data holds."""
    monkeypatch.setattr("pipeline.decision_engine.target_season_week", lambda: (2026, 5))
    decision = decide(
        "Jordan Love or Jared Goff?",
        players=["Jordan Love", "Jared Goff"],
        tables=_fixture_tables(),
        news_fn=_no_news,
    )

    assert decision.week == 5


def test_explicit_week_overrides_the_question_text(monkeypatch, stub_llm):
    tables = _fixture_tables()
    tables["player_stats"] = tables["player_stats"].with_columns(pl.lit(9).alias("week"))
    tables["projections"] = tables["projections"].with_columns(pl.lit(9).alias("week"))

    decision = decide(
        "Jordan Love or Jared Goff in week 5?",
        players=["Jordan Love", "Jared Goff"],
        season=2026,
        week=9,
        tables=tables,
        news_fn=_no_news,
    )
    assert decision.week == 9


def test_week_zero_in_the_question_is_not_swallowed_by_the_fallback(monkeypatch, stub_llm):
    """Week 0 is falsy - it must still be the resolved week, not silently replaced."""
    monkeypatch.setattr("pipeline.decision_engine.target_week", lambda: 18)
    tables = _fixture_tables()
    tables["player_stats"] = tables["player_stats"].with_columns(pl.lit(0).alias("week"))
    tables["projections"] = tables["projections"].with_columns(pl.lit(0).alias("week"))

    decision = decide(
        "Jordan Love or Jared Goff in week 0?",
        players=["Jordan Love", "Jared Goff"],
        season=2026,
        tables=tables,
        news_fn=_no_news,
    )
    assert decision.week == 0


def test_decide_defaults_news_fn_to_the_real_retriever(monkeypatch, stub_llm):
    """No news_fn passed: decide() must wire in retrieval.news_retriever.retrieve_news."""
    calls = []

    def fake_retrieve_news(player_id, player_name, *args, **kwargs):
        calls.append((player_id, player_name))
        return []

    monkeypatch.setattr("pipeline.decision_engine.retrieve_news", fake_retrieve_news)
    decide(
        "Jordan Love or Jared Goff in week 5?",
        players=["Jordan Love", "Jared Goff"],
        season=2026,
        tables=_fixture_tables(),
    )

    assert {name for _, name in calls} == {"Jordan Love", "Jared Goff"}


def test_decide_rejects_a_recommendation_about_other_players(monkeypatch):
    """A hallucinated name must fail loudly instead of rendering as a real answer."""
    monkeypatch.setattr(
        "pipeline.decision_engine.run_llm",
        lambda *_, **__: Recommendation(
            start="Somebody Else", bench="Jared Goff", confidence=0.9,
            key_factors=[], risk_factors=[],
        ),
    )
    with pytest.raises(DecisionError):
        decide(
            "Jordan Love or Jared Goff in week 5?",
            players=["Jordan Love", "Jared Goff"],
            season=2026,
            tables=_fixture_tables(),
            news_fn=_no_news,
        )


def test_recommendation_rejects_out_of_range_confidence():
    """Structured outputs guarantee shape; the schema has to guarantee the range."""
    with pytest.raises(ValidationError):
        Recommendation(
            start="Jordan Love", bench="Jared Goff", confidence=1.87,
            key_factors=[], risk_factors=[],
        )


@pytest.mark.parametrize("found", [[], ["Jordan Love"], ["Jordan Love", "Jared Goff", "Bo Nix"]])
def test_resolve_players_requires_exactly_two(found):
    """Anything but a two-player comparison is a clear error, not a guess."""
    with pytest.raises(DecisionError):
        resolve_players("some question", found)


def test_resolve_players_rejects_the_same_player_twice():
    """start == bench would satisfy the set-based output check, so block it up front."""
    with pytest.raises(DecisionError):
        resolve_players("some question", ["Jordan Love", "jordan love "])


def test_target_season_week_reads_the_meta_file(monkeypatch):
    fixture = pl.DataFrame({"season": [2026], "week": [8]})
    monkeypatch.setattr("pipeline.decision_engine.Path.exists", lambda self: True)
    monkeypatch.setattr("pipeline.decision_engine.pl.read_parquet", lambda *_: fixture)
    assert target_season_week() == (2026, 8)
    assert target_week() == 8


def test_target_season_week_raises_when_meta_file_is_missing(monkeypatch):
    monkeypatch.setattr("pipeline.decision_engine.Path.exists", lambda self: False)
    with pytest.raises(DataUnavailableError):
        target_season_week()


def test_target_season_week_raises_when_meta_file_is_empty(monkeypatch):
    empty = pl.DataFrame({"season": [], "week": []})
    monkeypatch.setattr("pipeline.decision_engine.Path.exists", lambda self: True)
    monkeypatch.setattr("pipeline.decision_engine.pl.read_parquet", lambda *_: empty)
    with pytest.raises(DataUnavailableError):
        target_season_week()


def test_resolve_season_prefers_the_explicit_argument(monkeypatch):
    monkeypatch.setattr("pipeline.decision_engine.target_season_week", lambda: (2026, 8))
    assert resolve_season(2025) == 2025


def test_resolve_season_falls_back_to_the_meta_file(monkeypatch):
    monkeypatch.setattr("pipeline.decision_engine.target_season_week", lambda: (2026, 8))
    assert resolve_season(None) == 2026


def test_format_decision_renders_the_recommendation_fields():
    decision = Decision(
        question="Jordan Love or Jared Goff?",
        players=["Jordan Love", "Jared Goff"],
        week=5,
        context="PLAYER COMPARISON",
        recommendation=_fixture_recommendation(),
    )
    output = format_decision(decision)

    assert "Week 5: Jordan Love vs Jared Goff" in output
    assert "START: Jordan Love" in output
    assert "BENCH: Jared Goff" in output
    assert "Confidence: 72%" in output
    assert "  - Higher recent average" in output
    assert "  - Questionable injury tag" in output
