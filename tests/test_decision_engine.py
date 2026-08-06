import polars as pl
import pytest
from pydantic import ValidationError

from llm.interface import Recommendation
from pipeline.decision_engine import (
    Decision,
    DecisionError,
    decide,
    format_decision,
    resolve_players,
    target_week,
)


def _fixture_tables():
    player_stats = pl.DataFrame({
        "player_name": ["Jordan Love", "Jared Goff"],
        "week": [5, 5],
        "avg_fantasy_points_last3": [18.4, 12.1],
    })
    projections = pl.DataFrame({
        "player_name": ["Jordan Love", "Jared Goff"],
        "week": [5, 5],
        "projected_points": [17.1, 14.3],
    })
    injuries = pl.DataFrame({
        "player_name": ["Jordan Love"],
        "status": ["Questionable"],
        "practice_level": ["Full practice Friday"],
    })
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


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
    decision = decide(question, tables=_fixture_tables(), players=["Jordan Love", "Jared Goff"])

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
        "Who do I start in week 5, Jared Goff or Jordan Love?", tables=_fixture_tables()
    )

    assert decision.players == ["Jared Goff", "Jordan Love"]
    assert decision.week == 5


def test_decide_falls_back_to_target_week_when_question_has_none(monkeypatch, stub_llm):
    """A question without "week N" uses whatever week the processed data holds."""
    monkeypatch.setattr("pipeline.decision_engine.target_week", lambda: 5)
    decision = decide(
        "Jordan Love or Jared Goff?",
        players=["Jordan Love", "Jared Goff"],
        tables=_fixture_tables(),
    )

    assert decision.week == 5


def test_explicit_week_overrides_the_question_text(monkeypatch, stub_llm):
    tables = _fixture_tables()
    tables["player_stats"] = tables["player_stats"].with_columns(pl.lit(9).alias("week"))
    tables["projections"] = tables["projections"].with_columns(pl.lit(9).alias("week"))

    decision = decide(
        "Jordan Love or Jared Goff in week 5?",
        players=["Jordan Love", "Jared Goff"],
        week=9,
        tables=tables,
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
        tables=tables,
    )
    assert decision.week == 0


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
            tables=_fixture_tables(),
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


def test_target_week_reads_the_max_week_in_processed_stats(monkeypatch):
    fixture = pl.DataFrame({"week": [16, 18, 17]})
    monkeypatch.setattr("pipeline.decision_engine.pl.read_parquet", lambda *_, **__: fixture)
    assert target_week() == 18


def test_target_week_raises_on_empty_processed_stats(monkeypatch):
    empty = pl.DataFrame({"week": []}, schema={"week": pl.Int32})
    monkeypatch.setattr("pipeline.decision_engine.pl.read_parquet", lambda *_, **__: empty)
    with pytest.raises(DecisionError):
        target_week()


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
