import polars as pl
import pytest

from pipeline.context_builder import PlayerNotFoundError, build_context
from pipeline.entity_extraction import extract_players, extract_week, known_player_names


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


def test_build_context_formats_two_player_comparison():
    """Matches the README §7 PLAYER COMPARISON layout: header, stats, projection, injury."""
    context = build_context(["Jordan Love", "Jared Goff"], week=5, tables=_fixture_tables())

    assert context.startswith("PLAYER COMPARISON\n\n")
    assert "Jordan Love:\n- Avg fantasy points (last 3 weeks): 18.4\n" \
        "- Projected points: 17.1\n- Injury: Questionable -> Full practice Friday" in context
    assert "Jared Goff:\n- Avg fantasy points (last 3 weeks): 12.1\n" \
        "- Projected points: 14.3\n- Injury: Healthy" in context


def test_build_context_raises_for_unknown_player():
    """A player missing from processed stats for the requested week is a clear error, not a silent skip."""
    with pytest.raises(PlayerNotFoundError):
        build_context(["Nobody Here"], week=5, tables=_fixture_tables())


def test_extract_week_finds_week_number_in_free_text():
    assert extract_week("Should I start Jordan Love in week 5?") == 5
    assert extract_week("who do I play in Week12") == 12
    assert extract_week("no week mentioned here") is None


def test_extract_players_matches_known_names_in_order_of_appearance():
    """"Love" must not falsely match "Loveland" - only whole-name matches count."""
    known_names = ["Jordan Love", "Colston Loveland", "Jared Goff"]
    text = "Should I start Jared Goff or Jordan Love this week?"
    assert extract_players(text, known_names=known_names) == ["Jared Goff", "Jordan Love"]


def test_extract_players_ignores_substring_collisions():
    known_names = ["Jordan Love", "Colston Loveland"]
    text = "Is Colston Loveland going to see more targets?"
    assert extract_players(text, known_names=known_names) == ["Colston Loveland"]


def test_known_player_names_drops_null_rows(monkeypatch):
    """A row with a null player_name (e.g. an unmapped team-level row) must not crash matching."""
    fixture = pl.DataFrame({"player_name": ["Jordan Love", None, "Jared Goff"]})
    monkeypatch.setattr("pipeline.entity_extraction.pl.read_parquet", lambda *_: fixture)
    assert known_player_names() == ["Jordan Love", "Jared Goff"]
