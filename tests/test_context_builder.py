import polars as pl
import pytest

from pipeline.context_builder import PlayerNotFoundError, build_context
from pipeline.entity_extraction import extract_players, extract_week, known_player_names
from retrieval.news_retriever import NewsItem


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


def _id_keyed_tables():
    """Real-data shape: player_id on every table, and names that disagree across
    tables (nflverse "Kenneth Walker III" vs Sleeper's "Kenneth Walker").
    """
    player_stats = pl.DataFrame({
        "player_id": ["00-1"],
        "player_name": ["Kenneth Walker III"],
        "week": [18],
        "avg_fantasy_points_last3": [11.0],
    })
    projections = pl.DataFrame({
        "player_id": ["00-1"],
        "player_name": ["Kenneth Walker"],
        "week": [18],
        "projected_points": [12.17],
    })
    injuries = pl.DataFrame({
        "player_id": ["00-1"],
        "player_name": ["Kenneth Walker"],
        "status": ["Questionable"],
        "practice_level": ["Limited Participation"],
    })
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_joins_on_player_id_not_name():
    """Suffixed names differ between sources; the projection and injury must still land."""
    context = build_context(["Kenneth Walker III"], week=18, tables=_id_keyed_tables())

    assert "- Projected points: 12.2" in context
    assert "- Injury: Questionable -> Limited Participation" in context


def test_build_context_prefers_a_real_injury_report_over_a_healthy_duplicate():
    """A player with two injury rows must never be reported healthy by row ordering."""
    tables = _id_keyed_tables()
    tables["injuries"] = pl.DataFrame({
        "player_id": ["00-1", "00-1"],
        "player_name": ["Kenneth Walker", "Kenneth Walker"],
        "status": ["Healthy", "Out"],
        "practice_level": [None, "Did Not Participate In Practice"],
    })

    context = build_context(["Kenneth Walker III"], week=18, tables=tables)
    assert "- Injury: Out -> Did Not Participate In Practice" in context


def test_build_context_ignores_another_players_projection_for_the_same_week():
    """Matching by id must not pick up a different player's row."""
    tables = _id_keyed_tables()
    tables["projections"] = pl.DataFrame({
        "player_id": ["00-2"],
        "player_name": ["Somebody Else"],
        "week": [18],
        "projected_points": [30.0],
    })

    context = build_context(["Kenneth Walker III"], week=18, tables=tables)
    assert "Projected points" not in context


def test_build_context_omits_news_bullet_when_news_fn_is_not_passed():
    """Pre-Phase-6 callers (no news_fn) must see byte-identical output - no bullet at all."""
    context = build_context(["Jordan Love", "Jared Goff"], week=5, tables=_fixture_tables())
    assert "Recent news" not in context


def test_build_context_omits_news_bullet_when_news_fn_returns_nothing():
    context = build_context(
        ["Jordan Love", "Jared Goff"], week=5, tables=_fixture_tables(), news_fn=lambda *_: []
    )
    assert "Recent news" not in context


def test_build_context_adds_a_news_bullet_per_snippet_when_news_fn_returns_some():
    def news_fn(player_id, player_name):
        if player_name != "Jordan Love":
            return []
        return [NewsItem(
            title="Packers stay aggressive",
            snippet="Packers plan to stay aggressive...",
            link="https://example.com/a",
            source="ESPN",
            published_at="2026-09-01T00:00:00+00:00",
        )]

    context = build_context(
        ["Jordan Love", "Jared Goff"], week=5, tables=_fixture_tables(), news_fn=news_fn
    )

    assert (
        "Jordan Love:\n- Avg fantasy points (last 3 weeks): 18.4\n"
        "- Projected points: 17.1\n- Injury: Questionable -> Full practice Friday\n"
        '- Recent news:\n  - "Packers plan to stay aggressive..."' in context
    )
    assert "Jared Goff:\n- Avg fantasy points (last 3 weeks): 12.1\n" \
        "- Projected points: 14.3\n- Injury: Healthy" in context
    assert "Recent news" not in context.split("Jared Goff:")[1]


def test_build_context_passes_player_id_to_news_fn():
    """news_fn must be able to filter by the same player_id the projections/injuries join uses."""
    seen = {}

    def news_fn(player_id, player_name):
        seen["player_id"] = player_id
        seen["player_name"] = player_name
        return []

    build_context(["Kenneth Walker III"], week=18, tables=_id_keyed_tables(), news_fn=news_fn)
    assert seen == {"player_id": "00-1", "player_name": "Kenneth Walker III"}


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


def test_extract_players_matches_names_ending_in_punctuation():
    """A trailing \\b fails after punctuation (e.g. "Jr."); the fix must still match it."""
    known_names = ["Marvin Harrison Jr."]
    text = "Start Marvin Harrison Jr. in week 5"
    assert extract_players(text, known_names=known_names) == ["Marvin Harrison Jr."]


def test_known_player_names_drops_null_rows(monkeypatch):
    """A row with a null player_name (e.g. an unmapped team-level row) must not crash matching."""
    fixture = pl.DataFrame({"player_name": ["Jordan Love", None, "Jared Goff"]})
    monkeypatch.setattr("pipeline.entity_extraction.pl.read_parquet", lambda *_: fixture)
    assert known_player_names() == ["Jordan Love", "Jared Goff"]
