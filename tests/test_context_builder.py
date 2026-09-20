import polars as pl
import pytest

from pipeline.context_builder import PlayerNotFoundError, build_context
from pipeline.entity_extraction import extract_players, extract_week, known_player_names
from pipeline.scoring import PRESET_HALF_PPR, PRESET_STANDARD
from retrieval.news_retriever import NewsItem


def _game_row(player_id, name, season, week, **stats):
    base = {
        "player_id": player_id, "player_name": name, "position": "WR", "team": "MIN",
        "season": season, "week": week,
        "pass_yards": 0, "pass_tds": 0, "pass_interceptions": 0, "pass_2pt": 0,
        "rush_yards": 0, "rush_tds": 0, "rush_2pt": 0, "rush_attempts": 0,
        "receptions": 0, "rec_yards": 0, "rec_tds": 0, "rec_2pt": 0, "fumbles_lost": 0,
    }
    base.update(stats)
    return base


def _fixture_tables():
    """Jordan Love and Jared Goff, 4 games played this season (weeks 1-4),
    week 5 is the target (not yet played). Round per-game rates (5 receptions
    always, rec_yards varies) so every tier's math is easy to hand-check:
    Love = 100 rec_yards/game -> PPR 15.0, Half-PPR 12.5, Standard 10.0 per game.
    Goff = 50 rec_yards/game -> PPR 10.0 per game.
    """
    rows = [
        _game_row("00-love", "Jordan Love", 2026, w, receptions=5, rec_yards=100)
        for w in range(1, 5)
    ] + [
        _game_row("00-goff", "Jared Goff", 2026, w, receptions=5, rec_yards=50)
        for w in range(1, 5)
    ]
    player_stats = pl.DataFrame(rows)
    projections = pl.DataFrame([
        _game_row("00-love", "Jordan Love", 2026, 5, receptions=5, rec_yards=80),
        _game_row("00-goff", "Jared Goff", 2026, 5, receptions=5, rec_yards=60),
    ])
    injuries = pl.DataFrame({
        "player_id": ["00-love"], "player_name": ["Jordan Love"],
        "status": ["Questionable"], "practice_level": ["Full practice Friday"],
    })
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_formats_two_player_comparison():
    """Matches the README §7 PLAYER COMPARISON layout: header, stats, projection, injury.

    Defaults to PPR: Love's 100 rec_yards + 5 receptions/game = 10.0 + 5.0 = 15.0;
    projection (80 rec_yards + 5 receptions) = 8.0 + 5.0 = 13.0. Goff: 50 rec_yards
    + 5 receptions/game = 5.0 + 5.0 = 10.0; projection (60 + 5) = 6.0 + 5.0 = 11.0.
    """
    context = build_context(
        ["Jordan Love", "Jared Goff"], season=2026, week=5, tables=_fixture_tables()
    )

    assert context.startswith("PLAYER COMPARISON\n\n")
    assert (
        "Jordan Love:\n- This season (4 games): 15.0 season avg, 15.0 avg over last 3 games\n"
        "- Projected points: 13.0\n- Injury: Questionable -> Full practice Friday" in context
    )
    assert (
        "Jared Goff:\n- This season (4 games): 10.0 season avg, 10.0 avg over last 3 games\n"
        "- Projected points: 11.0\n- Injury: Healthy" in context
    )


def test_build_context_reflects_the_selected_scoring_tier():
    """Same fixture, different tier - the numbers must actually change.

    Love's 100 rec_yards + 5 receptions per game: PPR = 10.0 + 5*1.0 = 15.0;
    Standard = 10.0 + 5*0.0 = 10.0; Half-PPR = 10.0 + 5*0.5 = 12.5.
    """
    tables = _fixture_tables()
    ppr_context = build_context(["Jordan Love"], season=2026, week=5, tables=tables)
    standard_context = build_context(
        ["Jordan Love"], season=2026, week=5, tables=tables, scoring_rules=PRESET_STANDARD
    )
    half_context = build_context(
        ["Jordan Love"], season=2026, week=5, tables=tables, scoring_rules=PRESET_HALF_PPR
    )

    assert "15.0 season avg" in ppr_context
    assert "10.0 season avg" in standard_context
    assert "12.5 season avg" in half_context


def test_build_context_raises_for_unknown_player():
    with pytest.raises(PlayerNotFoundError):
        build_context(["Nobody Here"], season=2026, week=5, tables=_fixture_tables())


def test_build_context_omits_news_bullet_when_news_fn_is_not_passed():
    context = build_context(
        ["Jordan Love", "Jared Goff"], season=2026, week=5, tables=_fixture_tables()
    )
    assert "Recent news" not in context


def test_build_context_adds_a_news_bullet_per_snippet_when_news_fn_returns_some():
    def news_fn(player_id, player_name):
        if player_name != "Jordan Love":
            return []
        return [NewsItem(
            title="Packers stay aggressive", snippet="Packers plan to stay aggressive...",
            link="https://example.com/a", source="ESPN", published_at="2026-09-01T00:00:00+00:00",
        )]

    context = build_context(
        ["Jordan Love", "Jared Goff"], season=2026, week=5,
        tables=_fixture_tables(), news_fn=news_fn,
    )
    assert '- Recent news:\n  - "Packers plan to stay aggressive..."' in context
    assert "Recent news" not in context.split("Jared Goff:")[1]


def _partial_current_season_tables(games_this_season=1, prior_games=4):
    """A player with a thin current-season sample - last season is still
    relevant. 60 rec_yards + 4 receptions/game (PPR default) = 6.0 + 4.0 = 10.0
    per game, both seasons, so every branch's expected number is the same
    round 10.0 regardless of how many games are in the window.
    """
    rows = [
        _game_row("00-nabers", "Malik Nabers", 2026, w, receptions=4, rec_yards=60)
        for w in range(1, 1 + games_this_season)
    ] + [
        _game_row("00-nabers", "Malik Nabers", 2025, w, receptions=4, rec_yards=60)
        for w in range(1, 1 + prior_games)
    ]
    player_stats = pl.DataFrame(rows)
    projections = pl.DataFrame({"player_id": [], "player_name": [], "season": [], "week": []}, schema={
        "player_id": pl.String, "player_name": pl.String, "season": pl.Int64, "week": pl.Int64,
    })
    injuries = pl.DataFrame({"player_id": [], "player_name": [], "status": [], "practice_level": []})
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_shows_last_season_separately_when_this_season_is_thin():
    context = build_context(
        ["Malik Nabers"], season=2026, week=2, tables=_partial_current_season_tables()
    )
    assert "- This season (1 game): 10.0 avg fantasy points" in context
    assert "- Last season (4 games): 10.0 season avg, 10.0 avg over final 3 games" in context
    assert "- Most recent game played (Week 1, 2026): 10.0 pts" in context


def test_build_context_omits_last_season_once_three_current_season_games_exist():
    context = build_context(
        ["Malik Nabers"], season=2026, week=4,
        tables=_partial_current_season_tables(games_this_season=3),
    )
    assert "Last season" not in context
    assert "Most recent game played" not in context


def test_build_context_shows_only_last_season_when_no_games_played_yet():
    context = build_context(
        ["Malik Nabers"], season=2026, week=1,
        tables=_partial_current_season_tables(games_this_season=0),
    )
    assert "- This season: no games played yet" in context
    assert "- Last season (4 games): 10.0 season avg, 10.0 avg over final 3 games" in context


def test_build_context_omits_last_season_line_when_player_has_none():
    context = build_context(
        ["Malik Nabers"], season=2026, week=2,
        tables=_partial_current_season_tables(prior_games=0),
    )
    assert "Last season" not in context
    assert "- Most recent game played (Week 1, 2026): 10.0 pts" in context


def _duplicate_display_name_tables():
    """Two distinct real players share the display name "Byron Young" (confirmed
    on real data, along with "Jaylon Jones"/"Jordan Phillips"/"Marcus Harris"/
    "Aaron Brewer"). "00-byron-a" is the first row for that name in the table
    (2 games this season); "00-byron-b" is a completely different player (3
    games, same season). A name-only filter would blend both players' rows
    into one 5-game average - `_format_player` must instead resolve to a
    single player_id (the first row's, matching `_match_player`'s existing
    first-row semantics) before computing recent form.
    """
    rows = [
        _game_row("00-byron-a", "Byron Young", 2026, w, receptions=4, rec_yards=60)
        for w in range(1, 3)
    ] + [
        _game_row("00-byron-b", "Byron Young", 2026, w, receptions=4, rec_yards=60)
        for w in range(1, 4)
    ]
    player_stats = pl.DataFrame(rows)
    projections = pl.DataFrame({"player_id": [], "player_name": [], "season": [], "week": []}, schema={
        "player_id": pl.String, "player_name": pl.String, "season": pl.Int64, "week": pl.Int64,
    })
    injuries = pl.DataFrame({"player_id": [], "player_name": [], "status": [], "practice_level": []})
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_does_not_blend_two_players_sharing_a_display_name():
    context = build_context(
        ["Byron Young"], season=2026, week=5, tables=_duplicate_display_name_tables()
    )
    # 00-byron-a is the first row for "Byron Young" and has 2 games this season.
    # Blending both players' rows would report 5 games instead.
    assert "- This season (2 games)" in context
    assert "5 games" not in context


def _no_real_projection_tables():
    """A projections row exists for the target week, but every one of the 13
    canonical stat fields is null/omitted - real Sleeper snapshots commonly
    have no real number yet for most players. This must render as "no
    projection", not a false "Projected points: 0.0".
    """
    player_stats = pl.DataFrame([
        _game_row("00-nabers", "Malik Nabers", 2026, w, receptions=4, rec_yards=60)
        for w in range(1, 4)
    ])
    projections = pl.DataFrame([
        {"player_id": "00-nabers", "player_name": "Malik Nabers", "season": 2026, "week": 5}
    ])
    injuries = pl.DataFrame({"player_id": [], "player_name": [], "status": [], "practice_level": []})
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_omits_projected_points_when_no_real_projection_exists():
    context = build_context(
        ["Malik Nabers"], season=2026, week=5, tables=_no_real_projection_tables()
    )
    assert "Projected points" not in context


def _multi_week_tables():
    """Jordan Love, 6 games played this season with a scoring jump after week
    3 (100 rec_yards/game weeks 1-3, 200 rec_yards/game weeks 4-6) - lets a
    past-week question (week 3) be distinguished from a question about the
    upcoming week (7, not yet played). PPR: 100 rec_yards + 5 receptions =
    15.0/game; 200 rec_yards + 5 receptions = 25.0/game.
    """
    rows = [
        _game_row("00-love", "Jordan Love", 2026, w, receptions=5, rec_yards=100)
        for w in range(1, 4)
    ] + [
        _game_row("00-love", "Jordan Love", 2026, w, receptions=5, rec_yards=200)
        for w in range(4, 7)
    ]
    player_stats = pl.DataFrame(rows)
    projections = pl.DataFrame({"player_id": [], "player_name": [], "season": [], "week": []}, schema={
        "player_id": pl.String, "player_name": pl.String, "season": pl.Int64, "week": pl.Int64,
    })
    injuries = pl.DataFrame({"player_id": [], "player_name": [], "status": [], "practice_level": []})
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_shows_the_asked_about_past_weeks_actual_line():
    context = build_context(["Jordan Love"], season=2026, week=3, tables=_multi_week_tables())

    # season avg/last3 through week 3 only - week 4-6's 25.0/game must not leak in
    assert "- This season (3 games): 15.0 season avg, 15.0 avg over last 3 games" in context
    assert "25.0" not in context
    assert "- Week 3 actual: 15.0 pts (5 receptions, 100 rec yds)" in context
    assert (
        "- Note: only the real stat line above reflects week 3 itself - there's "
        "no historical projection for that week, and the injury status and any "
        "news shown below are today's, not from back then." in context
    )


def test_build_context_omits_the_past_week_note_for_the_normal_upcoming_week_question():
    context = build_context(["Jordan Love"], season=2026, week=7, tables=_multi_week_tables())

    assert "actual:" not in context
    assert "Note: only the real stat line" not in context


def test_build_context_pluralizes_stat_labels_correctly():
    """_format_stat_breakdown's plural forms must be grammatically correct -
    naive label+'s' produces "carrys" and "fumble losts", which would read as
    sloppy (or get echoed verbatim) in an LLM-facing answer.
    """
    rows = [
        _game_row(
            "00-back", "Test Back", 2026, 1,
            receptions=1, rec_yards=10, rush_attempts=12, rush_yards=50, fumbles_lost=2,
        )
    ]
    player_stats = pl.DataFrame(rows)
    projections = pl.DataFrame({"player_id": [], "player_name": [], "season": [], "week": []}, schema={
        "player_id": pl.String, "player_name": pl.String, "season": pl.Int64, "week": pl.Int64,
    })
    injuries = pl.DataFrame({"player_id": [], "player_name": [], "status": [], "practice_level": []})
    tables = {"player_stats": player_stats, "projections": projections, "injuries": injuries}

    context = build_context(["Test Back"], season=2026, week=1, tables=tables)

    assert "12 carries" in context
    assert "1 reception" in context
    assert "2 fumbles lost" in context
    assert "carrys" not in context
    assert "fumble losts" not in context


def _future_debut_tables():
    """A player whose first career game is week 5 - asking about week 1 (before
    their debut, with no prior-season rows either) leaves zero eligible games,
    so `compute_recent_form`'s last_game_* fields are all None. Formatting
    that "most recent game" line must not crash.
    """
    rows = [_game_row("00-rookie", "Late Debut", 2026, 5, receptions=3, rec_yards=40)]
    player_stats = pl.DataFrame(rows)
    projections = pl.DataFrame({"player_id": [], "player_name": [], "season": [], "week": []}, schema={
        "player_id": pl.String, "player_name": pl.String, "season": pl.Int64, "week": pl.Int64,
    })
    injuries = pl.DataFrame({"player_id": [], "player_name": [], "status": [], "practice_level": []})
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_handles_a_player_with_no_eligible_games_before_the_asked_week():
    context = build_context(["Late Debut"], season=2026, week=1, tables=_future_debut_tables())

    assert "This season: no games played yet" in context
    assert "Most recent game played" not in context


def _cross_season_week_collision_tables():
    """Real projections only ever hold the CURRENT target season's snapshot,
    but its week number can coincide with a different (past) season's week
    being asked about via an explicit `season` override. A lookup keyed on
    `week` alone would attribute season 2026's week-6 projection to a
    season-2025 week-6 question just because the numbers match.
    """
    player_stats = pl.DataFrame([
        _game_row("00-love", "Jordan Love", 2025, 6, receptions=5, rec_yards=50),
    ])
    projections = pl.DataFrame([
        _game_row("00-love", "Jordan Love", 2026, 6, receptions=5, rec_yards=999),
    ])
    injuries = pl.DataFrame({"player_id": [], "player_name": [], "status": [], "practice_level": []})
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_does_not_attribute_a_different_seasons_projection_by_week_number_alone():
    context = build_context(
        ["Jordan Love"], season=2025, week=6, tables=_cross_season_week_collision_tables()
    )

    assert "Projected points" not in context


def _bye_week_tables():
    """Jordan Love played weeks 1, 2, 4, 5 (bye in week 3). Asking about week
    3 has no stat line for this player, but later games prove the query is
    still historical - the live-data disclaimer must appear even without a
    "Week 3 actual" line to attach it to.
    """
    rows = [
        _game_row("00-love", "Jordan Love", 2026, w, receptions=5, rec_yards=100)
        for w in [1, 2, 4, 5]
    ]
    player_stats = pl.DataFrame(rows)
    projections = pl.DataFrame({"player_id": [], "player_name": [], "season": [], "week": []}, schema={
        "player_id": pl.String, "player_name": pl.String, "season": pl.Int64, "week": pl.Int64,
    })
    injuries = pl.DataFrame({"player_id": [], "player_name": [], "status": [], "practice_level": []})
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_shows_the_historical_note_on_a_bye_week_with_no_stat_line():
    context = build_context(["Jordan Love"], season=2026, week=3, tables=_bye_week_tables())

    assert "actual:" not in context
    assert (
        "- Note: no recorded stat line for week 3 for this player (bye, injury, or "
        "otherwise didn't play) - there's no historical projection for it either, "
        "and the injury status and any news shown below are today's, not from back "
        "then." in context
    )


def test_extract_week_finds_week_number_in_free_text():
    assert extract_week("Should I start Jordan Love in week 5?") == 5
    assert extract_week("who do I play in Week12") == 12
    assert extract_week("no week mentioned here") is None


def test_extract_players_matches_known_names_in_order_of_appearance():
    known_names = ["Jordan Love", "Colston Loveland", "Jared Goff"]
    text = "Should I start Jared Goff or Jordan Love this week?"
    assert extract_players(text, known_names=known_names) == ["Jared Goff", "Jordan Love"]


def test_extract_players_ignores_substring_collisions():
    known_names = ["Jordan Love", "Colston Loveland"]
    text = "Is Colston Loveland going to see more targets?"
    assert extract_players(text, known_names=known_names) == ["Colston Loveland"]


def test_extract_players_matches_names_ending_in_punctuation():
    known_names = ["Marvin Harrison Jr."]
    text = "Start Marvin Harrison Jr. in week 5"
    assert extract_players(text, known_names=known_names) == ["Marvin Harrison Jr."]


def test_known_player_names_drops_null_rows(monkeypatch):
    fixture = pl.DataFrame({"player_name": ["Jordan Love", None, "Jared Goff"]})
    monkeypatch.setattr("pipeline.entity_extraction.pl.read_parquet", lambda *_: fixture)
    assert known_player_names() == ["Jordan Love", "Jared Goff"]
