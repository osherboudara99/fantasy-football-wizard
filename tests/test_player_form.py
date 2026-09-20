import polars as pl
import pytest

from pipeline.player_form import compute_recent_form
from pipeline.scoring import PRESET_PPR, ScoringRules


def _rows(seasons, weeks, rec_yards):
    n = len(weeks)
    return pl.DataFrame({
        "season": seasons, "week": weeks,
        "receptions": [0] * n, "rec_yards": rec_yards, "rec_tds": [0] * n,
    })


def test_compute_recent_form_averages_this_seasons_last_3_games():
    rows = _rows(seasons=[2026] * 3, weeks=[1, 2, 3], rec_yards=[50, 100, 150])
    form = compute_recent_form(rows, season=2026, week=3, rules=PRESET_PPR)

    assert form["games_played_this_season"] == 3
    assert form["avg_fantasy_points_last3"] == pytest.approx(10.0)  # (5+10+15)/3
    assert form["avg_fantasy_points_season"] == pytest.approx(10.0)
    assert form["prior_season_games_played"] == 0
    assert form["last_game_season"] == 2026 and form["last_game_week"] == 3
    assert form["last_game_fantasy_points"] == pytest.approx(15.0)


def test_compute_recent_form_keeps_seasons_separate_when_current_is_thin():
    rows = _rows(
        seasons=[2025, 2025, 2025], weeks=[16, 17, 18],
        rec_yards=[60, 120, 180],
    )
    form = compute_recent_form(rows, season=2026, week=1, rules=PRESET_PPR)

    assert form["games_played_this_season"] == 0
    assert form["avg_fantasy_points_last3"] is None
    assert form["avg_fantasy_points_season"] is None
    assert form["prior_season_games_played"] == 3
    assert form["prior_season_avg_fantasy_points"] == pytest.approx(12.0)  # (6+12+18)/3
    assert form["prior_season_last3_avg_fantasy_points"] == pytest.approx(12.0)
    # the most recent game overall, regardless of season
    assert form["last_game_season"] == 2025 and form["last_game_week"] == 18
    assert form["last_game_fantasy_points"] == pytest.approx(18.0)


def test_compute_recent_form_reflects_the_selected_scoring_rules():
    """The same rows score differently under Standard vs PPR - proves the
    aggregation actually uses `rules`, not a hard-coded formula.
    """
    rows = _rows(seasons=[2026], weeks=[1], rec_yards=[0])
    rows = rows.with_columns(pl.lit(10).alias("receptions"))
    standard = compute_recent_form(rows, season=2026, week=1, rules=ScoringRules())
    ppr = compute_recent_form(rows, season=2026, week=1, rules=PRESET_PPR)

    assert standard["avg_fantasy_points_season"] == pytest.approx(0.0)
    assert ppr["avg_fantasy_points_season"] == pytest.approx(10.0)


def test_compute_recent_form_handles_a_rookie_with_no_prior_season_rows():
    rows = _rows(seasons=[2026], weeks=[1], rec_yards=[50])
    form = compute_recent_form(rows, season=2026, week=1, rules=PRESET_PPR)

    assert form["prior_season_games_played"] == 0
    assert form["prior_season_avg_fantasy_points"] is None
    assert form["prior_season_last3_avg_fantasy_points"] is None


def test_compute_recent_form_excludes_games_after_the_asked_about_week():
    """Asking about week 3 while the player has since played weeks 4-5 must
    not let those later games leak into "this season"/"last 3" averages -
    the real bug this test guards against (a past-week question silently
    answered with numbers from games that hadn't happened yet, from its
    perspective).
    """
    rows = _rows(
        seasons=[2026] * 5, weeks=[1, 2, 3, 4, 5],
        rec_yards=[10, 20, 30, 40, 50],
    )
    form = compute_recent_form(rows, season=2026, week=3, rules=PRESET_PPR)

    # only weeks 1-3 count: (1.0+2.0+3.0)/3 = 2.0 fantasy points/game (rec_yards*0.1)
    assert form["games_played_this_season"] == 3
    assert form["avg_fantasy_points_season"] == pytest.approx(2.0)
    assert form["avg_fantasy_points_last3"] == pytest.approx(2.0)
    assert form["last_game_season"] == 2026 and form["last_game_week"] == 3
    assert form["last_game_fantasy_points"] == pytest.approx(3.0)


def test_compute_recent_form_surfaces_the_requested_weeks_actual_stat_line():
    """The asked-about week's own real stat line, needed to answer a "why did
    he do well in week 3" question with that game's actual numbers instead of
    a blended average.
    """
    rows = pl.DataFrame({
        "season": [2026, 2026, 2026], "week": [1, 2, 3],
        "receptions": [3, 5, 9], "rec_yards": [30, 60, 145], "rec_tds": [0, 0, 1],
    })
    form = compute_recent_form(rows, season=2026, week=3, rules=PRESET_PPR)

    assert form["requested_week_played"] is True
    # 9 receptions*1.0 + 145 rec_yards*0.1 + 1 rec_td*6 = 9 + 14.5 + 6 = 29.5
    assert form["requested_week_fantasy_points"] == pytest.approx(29.5)
    assert form["requested_week_stats"]["receptions"] == 9
    assert form["requested_week_stats"]["rec_yards"] == 145
    assert form["requested_week_stats"]["rec_tds"] == 1


def test_compute_recent_form_requested_week_not_played_has_no_stat_line():
    """The normal case (asking about the upcoming, not-yet-played week) must
    not fabricate a stat line - there isn't one yet.
    """
    rows = _rows(seasons=[2026, 2026], weeks=[1, 2], rec_yards=[50, 80])
    form = compute_recent_form(rows, season=2026, week=5, rules=PRESET_PPR)

    assert form["requested_week_played"] is False
    assert form["requested_week_fantasy_points"] is None
    assert form["requested_week_stats"] is None


def test_compute_recent_form_flags_a_bye_week_as_historical_when_later_games_exist():
    """A bye week (or any week this player didn't suit up for) is still a
    historical question once later games prove time has moved past it -
    `requested_week_played` alone can't signal this, since it's False for
    both "hasn't happened yet" and "happened, but this player sat out".
    """
    rows = _rows(seasons=[2026] * 4, weeks=[1, 2, 4, 5], rec_yards=[10, 20, 40, 50])
    form = compute_recent_form(rows, season=2026, week=3, rules=PRESET_PPR)

    assert form["requested_week_played"] is False
    assert form["query_is_historical"] is True


def test_compute_recent_form_query_is_not_historical_for_the_upcoming_week():
    """The normal case - asking about the upcoming, not-yet-played week -
    must not be flagged historical: nothing has happened after it yet.
    """
    rows = _rows(seasons=[2026, 2026], weeks=[1, 2], rec_yards=[50, 80])
    form = compute_recent_form(rows, season=2026, week=5, rules=PRESET_PPR)

    assert form["query_is_historical"] is False


def test_compute_recent_form_query_is_historical_when_the_week_was_played():
    """A played past week (later games exist in the same season) is historical
    too, not just a bye week - both share the same live-data caveat.
    """
    rows = _rows(seasons=[2026] * 5, weeks=[1, 2, 3, 4, 5], rec_yards=[10, 20, 30, 40, 50])
    form = compute_recent_form(rows, season=2026, week=3, rules=PRESET_PPR)

    assert form["requested_week_played"] is True
    assert form["query_is_historical"] is True
