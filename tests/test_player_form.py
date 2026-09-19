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
    form = compute_recent_form(rows, season=2026, rules=PRESET_PPR)

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
    form = compute_recent_form(rows, season=2026, rules=PRESET_PPR)

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
    standard = compute_recent_form(rows, season=2026, rules=ScoringRules())
    ppr = compute_recent_form(rows, season=2026, rules=PRESET_PPR)

    assert standard["avg_fantasy_points_season"] == pytest.approx(0.0)
    assert ppr["avg_fantasy_points_season"] == pytest.approx(10.0)


def test_compute_recent_form_handles_a_rookie_with_no_prior_season_rows():
    rows = _rows(seasons=[2026], weeks=[1], rec_yards=[50])
    form = compute_recent_form(rows, season=2026, rules=PRESET_PPR)

    assert form["prior_season_games_played"] == 0
    assert form["prior_season_avg_fantasy_points"] is None
    assert form["prior_season_last3_avg_fantasy_points"] is None
