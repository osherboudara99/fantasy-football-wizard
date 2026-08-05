from datetime import date

import polars as pl
import pytest

from scripts.refresh_stats import (
    build_processed_injuries,
    build_processed_player_stats,
    build_staged_injuries,
    resolve_target_week,
    stats_seasons,
)


def _fake_schedules():
    """Two seasons of REG games as of "today" = 2026-07-18: 2025 fully played
    through week 18, 2026 not yet started.
    """
    return pl.DataFrame({
        "season": [2025, 2025, 2026, 2026],
        "week": [17, 18, 1, 2],
        "game_type": ["REG", "REG", "REG", "REG"],
        "gameday": ["2026-01-04", "2026-01-11", "2026-09-10", "2026-09-17"],
    })


def _partial_week_schedule():
    """2026 week 1 has played its Thursday game as of "today" but not yet its Sunday game."""
    return pl.DataFrame({
        "season": [2026, 2026],
        "week": [1, 1],
        "game_type": ["REG", "REG"],
        "gameday": ["2026-07-16", "2026-07-20"],
    })


class _FakeDate(date):
    """Stand-in for datetime.date with a fixed today() so tests are deterministic."""

    @classmethod
    def today(cls):
        return date(2026, 7, 18)


def test_resolve_target_week_returns_explicit_pair_unchanged(monkeypatch):
    """Both season and week given: no schedule lookup needed, returned as-is."""
    monkeypatch.setattr("scripts.refresh_stats.nfl.load_schedules", lambda **_: (_ for _ in ()).throw(
        AssertionError("should not fetch schedules when season and week are both given")
    ))
    assert resolve_target_week(2025, 5) == (2025, 5)


def test_resolve_target_week_defaults_to_the_next_unplayed_week(monkeypatch):
    """2025 is finished as of "today", so the decision to make is about 2026 week 1."""
    monkeypatch.setattr("scripts.refresh_stats.nfl.load_schedules", lambda **_: _fake_schedules())
    monkeypatch.setattr("scripts.refresh_stats.date", _FakeDate)
    assert resolve_target_week(None, None) == (2026, 1)


def test_resolve_target_week_stays_on_a_week_still_in_progress(monkeypatch):
    """Mid-week (Thursday played, Sunday not) the target is still that same week."""
    monkeypatch.setattr("scripts.refresh_stats.nfl.load_schedules", lambda **_: _partial_week_schedule())
    monkeypatch.setattr("scripts.refresh_stats.date", _FakeDate)
    assert resolve_target_week(None, None) == (2026, 1)


def test_resolve_target_week_raises_when_the_requested_season_is_over(monkeypatch):
    """--season 2025 alone has no unplayed weeks left; that's an error, not 2026's week 1."""
    monkeypatch.setattr("scripts.refresh_stats.nfl.load_schedules", lambda **_: _fake_schedules())
    monkeypatch.setattr("scripts.refresh_stats.date", _FakeDate)
    with pytest.raises(RuntimeError):
        resolve_target_week(2025, None)


def test_stats_seasons_reaches_back_only_early_in_the_season():
    assert stats_seasons(2026, 1) == [2025, 2026]
    assert stats_seasons(2026, 3) == [2025, 2026]
    assert stats_seasons(2026, 4) == [2026]


def _weekly_staged(seasons, weeks, points, season_types=None):
    n = len(weeks)
    return pl.DataFrame({
        "player_id": ["00-1"] * n,
        "player_name": ["Test Player"] * n,
        "position": ["WR"] * n,
        "team": ["MIN"] * n,
        "season": seasons,
        "week": weeks,
        "season_type": season_types or ["REG"] * n,
        "fantasy_points": points,
        "fantasy_points_ppr": [p + 3.0 for p in points],
        "snap_percentage": [0.7] * n,
        "targets": [5] * n,
        "carries": [0] * n,
        "xfp": [10.0] * n,
    })


def test_build_processed_player_stats_excludes_the_target_week():
    """Week 18 is the decision, so form covers 15-17 - not the game being projected."""
    staged = _weekly_staged(
        seasons=[2025] * 4, weeks=[15, 16, 17, 18],
        points=[5.0, 10.0, 15.0, 20.0],
    )

    result = build_processed_player_stats(staged, season=2025, week=18)
    row = result.row(0, named=True)

    # last 3 completed weeks = 15, 16, 17
    assert row["avg_fantasy_points_last3"] == 10.0
    # season to date = weeks 15-17, week 18 hasn't happened
    assert row["avg_fantasy_points_season"] == 10.0
    assert row["fantasy_points_last_week"] == 15.0


def test_build_processed_player_stats_carries_form_over_into_a_new_season():
    """Week 1 has no completed weeks of its own - recent form comes from last season."""
    staged = _weekly_staged(
        seasons=[2025, 2025, 2025, 2026], weeks=[16, 17, 18, 1],
        points=[6.0, 12.0, 18.0, 99.0],
    )

    result = build_processed_player_stats(staged, season=2026, week=1)
    row = result.row(0, named=True)

    assert row["avg_fantasy_points_last3"] == 12.0
    assert row["fantasy_points_last_week"] == 18.0
    # season-to-date is the *target* season, which hasn't started
    assert row["avg_fantasy_points_season"] is None
    assert row["season"] == 2026 and row["week"] == 1


def test_build_processed_player_stats_ignores_postseason_weeks():
    """POST week 19 outranks REG week 18 numerically - it must not count as recent form."""
    staged = _weekly_staged(
        seasons=[2025] * 3, weeks=[17, 18, 19],
        points=[10.0, 10.0, 40.0],
        season_types=["REG", "REG", "POST"],
    )

    result = build_processed_player_stats(staged, season=2026, week=1)
    row = result.row(0, named=True)

    assert row["avg_fantasy_points_last3"] == 10.0
    assert row["fantasy_points_last_week"] == 10.0


def test_build_staged_injuries_matches_whitespace_padded_sleeper_ids():
    """Sleeper pads ~20% of its gsis_ids; unstripped they split one player into two
    rows - an official "Out" report plus a Sleeper row that later defaults to Healthy.
    """
    raw = {
        "injuries": pl.DataFrame({
            "gsis_id": ["00-1"],
            "season": [2025],
            "week": [18],
            "team": ["MIN"],
            "position": ["TE"],
            "full_name": ["Padded Id Player"],
            "report_primary_injury": ["Shoulder"],
            "report_status": ["Out"],
            "practice_status": ["Did Not Participate In Practice"],
            "date_modified": ["2026-01-09"],
        }),
        "sleeper_players": pl.DataFrame({
            "gsis_id": [" 00-1"],
            "full_name": ["Padded Id Player"],
            "position": ["TE"],
            "team": ["MIN"],
            "injury_status": [None],
            "injury_body_part": [None],
            "injury_notes": [None],
            "injury_start_date": [None],
            "practice_participation": [None],
        }),
    }

    staged = build_staged_injuries(raw, season=2025, week=18)

    assert staged.height == 1
    row = staged.row(0, named=True)
    assert row["player_id"] == "00-1"
    assert row["report_status"] == "Out"
    assert build_processed_injuries(staged).row(0, named=True)["status"] == "Out"


def test_build_processed_injuries_overlays_sleeper_and_defaults_healthy():
    """Sleeper-only rows should fill in name/position/team and default to a Healthy status."""
    staged = pl.DataFrame({
        "player_id": ["00-1", "00-2"],
        "player_name": ["Official Report Name", None],
        "full_name": [None, "Sleeper Only Name"],
        "position": ["WR", None],
        "position_sleeper": [None, "RB"],
        "team": ["MIN", None],
        "team_sleeper": [None, "DAL"],
        "injury_status": ["Questionable", None],
        "report_status": [None, None],
        "practice_status": ["Limited Participation", None],
        "practice_participation": [None, None],
        "date_modified": ["2026-09-01", None],
        "injury_start_date": [None, None],
        "injury_notes": ["Hamstring tightness", None],
        "report_primary_injury": [None, None],
    })

    result = build_processed_injuries(staged).sort("player_id")
    healthy_row = result.row(1, named=True)
    reported_row = result.row(0, named=True)

    assert reported_row["player_name"] == "Official Report Name"
    assert reported_row["status"] == "Questionable"
    assert reported_row["notes"] == "Hamstring tightness"

    assert healthy_row["player_name"] == "Sleeper Only Name"
    assert healthy_row["position"] == "RB"
    assert healthy_row["team"] == "DAL"
    assert healthy_row["status"] == "Healthy"
