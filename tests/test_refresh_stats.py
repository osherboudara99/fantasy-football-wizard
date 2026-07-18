from datetime import date

import polars as pl
import pytest

from scripts.refresh_stats import (
    build_processed_injuries,
    build_processed_player_stats,
    resolve_season_week,
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


def test_resolve_season_week_returns_explicit_pair_unchanged(monkeypatch):
    """Both season and week given: no schedule lookup needed, returned as-is."""
    monkeypatch.setattr("scripts.refresh_stats.nfl.load_schedules", lambda **_: (_ for _ in ()).throw(
        AssertionError("should not fetch schedules when season and week are both given")
    ))
    assert resolve_season_week(2025, 5) == (2025, 5)


def test_resolve_season_week_defaults_to_latest_completed_week_overall(monkeypatch):
    """No season/week given: falls back to the latest fully completed week across seasons."""
    monkeypatch.setattr("scripts.refresh_stats.nfl.load_schedules", lambda **_: _fake_schedules())
    monkeypatch.setattr("scripts.refresh_stats.date", _FakeDate)
    assert resolve_season_week(None, None) == (2025, 18)


def test_resolve_season_week_requires_played_games_within_requested_season(monkeypatch):
    """--season 2026 alone must not borrow 2025's latest week; 2026 has no played games yet."""
    monkeypatch.setattr("scripts.refresh_stats.nfl.load_schedules", lambda **_: _fake_schedules())
    monkeypatch.setattr("scripts.refresh_stats.date", _FakeDate)
    with pytest.raises(RuntimeError):
        resolve_season_week(2026, None)


def test_resolve_season_week_skips_a_week_still_in_progress(monkeypatch):
    """A week with a game that hasn't happened yet must not be picked as complete."""
    monkeypatch.setattr("scripts.refresh_stats.nfl.load_schedules", lambda **_: _partial_week_schedule())
    monkeypatch.setattr("scripts.refresh_stats.date", _FakeDate)
    with pytest.raises(RuntimeError):
        resolve_season_week(2026, None)


def test_build_processed_player_stats_aggregates_last3_and_season():
    """A 4-week staged history should yield a last-3-week avg, season avg, and last-week snapshot."""
    staged = pl.DataFrame({
        "player_id": ["00-1", "00-1", "00-1", "00-1"],
        "player_name": ["Test Player"] * 4,
        "position": ["WR"] * 4,
        "team": ["MIN"] * 4,
        "season": [2025] * 4,
        "week": [15, 16, 17, 18],
        "fantasy_points": [5.0, 10.0, 15.0, 20.0],
        "fantasy_points_ppr": [8.0, 13.0, 18.0, 23.0],
        "snap_percentage": [0.5, 0.6, 0.7, 0.8],
        "targets": [4, 5, 6, 7],
        "carries": [0, 0, 0, 0],
        "xfp": [6.0, 11.0, 16.0, 21.0],
    })

    result = build_processed_player_stats(staged, season=2025, week=18)
    row = result.row(0, named=True)

    # last 3 weeks = 16, 17, 18 (week > 18 - 3)
    assert row["avg_fantasy_points_last3"] == 15.0
    # season to date = all 4 weeks
    assert row["avg_fantasy_points_season"] == 12.5
    assert row["fantasy_points_last_week"] == 20.0


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
