import polars as pl

from scripts.refresh_stats import build_processed_injuries, build_processed_player_stats


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
