from datetime import date

import polars as pl
import pytest

from scripts.refresh_stats import (
    build_processed,
    build_processed_injuries,
    build_processed_player_stats,
    build_processed_projections,
    build_staged_injuries,
    build_staged_player_stats,
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


def test_stats_seasons_always_includes_the_prior_season():
    """Prior-season data is a per-player fallback for thin current-season samples
    (injury, suspension, late call-up) that can happen at any week, not just
    early in the season - so it must always be fetched, never dropped mid-season.
    """
    assert stats_seasons(2026) == [2025, 2026]


def _weekly_staged(seasons, weeks, rec_yards, season_types=None):
    n = len(weeks)
    return pl.DataFrame({
        "player_id": ["00-1"] * n,
        "player_name": ["Test Player"] * n,
        "position": ["WR"] * n,
        "team": ["MIN"] * n,
        "season": seasons,
        "week": weeks,
        "season_type": season_types or ["REG"] * n,
        "pass_yards": [0] * n, "pass_tds": [0] * n, "pass_interceptions": [0] * n,
        "pass_2pt": [0] * n, "rush_yards": [0] * n, "rush_tds": [0] * n,
        "rush_2pt": [0] * n, "rush_attempts": [0] * n,
        "receptions": [5] * n, "rec_yards": rec_yards, "rec_tds": [0] * n,
        "rec_2pt": [0] * n, "fumbles_lost": [0] * n,
    })


def test_build_processed_player_stats_returns_one_row_per_completed_game():
    """Week 18 is the decision - only weeks strictly before it may appear."""
    staged = _weekly_staged(
        seasons=[2025] * 4, weeks=[15, 16, 17, 18], rec_yards=[50, 100, 150, 999],
    )

    result = build_processed_player_stats(staged, season=2025, week=18, current_player_ids=["00-1"])

    assert result.height == 3
    assert sorted(result["week"].to_list()) == [15, 16, 17]
    assert result.row(0, named=True)["rec_yards"] in (50, 100, 150)


def test_build_processed_player_stats_excludes_players_with_no_current_relevance():
    """Always fetching the prior season (stats_seasons) must not resurrect retirees
    or unsigned free agents - only players with a current-season game or a spot in
    this week's roster/projection universe (current_player_ids) belong in the table.
    """
    staged = _weekly_staged(seasons=[2025, 2025, 2025], weeks=[16, 17, 18], rec_yards=[50, 100, 150])

    result = build_processed_player_stats(staged, season=2026, week=1, current_player_ids=[])

    assert result.height == 0


def test_build_processed_player_stats_ignores_postseason_weeks():
    """POST week 19 outranks REG week 18 numerically - it must not appear as a completed game."""
    staged = _weekly_staged(
        seasons=[2025] * 3, weeks=[17, 18, 19], rec_yards=[50, 50, 999],
        season_types=["REG", "REG", "POST"],
    )

    result = build_processed_player_stats(staged, season=2026, week=1, current_player_ids=["00-1"])

    assert result.height == 2
    assert 19 not in result["week"].to_list()


def test_build_staged_player_stats_selects_canonical_scoring_columns():
    """The 13 ScoringRules-named columns must exist, renamed from nflverse's names."""
    raw = {
        "player_stats": pl.DataFrame({
            "player_id": ["00-1"], "player_display_name": ["Test Player"],
            "position": ["WR"], "team": ["MIN"], "opponent_team": ["GB"],
            "season": [2026], "week": [1], "season_type": ["REG"],
            "targets": [8], "receptions": [6], "carries": [1],
            "rushing_epa": [0.1], "receiving_epa": [0.2], "passing_epa": [0.0],
            "target_share": [0.3], "air_yards_share": [0.2],
            "passing_yards": [0], "passing_tds": [0], "passing_interceptions": [0],
            "passing_2pt_conversions": [0],
            "rushing_yards": [5], "rushing_tds": [0], "rushing_2pt_conversions": [0],
            "receiving_yards": [80], "receiving_tds": [1], "receiving_2pt_conversions": [0],
            "fumbles_lost_total": [0],
        }),
        "snap_counts": pl.DataFrame(schema={"pfr_player_id": pl.String, "season": pl.Int64, "week": pl.Int64, "offense_pct": pl.Float64}),
        "ff_playerids": pl.DataFrame(schema={"pfr_id": pl.String, "gsis_id": pl.String}),
        "ff_opportunity": pl.DataFrame(schema={"player_id": pl.String, "season": pl.String, "week": pl.String, "total_fantasy_points_exp": pl.Float64}),
        "ngs_receiving": pl.DataFrame(schema={"player_gsis_id": pl.String, "season": pl.Int64, "week": pl.Int64, "avg_separation": pl.Float64, "avg_yac_above_expectation": pl.Float64}),
        "ngs_rushing": pl.DataFrame(schema={"player_gsis_id": pl.String, "season": pl.Int64, "week": pl.Int64, "rush_yards_over_expected_per_att": pl.Float64}),
        "ngs_passing": pl.DataFrame(schema={"player_gsis_id": pl.String, "season": pl.Int64, "week": pl.Int64, "completion_percentage_above_expectation": pl.Float64, "aggressiveness": pl.Float64}),
    }

    staged = build_staged_player_stats(raw)
    row = staged.row(0, named=True)

    assert row["rush_attempts"] == 1  # renamed from "carries"
    assert row["rec_yards"] == 80 and row["rec_tds"] == 1
    assert row["fumbles_lost"] == 0


def test_build_processed_writes_a_target_season_week_meta_file(tmp_path, monkeypatch):
    """player_stats.parquet no longer carries a single target season/week per row -
    a dedicated meta table is the only place that survives the refactor.
    """
    monkeypatch.setattr("scripts.refresh_stats.PROCESSED_DIR", tmp_path)
    # Empty-but-fully-typed stand-ins for staged injuries/projections: build_processed
    # calls build_processed_injuries/build_processed_projections, which .select() a
    # fixed set of columns via pl.coalesce - even at 0 rows, polars' select() raises
    # ColumnNotFoundError if a referenced column is absent from the schema entirely,
    # so a bare {"player_id": pl.String} frame (as a minimal literal fixture might
    # suggest) isn't enough here.
    empty_injuries_schema = {
        "player_id": pl.String, "player_name": pl.String, "full_name": pl.String,
        "position": pl.String, "position_sleeper": pl.String,
        "team": pl.String, "team_sleeper": pl.String,
        "injury_status": pl.String, "report_status": pl.String,
        "practice_status": pl.String, "practice_participation": pl.String,
        "date_modified": pl.String, "injury_start_date": pl.String,
        "injury_notes": pl.String, "report_primary_injury": pl.String,
    }
    empty_projections_schema = {
        "player_id": pl.String, "player_name": pl.String, "player_name_ecr": pl.String,
        "position": pl.String, "pos": pl.String,
        "team": pl.String, "team_ecr": pl.String,
        "season": pl.Int64, "week": pl.Int64,
        "pass_yards": pl.Float64, "pass_tds": pl.Float64,
        "pass_interceptions": pl.Float64, "pass_2pt": pl.Float64,
        "rush_yards": pl.Float64, "rush_tds": pl.Float64,
        "rush_2pt": pl.Float64, "rush_attempts": pl.Float64,
        "receptions": pl.Float64, "rec_yards": pl.Float64,
        "rec_tds": pl.Float64, "rec_2pt": pl.Float64,
        "fumbles_lost": pl.Float64,
        "ecr_rank": pl.Float64, "ecr_position_rank": pl.String,
    }
    staged = {
        "player_stats": _weekly_staged(seasons=[2026], weeks=[1], rec_yards=[10]),
        "injuries": pl.DataFrame(schema=empty_injuries_schema),
        "projections": pl.DataFrame(schema=empty_projections_schema),
    }

    build_processed(staged, season=2026, week=2)

    meta = pl.read_parquet(tmp_path / "meta.parquet")
    assert meta.row(0, named=True) == {"season": 2026, "week": 2}


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


def test_build_processed_projections_carries_raw_canonical_stat_columns():
    staged = pl.DataFrame({
        "player_id": ["00-1"], "player_name": ["Test Player"], "player_name_ecr": [None],
        "position": ["WR"], "pos": [None], "team": ["MIN"], "team_ecr": [None],
        "season": [2026], "week": [2],
        "pass_yards": [0.0], "pass_tds": [0.0], "pass_interceptions": [0.0], "pass_2pt": [0.0],
        "rush_yards": [2.0], "rush_tds": [0.0], "rush_2pt": [0.0], "rush_attempts": [0.5],
        "receptions": [4.4], "rec_yards": [57.0], "rec_tds": [0.4], "rec_2pt": [0.0],
        "fumbles_lost": [0.02],
        "ecr_rank": [18.0], "ecr_position_rank": ["WR9"],
    })

    result = build_processed_projections(staged)
    row = result.row(0, named=True)

    assert row["receptions"] == pytest.approx(4.4)
    assert row["rec_yards"] == pytest.approx(57.0)
    assert row["ecr_rank"] == pytest.approx(18.0)
    assert "projected_points" not in result.columns
