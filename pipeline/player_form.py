"""Per-player recent-form aggregation over raw per-game rows, computed at
chat-request time under whichever ScoringRules the caller selected (docs/
superpowers/specs/2026-09-19-custom-league-scoring-design.md). Moved out of
scripts/refresh_stats.py, which used to pre-aggregate this at nightly-refresh
time under one fixed formula - that stopped working once scoring became a
per-request choice.
"""
from __future__ import annotations

import polars as pl

from pipeline.scoring import ScoringRules, compute_fantasy_points

LAST_N_WEEKS = 3


def _own_last_n(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """This player's own last `n` rows of `df`, most-recent first.

    Per-player, not a global top-`n`-weeks list: a player who missed the
    league's single most recent week (bye, injury) must still get their own
    last `n` played games, not a thinner or misaligned window.
    """
    return df.sort(["season", "week"], descending=True).head(n)


def _avg_points(rows: pl.DataFrame, rules: ScoringRules) -> float | None:
    if rows.height == 0:
        return None
    points = [compute_fantasy_points(row, rules) for row in rows.iter_rows(named=True)]
    return sum(points) / len(points)


def compute_recent_form(player_rows: pl.DataFrame, season: int, rules: ScoringRules) -> dict:
    """One player's recent-form summary, given all of their completed-game rows.

    `player_rows` is expected to already be filtered to completed games for
    this one player (scripts/refresh_stats.py's build_processed_player_stats
    guarantees both). "Last 3" never blends across the season boundary: this
    season's window only ever contains this season's own rows, even if that's
    0, 1, or 2 games - last season is surfaced separately, never averaged in.
    """
    this_season = player_rows.filter(pl.col("season") == season)
    prior_season = player_rows.filter(pl.col("season") == season - 1)

    this_season_last3 = _own_last_n(this_season, LAST_N_WEEKS)
    prior_season_last3 = _own_last_n(prior_season, LAST_N_WEEKS)
    most_recent_overall = _own_last_n(player_rows, 1)

    last_game = most_recent_overall.row(0, named=True) if most_recent_overall.height else None

    return {
        "games_played_this_season": this_season.height,
        "avg_fantasy_points_last3": _avg_points(this_season_last3, rules),
        "avg_fantasy_points_season": _avg_points(this_season, rules),
        "prior_season_games_played": prior_season.height,
        "prior_season_avg_fantasy_points": _avg_points(prior_season, rules),
        "prior_season_last3_avg_fantasy_points": _avg_points(prior_season_last3, rules),
        "last_game_season": last_game["season"] if last_game else None,
        "last_game_week": last_game["week"] if last_game else None,
        "last_game_fantasy_points": (
            compute_fantasy_points(last_game, rules) if last_game else None
        ),
    }
