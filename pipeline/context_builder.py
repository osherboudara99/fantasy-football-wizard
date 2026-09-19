"""Assemble the LLM-ready player comparison context from processed structured
data (README §7). Deliberately outside the LLM: deterministic, testable, and
debuggable - the LLM only ever reasons over the string this produces.

The "Recent news" bullet (README §7's example) only appears when a `news_fn`
is passed in - callers that don't care about news (or are running before the
Phase 6 embeddings index exists) get the pre-Phase-6 output unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import polars as pl

from retrieval.news_retriever import NewsItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


class PlayerNotFoundError(ValueError):
    """Raised when a requested player has no row in the processed stats for the given week."""


def _load_processed() -> dict[str, pl.DataFrame]:
    """Load the three processed tables build_context needs, from data/processed/."""
    return {
        "player_stats": pl.read_parquet(PROCESSED_DIR / "player_stats.parquet"),
        "projections": pl.read_parquet(PROCESSED_DIR / "projections.parquet"),
        "injuries": pl.read_parquet(PROCESSED_DIR / "injuries.parquet"),
    }


def _match_player(df: pl.DataFrame, name: str, player_id: str | None) -> pl.DataFrame:
    """Rows for one player, keyed on player_id when both sides carry it.

    Names are not a join key: player_stats uses nflverse display names ("Kenneth
    Walker III") while projections come from Sleeper ("Kenneth Walker"), so
    name-matching silently drops every suffixed player. player_id (gsis_id) is on
    all three processed tables - fall back to the name only for fixture frames
    that don't carry it.
    """
    if player_id is not None and "player_id" in df.columns:
        return df.filter(pl.col("player_id") == player_id)
    return df.filter(pl.col("player_name") == name)


def _format_injury(injuries: pl.DataFrame, name: str, player_id: str | None) -> str:
    """"Questionable -> Full practice Friday" style line, or "Healthy" if no report."""
    rows = _match_player(injuries, name, player_id)
    if rows.height == 0:
        return "Healthy"
    # A real report always outranks the "Healthy" default when a player somehow
    # lands more than one row - never downgrade an injury by row ordering.
    reported = rows.filter(pl.col("status") != "Healthy")
    record = (reported if reported.height else rows).row(0, named=True)
    status, practice = record["status"], record["practice_level"]
    return f"{status} -> {practice}" if practice else status


def _plural(n: int) -> str:
    return "" if n == 1 else "s"


def _format_recent_form(stats_row: dict) -> list[str]:
    """This-season form, never blended with last season's numbers.

    Last season only appears once there aren't yet 3 games of this-season data
    to judge by - once there are, it stops being relevant and is left out.

    Uses the PPR aggregates throughout: `projected_points` (README §7) is
    defined from Sleeper's `pts_ppr`, so historical figures must be on the
    same scoring basis or a receiver's/pass-catching back's projection would
    silently be compared against a lower, non-PPR history.
    """
    games_this_season = stats_row["games_played_this_season"]
    if games_this_season == 0:
        lines = ["- This season: no games played yet"]
    elif games_this_season < 3:
        lines = [
            f"- This season ({games_this_season} game{_plural(games_this_season)}): "
            f"{stats_row['avg_fantasy_points_ppr_season']:.1f} avg fantasy points"
        ]
    else:
        lines = [
            f"- This season ({games_this_season} games): "
            f"{stats_row['avg_fantasy_points_ppr_season']:.1f} season avg, "
            f"{stats_row['avg_fantasy_points_ppr_last3']:.1f} avg over last 3 games"
        ]
        return lines

    prior_games = stats_row["prior_season_games_played"]
    if prior_games > 0:
        finish_n = min(3, prior_games)
        lines.append(
            f"- Last season ({prior_games} game{_plural(prior_games)}): "
            f"{stats_row['prior_season_avg_fantasy_points_ppr']:.1f} season avg, "
            f"{stats_row['prior_season_last3_avg_fantasy_points_ppr']:.1f} "
            f"avg over final {finish_n} game{_plural(finish_n)}"
        )
    lines.append(
        f"- Most recent game played (Week {stats_row['last_game_week']}, "
        f"{stats_row['last_game_season']}): {stats_row['last_game_fantasy_points_ppr']:.1f} pts"
    )
    return lines


def _format_player(
    name: str,
    week: int,
    tables: dict[str, pl.DataFrame],
    news_fn: Callable[[str | None, str], list[NewsItem]] | None,
) -> str:
    """One player's block: name header, recent form, projection, injury status, news."""
    stats = tables["player_stats"].filter(
        (pl.col("player_name") == name) & (pl.col("week") == week)
    )
    if stats.height == 0:
        raise PlayerNotFoundError(f'No stats found for "{name}" in week {week}')
    stats_row = stats.row(0, named=True)
    player_id = stats_row.get("player_id")

    proj = _match_player(tables["projections"], name, player_id).filter(pl.col("week") == week)
    projected = proj.row(0, named=True)["projected_points"] if proj.height else None

    lines = [f"{name}:", *_format_recent_form(stats_row)]
    if projected is not None:
        lines.append(f"- Projected points: {projected:.1f}")
    lines.append(f"- Injury: {_format_injury(tables['injuries'], name, player_id)}")

    if news_fn is not None:
        news_items = news_fn(player_id, name)
        if news_items:
            lines.append("- Recent news:")
            lines.extend(f'  - "{item.snippet}"' for item in news_items)
    return "\n".join(lines)


def build_context(
    players: list[str],
    week: int,
    tables: dict[str, pl.DataFrame] | None = None,
    news_fn: Callable[[str | None, str], list[NewsItem]] | None = None,
) -> str:
    """Build the §7 PLAYER COMPARISON block for the given players and week.

    `tables` lets tests inject fixture DataFrames instead of reading data/processed/.
    `news_fn(player_id, player_name) -> list[str]` adds a "Recent news" bullet per
    player when it returns any snippets; omitted (the default) or an empty return
    skips the bullet entirely.
    """
    tables = tables if tables is not None else _load_processed()
    blocks = [_format_player(name, week, tables, news_fn) for name in players]
    return "PLAYER COMPARISON\n\n" + "\n\n".join(blocks)
