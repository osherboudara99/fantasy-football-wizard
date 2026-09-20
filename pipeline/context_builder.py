"""Assemble the LLM-ready player comparison context from processed structured
data (README §7). Deliberately outside the LLM: deterministic, testable, and
debuggable - the LLM only ever reasons over the string this produces.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import polars as pl

from pipeline.player_form import compute_recent_form
from pipeline.scoring import PRESET_PPR, ScoringRules, compute_fantasy_points
from retrieval.news_retriever import NewsItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


class PlayerNotFoundError(ValueError):
    """Raised when a requested player has no rows in the processed player_stats table."""


def _load_processed() -> dict[str, pl.DataFrame]:
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
    rows = _match_player(injuries, name, player_id)
    if rows.height == 0:
        return "Healthy"
    reported = rows.filter(pl.col("status") != "Healthy")
    record = (reported if reported.height else rows).row(0, named=True)
    status, practice = record["status"], record["practice_level"]
    return f"{status} -> {practice}" if practice else status


def _plural(n: int) -> str:
    return "" if n == 1 else "s"


# (field, display label, is a countable stat vs. a yardage total) - display
# order groups pass/rush/rec together the way a real stat line reads, rather
# than following ScoringRules.model_fields' definition order.
_STAT_LABELS = [
    ("pass_yards", "pass yds", False),
    ("pass_tds", "pass TD", True),
    ("pass_interceptions", "INT", True),
    ("pass_2pt", "pass 2pt conversion", True),
    ("rush_attempts", "carry", True),
    ("rush_yards", "rush yds", False),
    ("rush_tds", "rush TD", True),
    ("rush_2pt", "rush 2pt conversion", True),
    ("receptions", "reception", True),
    ("rec_yards", "rec yds", False),
    ("rec_tds", "rec TD", True),
    ("rec_2pt", "rec 2pt conversion", True),
    ("fumbles_lost", "fumble lost", True),
]


def _format_stat_breakdown(stats: dict) -> str:
    parts = []
    for field, label, countable in _STAT_LABELS:
        value = stats.get(field)
        if not value:
            continue
        if countable:
            n = int(value)
            parts.append(f"{n} {label}{_plural(n)}")
        else:
            parts.append(f"{value:g} {label}")
    return ", ".join(parts) if parts else "no recorded stats"


def _format_requested_week(week: int, form: dict) -> list[str]:
    """The asked-about week's own real stat line, plus an explicit note that
    the app has no historical projection or news data for it - only ever
    surfaced when that week was actually played (the normal "ask about the
    upcoming week" case has nothing here, since it hasn't happened yet).
    """
    if not form["requested_week_played"]:
        return []
    breakdown = _format_stat_breakdown(form["requested_week_stats"])
    return [
        f"- Week {week} actual: {form['requested_week_fantasy_points']:.1f} pts ({breakdown})",
        f"- Note: historical projections and news aren't retained past their week - "
        f"only the real stat line above is available for week {week}, not what was "
        f"projected beforehand or what was reported at the time.",
    ]


def _format_recent_form(form: dict) -> list[str]:
    """This-season form, never blended with last season's numbers.

    Last season only appears once there aren't yet 3 games of this-season data
    to judge by - once there are, it stops being relevant and is left out.
    """
    games_this_season = form["games_played_this_season"]
    if games_this_season == 0:
        lines = ["- This season: no games played yet"]
    elif games_this_season < 3:
        lines = [
            f"- This season ({games_this_season} game{_plural(games_this_season)}): "
            f"{form['avg_fantasy_points_season']:.1f} avg fantasy points"
        ]
    else:
        lines = [
            f"- This season ({games_this_season} games): "
            f"{form['avg_fantasy_points_season']:.1f} season avg, "
            f"{form['avg_fantasy_points_last3']:.1f} avg over last 3 games"
        ]
        return lines

    prior_games = form["prior_season_games_played"]
    if prior_games > 0:
        finish_n = min(3, prior_games)
        lines.append(
            f"- Last season ({prior_games} game{_plural(prior_games)}): "
            f"{form['prior_season_avg_fantasy_points']:.1f} season avg, "
            f"{form['prior_season_last3_avg_fantasy_points']:.1f} "
            f"avg over final {finish_n} game{_plural(finish_n)}"
        )
    lines.append(
        f"- Most recent game played (Week {form['last_game_week']}, "
        f"{form['last_game_season']}): {form['last_game_fantasy_points']:.1f} pts"
    )
    return lines


def _format_player(
    name: str,
    season: int,
    week: int,
    tables: dict[str, pl.DataFrame],
    news_fn: Callable[[str | None, str], list[NewsItem]] | None,
    scoring_rules: ScoringRules,
) -> str:
    """One player's block: name header, recent form, projection, injury status, news."""
    player_rows = tables["player_stats"].filter(pl.col("player_name") == name)
    if player_rows.height == 0:
        raise PlayerNotFoundError(f'No stats found for "{name}"')
    player_id = player_rows.row(0, named=True).get("player_id")
    player_rows = _match_player(player_rows, name, player_id)
    form = compute_recent_form(player_rows, season, week, scoring_rules)

    proj = _match_player(tables["projections"], name, player_id).filter(pl.col("week") == week)
    proj_row = proj.row(0, named=True) if proj.height else None
    has_projection = proj_row is not None and any(
        proj_row.get(field) is not None for field in ScoringRules.model_fields
    )
    projected = compute_fantasy_points(proj_row, scoring_rules) if has_projection else None

    lines = [f"{name}:", *_format_recent_form(form), *_format_requested_week(week, form)]
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
    season: int,
    week: int,
    tables: dict[str, pl.DataFrame] | None = None,
    news_fn: Callable[[str | None, str], list[NewsItem]] | None = None,
    scoring_rules: ScoringRules | None = None,
) -> str:
    """Build the §7 PLAYER COMPARISON block for the given players/season/week.

    `season` distinguishes this-season from prior-season rows in the
    processed player_stats table, which now holds many historical per-game
    rows instead of one row per player for a single target week (docs/
    superpowers/specs/2026-09-19-custom-league-scoring-design.md).
    `scoring_rules` defaults to full PPR when omitted, matching this app's
    long-standing default.
    """
    tables = tables if tables is not None else _load_processed()
    scoring_rules = scoring_rules if scoring_rules is not None else PRESET_PPR
    blocks = [
        _format_player(name, season, week, tables, news_fn, scoring_rules) for name in players
    ]
    return "PLAYER COMPARISON\n\n" + "\n\n".join(blocks)
