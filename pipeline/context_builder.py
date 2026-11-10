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


# (field, singular label, plural label, is a countable stat vs. a yardage
# total - yardage fields carry no plural label, they're never pluralized).
# Explicit plural forms rather than label+"s": "carry"/"fumble lost" don't
# pluralize regularly ("carrys", "fumble losts"). Display order groups pass/
# rush/rec together the way a real stat line reads, rather than following
# ScoringRules.model_fields' definition order.
_STAT_LABELS = [
    ("pass_yards", "pass yds", None, False),
    ("pass_tds", "pass TD", "pass TDs", True),
    ("pass_interceptions", "INT", "INTs", True),
    ("pass_2pt", "pass 2pt conversion", "pass 2pt conversions", True),
    ("rush_attempts", "carry", "carries", True),
    ("rush_yards", "rush yds", None, False),
    ("rush_tds", "rush TD", "rush TDs", True),
    ("rush_2pt", "rush 2pt conversion", "rush 2pt conversions", True),
    ("receptions", "reception", "receptions", True),
    ("rec_yards", "rec yds", None, False),
    ("rec_tds", "rec TD", "rec TDs", True),
    ("rec_2pt", "rec 2pt conversion", "rec 2pt conversions", True),
    ("fumbles_lost", "fumble lost", "fumbles lost", True),
]


def _format_stat_breakdown(stats: dict) -> str:
    parts = []
    for field, singular, plural, countable in _STAT_LABELS:
        value = stats.get(field)
        if not value:
            continue
        if countable:
            n = int(value)
            parts.append(f"{n} {singular if n == 1 else plural}")
        else:
            parts.append(f"{value:g} {singular}")
    return ", ".join(parts) if parts else "no recorded stats"


def _format_requested_week(week: int, form: dict, is_historical: bool | None) -> list[str]:
    """The asked-about week's own real stat line (when the player suited up),
    plus a live-data disclaimer whenever the query itself is historical.

    `is_historical` is the caller's authoritative answer (the real target
    season/week the processed data describes, compared against what was
    asked) when it has one; `None` means the caller doesn't know (e.g. a
    unit test calling build_context directly), so this falls back to
    `form["query_is_historical"]` - a per-player heuristic (does a later row
    exist for this player) that's usually right but misses one real case: a
    question about a player's own most-recently-played week, which has no
    later row for that player even though the actual target week has since
    moved on. The caller-provided signal takes precedence exactly to cover
    that gap.

    Not gated on `requested_week_played`: a bye week, an injury, or any week
    the player sat out is still historical once time has moved past it, even
    though there's no stat line for that specific week to show. The normal
    "ask about the upcoming week" case has neither line, since nothing has
    happened after it yet.

    The injuries table keeps no season/week at all (scripts/refresh_stats.py's
    build_processed_injuries drops both), so the "Injury:" line elsewhere in
    this block is always today's live status, never that week's - the note
    says so explicitly rather than letting it read as period-accurate.
    """
    lines = []
    if form["requested_week_played"]:
        breakdown = _format_stat_breakdown(form["requested_week_stats"])
        lines.append(
            f"- Week {week} actual: {form['requested_week_fantasy_points']:.1f} pts ({breakdown})"
        )
    historical = form["query_is_historical"] if is_historical is None else is_historical
    if not historical:
        return lines
    if form["requested_week_played"]:
        lines.append(
            f"- Note: only the real stat line above reflects week {week} itself - "
            f"there's no historical projection for that week, and the injury status "
            f"and any news shown below are today's, not from back then."
        )
    else:
        lines.append(
            f"- Note: no recorded stat line for week {week} for this player (bye, "
            f"injury, or otherwise didn't play) - there's no historical projection "
            f"for it either, and the injury status and any news shown below are "
            f"today's, not from back then."
        )
    return lines


def _format_future_note(week: int, is_future: bool | None) -> list[str]:
    """A note for a question about a week beyond what the processed data
    currently covers. Only the target week ever gets a real Sleeper
    projection fetched (scripts/refresh_stats.py), so a week further out has
    no prepared projection to show - `_format_player`'s projection lookup
    already omits it silently, but that alone reads as an ordinary
    "not projected" player rather than "this week isn't covered yet". This
    note makes the distinction explicit.

    `is_future` is the caller's authoritative signal (comparing the resolved
    season/week against the real target); unlike `_format_requested_week`,
    there's no per-player fallback to compute here (a future week has no
    rows for anyone), so `None` (target unavailable) simply shows nothing.
    """
    if not is_future:
        return []
    return [
        f"- Note: week {week} hasn't been prepared yet - there's no "
        f"projection for it (only the upcoming target week gets one), so "
        f"the numbers above reflect current form only, not a "
        f"week-{week}-specific prediction."
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
    if form["last_game_week"] is not None:
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
    is_historical: bool | None = None,
    is_future: bool | None = None,
) -> str:
    """One player's block: name header, recent form, projection, injury status, news."""
    player_rows = tables["player_stats"].filter(pl.col("player_name") == name)
    if player_rows.height == 0:
        raise PlayerNotFoundError(f'No stats found for "{name}"')
    player_id = player_rows.row(0, named=True).get("player_id")
    player_rows = _match_player(player_rows, name, player_id)
    form = compute_recent_form(player_rows, season, week, scoring_rules)

    proj = _match_player(tables["projections"], name, player_id).filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    )
    proj_row = proj.row(0, named=True) if proj.height else None
    has_projection = proj_row is not None and any(
        proj_row.get(field) is not None for field in ScoringRules.model_fields
    )
    projected = compute_fantasy_points(proj_row, scoring_rules) if has_projection else None

    lines = [
        f"{name}:", *_format_recent_form(form), *_format_requested_week(week, form, is_historical),
        *_format_future_note(week, is_future),
    ]
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
    is_historical: bool | None = None,
    is_future: bool | None = None,
) -> str:
    """Build the §7 PLAYER COMPARISON block for the given players/season/week.

    `season` distinguishes this-season from prior-season rows in the
    processed player_stats table, which now holds many historical per-game
    rows instead of one row per player for a single target week (docs/
    superpowers/specs/2026-09-19-custom-league-scoring-design.md).
    `scoring_rules` defaults to full PPR when omitted, matching this app's
    long-standing default. `is_historical` is the caller's authoritative
    answer (comparing the resolved season/week against the real target the
    processed data describes) for whether this query is about the past;
    `None` (the default) lets each player's block fall back to its own
    per-player heuristic - see `_format_requested_week`. `is_future` is the
    symmetric signal for a week beyond the target - see `_format_future_note`.
    """
    tables = tables if tables is not None else _load_processed()
    scoring_rules = scoring_rules if scoring_rules is not None else PRESET_PPR
    blocks = [
        _format_player(name, season, week, tables, news_fn, scoring_rules, is_historical, is_future)
        for name in players
    ]
    return "PLAYER COMPARISON\n\n" + "\n\n".join(blocks)
