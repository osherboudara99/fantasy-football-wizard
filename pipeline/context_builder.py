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


def _format_player(
    name: str,
    week: int,
    tables: dict[str, pl.DataFrame],
    news_fn: Callable[[str | None, str], list[NewsItem]] | None,
) -> str:
    """One player's block: name header, last-3-week avg, projection, injury status, news."""
    stats = tables["player_stats"].filter(
        (pl.col("player_name") == name) & (pl.col("week") == week)
    )
    if stats.height == 0:
        raise PlayerNotFoundError(f'No stats found for "{name}" in week {week}')
    stats_row = stats.row(0, named=True)
    avg_last3 = stats_row["avg_fantasy_points_last3"]
    player_id = stats_row.get("player_id")

    proj = _match_player(tables["projections"], name, player_id).filter(pl.col("week") == week)
    projected = proj.row(0, named=True)["projected_points"] if proj.height else None

    lines = [f"{name}:", f"- Avg fantasy points (last 3 weeks): {avg_last3:.1f}"]
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
