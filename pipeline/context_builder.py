"""Assemble the LLM-ready player comparison context from processed structured
data (README §7). Deliberately outside the LLM: deterministic, testable, and
debuggable - the LLM only ever reasons over the string this produces.

News retrieval (RAG) is wired in during Phase 6; until then the comparison
omits the "Recent news" bullet shown in README §7's example.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

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


def _format_injury(injuries: pl.DataFrame, player_name: str) -> str:
    """"Questionable -> Full practice Friday" style line, or "Healthy" if no report."""
    row = injuries.filter(pl.col("player_name") == player_name)
    if row.height == 0:
        return "Healthy"
    record = row.row(0, named=True)
    status, practice = record["status"], record["practice_level"]
    return f"{status} -> {practice}" if practice else status


def _format_player(name: str, week: int, tables: dict[str, pl.DataFrame]) -> str:
    """One player's block: name header, last-3-week avg, projection, injury status."""
    stats = tables["player_stats"].filter(
        (pl.col("player_name") == name) & (pl.col("week") == week)
    )
    if stats.height == 0:
        raise PlayerNotFoundError(f'No stats found for "{name}" in week {week}')
    avg_last3 = stats.row(0, named=True)["avg_fantasy_points_last3"]

    proj = tables["projections"].filter(
        (pl.col("player_name") == name) & (pl.col("week") == week)
    )
    projected = proj.row(0, named=True)["projected_points"] if proj.height else None

    lines = [f"{name}:", f"- Avg fantasy points (last 3 weeks): {avg_last3:.1f}"]
    if projected is not None:
        lines.append(f"- Projected points: {projected:.1f}")
    lines.append(f"- Injury: {_format_injury(tables['injuries'], name)}")
    return "\n".join(lines)


def build_context(
    players: list[str], week: int, tables: dict[str, pl.DataFrame] | None = None
) -> str:
    """Build the §7 PLAYER COMPARISON block for the given players and week.

    `tables` lets tests inject fixture DataFrames instead of reading data/processed/.
    """
    tables = tables if tables is not None else _load_processed()
    blocks = [_format_player(name, week, tables) for name in players]
    return "PLAYER COMPARISON\n\n" + "\n\n".join(blocks)
