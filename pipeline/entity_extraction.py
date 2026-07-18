"""Regex-based entity extraction: pull player names and a week number out of a
free-text question (README §5).

Player names are matched against the known universe of names in
data/processed/player_stats.parquet via whole-name, case-insensitive matching -
not fuzzy string matching. The canonical name list already comes from
nflreadpy/Sleeper in Phase 1, so there's no need to guess at a name.
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

WEEK_PATTERN = re.compile(r"\bweek\s*(\d{1,2})\b", re.IGNORECASE)


def known_player_names() -> list[str]:
    """All distinct, non-null player names in the current processed player_stats table.

    A handful of rows come through with a null player_name (e.g. an unmapped
    team-level row) - those can't be matched against free text, so drop them here.
    """
    df = pl.read_parquet(PROCESSED_DIR / "player_stats.parquet")
    return df.select("player_name").drop_nulls().unique(maintain_order=True).to_series().to_list()


def extract_week(text: str) -> int | None:
    """Pull a week number like "week 5" or "Week12" out of free text."""
    match = WEEK_PATTERN.search(text)
    return int(match.group(1)) if match else None


def extract_players(text: str, known_names: list[str] | None = None) -> list[str]:
    """Find known player names mentioned in free text, in the order they appear.

    Longer names are checked first isn't needed here since each name is matched
    independently against the full text and results are then sorted by the
    position of their first match - this naturally handles one name being a
    substring of another (e.g. "Love" inside "Loveland") because only whole-word
    boundary matches count.

    Uses (?<!\\w)/(?!\\w) lookarounds instead of \\b: a plain \\b fails to match
    right after a name ending in punctuation (e.g. "Marvin Harrison Jr.") because
    neither the period nor the following space/end-of-string is a word character,
    so no word boundary exists there even though it's a valid match.
    """
    names = known_names if known_names is not None else known_player_names()
    matches: list[tuple[int, str]] = []
    for name in names:
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        found = pattern.search(text)
        if found:
            matches.append((found.start(), name))
    matches.sort(key=lambda pair: pair[0])
    return [name for _, name in matches]
