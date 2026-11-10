"""Consistency check against nflverse's own fantasy_points/fantasy_points_ppr
columns, kept in the raw/staged table for exactly this reason (docs/
superpowers/specs/2026-09-19-custom-league-scoring-design.md). This was
required by the original design spec but silently dropped from the
implementation plan - added back here per the final whole-branch review.

A small disagreement is expected and documented in pipeline/scoring.py:
special_teams_tds isn't a ScoringRules field, and canonical fumbles_lost maps
to nflverse's fumbles_lost_total (which includes special-teams fumbles,
unlike nflverse's own formula). Real data (2026-09-20 refresh, 21,096 rows)
shows 99.67% agreement within 0.11 pts for both presets.
"""
from pathlib import Path

import polars as pl
import pytest

from pipeline.scoring import PRESET_PPR, PRESET_STANDARD, compute_fantasy_points

RAW_PLAYER_STATS = Path(__file__).resolve().parent.parent / "data" / "raw" / "player_stats.parquet"

# Minimal rename of nflverse's native columns to the 13 canonical ScoringRules
# fields, mirroring scripts/refresh_stats.py's build_staged_player_stats (just
# the stat-column renames, none of the snap/xFP/NGS joins - those don't feed
# compute_fantasy_points).
_CANONICAL_RENAME = {
    "passing_yards": "pass_yards",
    "passing_tds": "pass_tds",
    "passing_interceptions": "pass_interceptions",
    "passing_2pt_conversions": "pass_2pt",
    "rushing_yards": "rush_yards",
    "rushing_tds": "rush_tds",
    "rushing_2pt_conversions": "rush_2pt",
    "carries": "rush_attempts",
    "receiving_yards": "rec_yards",
    "receiving_tds": "rec_tds",
    "receiving_2pt_conversions": "rec_2pt",
    "fumbles_lost_total": "fumbles_lost",
}

TOLERANCE = 0.5
MIN_AGREEMENT = 0.99


def _canonical_rows() -> pl.DataFrame:
    raw = pl.read_parquet(RAW_PLAYER_STATS)
    return raw.rename(_CANONICAL_RENAME)


def _agreement_fraction(rows: pl.DataFrame, rules, nflverse_column: str) -> float:
    within_tolerance = 0
    for row in rows.iter_rows(named=True):
        ours = compute_fantasy_points(row, rules)
        theirs = row[nflverse_column]
        if abs(ours - theirs) <= TOLERANCE:
            within_tolerance += 1
    return within_tolerance / rows.height


@pytest.mark.skipif(not RAW_PLAYER_STATS.exists(), reason="data/raw/player_stats.parquet not refreshed yet")
def test_compute_fantasy_points_matches_nflverse_standard_within_tolerance():
    rows = _canonical_rows()
    agreement = _agreement_fraction(rows, PRESET_STANDARD, "fantasy_points")
    assert agreement >= MIN_AGREEMENT, (
        f"only {agreement:.4%} of rows agreed with nflverse's fantasy_points within "
        f"{TOLERANCE} pts (expected >= {MIN_AGREEMENT:.0%})"
    )


@pytest.mark.skipif(not RAW_PLAYER_STATS.exists(), reason="data/raw/player_stats.parquet not refreshed yet")
def test_compute_fantasy_points_matches_nflverse_ppr_within_tolerance():
    rows = _canonical_rows()
    agreement = _agreement_fraction(rows, PRESET_PPR, "fantasy_points_ppr")
    assert agreement >= MIN_AGREEMENT, (
        f"only {agreement:.4%} of rows agreed with nflverse's fantasy_points_ppr within "
        f"{TOLERANCE} pts (expected >= {MIN_AGREEMENT:.0%})"
    )
