"""Fantasy-point scoring rules and the formula that applies them to raw stat
counts, at request time (docs/superpowers/specs/2026-09-19-custom-league-
scoring-design.md). One formula for all four tiers - PPR/Half-PPR/Standard
are just specific weight values, Custom is the same math with different
weights - so there is exactly one code path, never "provider's number for
presets, computed for Custom".
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ScoringRules(BaseModel):
    """Points per unit of each raw stat category. Field names are canonical:
    both nflverse's historical column names and Sleeper's projection keys are
    renamed to these at the staging step (scripts/refresh_stats.py), so
    compute_fantasy_points never needs to know which provider a row came from.

    Known, documented deltas against nflverse's own fantasy_points/
    fantasy_points_ppr columns (99.67% agreement within 0.11 pts across
    21,096 real rows, per tests/test_scoring_consistency.py):
    1. `special_teams_tds` (kick/punt-return touchdowns) has no ScoringRules
       field, so a return-TD game scores 6 points lower here than nflverse's
       number. Not in scope to add - a separate future field expansion.
    2. Canonical `fumbles_lost` maps to nflverse's `fumbles_lost_total`, which
       includes special-teams/return fumbles, while nflverse's own formula
       only penalizes offensive (sack/rush/receiving) fumbles lost - a small,
       arguably-more-correct-for-real-leagues divergence, left as-is.
    """

    pass_yards: float = Field(default=0.0, ge=-10, le=10)
    pass_tds: float = Field(default=0.0, ge=-10, le=10)
    pass_interceptions: float = Field(default=0.0, ge=-10, le=10)
    pass_2pt: float = Field(default=0.0, ge=-10, le=10)
    rush_yards: float = Field(default=0.0, ge=-10, le=10)
    rush_tds: float = Field(default=0.0, ge=-10, le=10)
    rush_2pt: float = Field(default=0.0, ge=-10, le=10)
    rush_attempts: float = Field(default=0.0, ge=-10, le=10)
    receptions: float = Field(default=0.0, ge=-10, le=10)
    rec_yards: float = Field(default=0.0, ge=-10, le=10)
    rec_tds: float = Field(default=0.0, ge=-10, le=10)
    rec_2pt: float = Field(default=0.0, ge=-10, le=10)
    fumbles_lost: float = Field(default=0.0, ge=-10, le=10)


PRESET_STANDARD = ScoringRules(
    pass_yards=0.04, pass_tds=4, pass_interceptions=-2, pass_2pt=2,
    rush_yards=0.1, rush_tds=6, rush_2pt=2,
    rec_yards=0.1, rec_tds=6, rec_2pt=2,
    fumbles_lost=-2,
)
PRESET_HALF_PPR = PRESET_STANDARD.model_copy(update={"receptions": 0.5})
PRESET_PPR = PRESET_STANDARD.model_copy(update={"receptions": 1.0})


def compute_fantasy_points(row: dict, rules: ScoringRules) -> float:
    """Weighted sum of row[field] * rules.field over every ScoringRules field.

    A field missing from `row` (e.g. a QB row has no `receptions` key) or
    `None` contributes 0 rather than raising - real per-game rows only carry
    the categories that apply to that position.
    """
    total = 0.0
    for field in ScoringRules.model_fields:
        value = row.get(field)
        if value is not None:
            total += value * getattr(rules, field)
    return total
