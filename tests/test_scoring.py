import pytest

from pipeline.scoring import (
    PRESET_HALF_PPR,
    PRESET_PPR,
    PRESET_STANDARD,
    ScoringRules,
    compute_fantasy_points,
)


def test_presets_differ_only_in_reception_value():
    """PPR/Half-PPR/Standard are the same formula - only the reception weight moves."""
    assert PRESET_STANDARD.receptions == 0.0
    assert PRESET_HALF_PPR.receptions == 0.5
    assert PRESET_PPR.receptions == 1.0
    for field in ScoringRules.model_fields:
        if field == "receptions":
            continue
        assert getattr(PRESET_STANDARD, field) == getattr(PRESET_HALF_PPR, field) == getattr(PRESET_PPR, field)


def test_compute_fantasy_points_matches_the_standard_formula():
    """250 pass yds (0.04/yd) + 2 pass TD (4 ea) - 1 INT (-2) + 20 rush yds (0.1/yd)."""
    row = {
        "pass_yards": 250, "pass_tds": 2, "pass_interceptions": 1,
        "rush_yards": 20, "rush_tds": 0,
    }
    assert compute_fantasy_points(row, PRESET_STANDARD) == pytest.approx(18.0)


def test_compute_fantasy_points_reception_weight_is_the_ppr_knob():
    """A WR line: 6 receptions, 80 rec yards, 1 rec TD - only the reception credit changes per tier."""
    row = {"receptions": 6, "rec_yards": 80, "rec_tds": 1}
    assert compute_fantasy_points(row, PRESET_STANDARD) == pytest.approx(14.0)
    assert compute_fantasy_points(row, PRESET_HALF_PPR) == pytest.approx(17.0)
    assert compute_fantasy_points(row, PRESET_PPR) == pytest.approx(20.0)


def test_compute_fantasy_points_treats_missing_fields_as_zero():
    """A QB row has no receiving keys at all - that must contribute 0, not raise."""
    row = {"pass_yards": 300, "pass_tds": 3}
    assert compute_fantasy_points(row, PRESET_STANDARD) == pytest.approx(24.0)


def test_compute_fantasy_points_custom_weight_for_a_non_standard_category():
    """rush_attempts is 0 in every built-in preset but Custom mode can set it -
    e.g. the user's own league scores 1 point per carry ("handoff")."""
    custom = PRESET_PPR.model_copy(update={"rush_attempts": 1.0})
    row = {"rush_attempts": 15, "rush_yards": 60, "rush_tds": 1}
    assert compute_fantasy_points(row, custom) == pytest.approx(15 * 1.0 + 60 * 0.1 + 6)
