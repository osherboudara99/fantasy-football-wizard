# Custom League Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the agent compute fantasy points under PPR / Half-PPR / Standard / Custom league scoring instead of hard-coded full PPR, with the choice stored client-side (no auth needed) and Custom rules parsed from a free-text description once, not per message.

**Architecture:** Move fantasy-point computation from nightly refresh time to chat-request time, over raw per-game stat counts, for just the players in that request. Two new modules (`pipeline/scoring.py`, `pipeline/player_form.py`) hold the weight model and the aggregation that used to live in `scripts/refresh_stats.py`. A new `data/processed/meta.parquet` artifact replaces the old (now-invalid) trick of reading the target season/week off `player_stats.parquet` itself, since that table now holds many historical per-game rows instead of one row per player for a single target week.

**Tech Stack:** Python 3.14, Polars, Pydantic, FastAPI, `anthropic` SDK (`client.messages.parse`), React (Vite), browser `localStorage`.

**Spec:** `docs/superpowers/specs/2026-09-19-custom-league-scoring-design.md`

## Global Constraints

- `ScoringRules` field names are canonical everywhere downstream of staging — both nflverse historical columns and Sleeper projection keys get renamed to these at the staging step. See the mapping table in Task 3/4.
- All `ScoringRules` fields default to `0.0` and are bounded `Field(ge=-10, le=10)` (spec's "Error handling" section).
- No `_ppr`-suffixed field names anywhere post-refactor — there's one set of numbers now, computed under whichever `ScoringRules` was selected, not parallel std/PPR columns (spec's context_builder section).
- Custom-rule parsing (`parse_custom_scoring_rules`) runs once, at `POST /scoring-rules` time — never per `/chat` message.
- The resolved `ScoringRules` (never raw description text) is what the frontend stores in `localStorage` and sends with `/chat`.
- No frontend test suite exists in this repo — frontend tasks verify manually via `npm run dev`, consistent with the chat-assistant plan.
- TDD + incremental commits per this repo's `incremental-commits`/`branch-naming` conventions: one commit per task step group, `feat`/`fix`/`test`/`docs` prefixes, on branch `feat/custom-league-scoring` (already created, spec already committed there).

---

### Task 1: `pipeline/scoring.py` — scoring rules model and formula

**Files:**
- Create: `pipeline/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Produces: `ScoringRules` (pydantic `BaseModel`, 13 float fields, all default `0.0`, `Field(ge=-10, le=10)`), `PRESET_STANDARD`, `PRESET_HALF_PPR`, `PRESET_PPR` (module-level `ScoringRules` instances), `compute_fantasy_points(row: dict, rules: ScoringRules) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.scoring'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/scoring.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_scoring.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/scoring.py tests/test_scoring.py
git commit -m "feat: add scoring rules model and fantasy-point formula"
```

---

### Task 2: `pipeline/player_form.py` — per-player recent-form aggregation

**Files:**
- Create: `pipeline/player_form.py`
- Test: `tests/test_player_form.py`

**Interfaces:**
- Consumes: `pipeline.scoring.ScoringRules`, `compute_fantasy_points` (Task 1).
- Produces: `compute_recent_form(player_rows: pl.DataFrame, season: int, rules: ScoringRules) -> dict` returning keys `games_played_this_season`, `avg_fantasy_points_last3`, `avg_fantasy_points_season`, `prior_season_games_played`, `prior_season_avg_fantasy_points`, `prior_season_last3_avg_fantasy_points`, `last_game_season`, `last_game_week`, `last_game_fantasy_points`. `LAST_N_WEEKS = 3` (moved here from `scripts/refresh_stats.py`, which no longer aggregates).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_form.py
import polars as pl
import pytest

from pipeline.player_form import compute_recent_form
from pipeline.scoring import PRESET_PPR, ScoringRules


def _rows(seasons, weeks, rec_yards):
    n = len(weeks)
    return pl.DataFrame({
        "season": seasons, "week": weeks,
        "receptions": [0] * n, "rec_yards": rec_yards, "rec_tds": [0] * n,
    })


def test_compute_recent_form_averages_this_seasons_last_3_games():
    rows = _rows(seasons=[2026] * 3, weeks=[1, 2, 3], rec_yards=[50, 100, 150])
    form = compute_recent_form(rows, season=2026, rules=PRESET_PPR)

    assert form["games_played_this_season"] == 3
    assert form["avg_fantasy_points_last3"] == pytest.approx(10.0)  # (5+10+15)/3
    assert form["avg_fantasy_points_season"] == pytest.approx(10.0)
    assert form["prior_season_games_played"] == 0
    assert form["last_game_season"] == 2026 and form["last_game_week"] == 3
    assert form["last_game_fantasy_points"] == pytest.approx(15.0)


def test_compute_recent_form_keeps_seasons_separate_when_current_is_thin():
    rows = _rows(
        seasons=[2025, 2025, 2025, 2026], weeks=[16, 17, 18, 1],
        rec_yards=[60, 120, 180, 990],
    )
    form = compute_recent_form(rows, season=2026, rules=PRESET_PPR)

    assert form["games_played_this_season"] == 0
    assert form["avg_fantasy_points_last3"] is None
    assert form["avg_fantasy_points_season"] is None
    assert form["prior_season_games_played"] == 3
    assert form["prior_season_avg_fantasy_points"] == pytest.approx(12.0)  # (6+12+18)/3
    assert form["prior_season_last3_avg_fantasy_points"] == pytest.approx(12.0)
    # the most recent game overall, regardless of season
    assert form["last_game_season"] == 2026 and form["last_game_week"] == 1
    assert form["last_game_fantasy_points"] == pytest.approx(99.0)


def test_compute_recent_form_reflects_the_selected_scoring_rules():
    """The same rows score differently under Standard vs PPR - proves the
    aggregation actually uses `rules`, not a hard-coded formula.
    """
    rows = _rows(seasons=[2026], weeks=[1], rec_yards=[0])
    rows = rows.with_columns(pl.lit(10).alias("receptions"))
    standard = compute_recent_form(rows, season=2026, rules=ScoringRules())
    ppr = compute_recent_form(rows, season=2026, rules=PRESET_PPR)

    assert standard["avg_fantasy_points_season"] == pytest.approx(0.0)
    assert ppr["avg_fantasy_points_season"] == pytest.approx(10.0)


def test_compute_recent_form_handles_a_rookie_with_no_prior_season_rows():
    rows = _rows(seasons=[2026], weeks=[1], rec_yards=[50])
    form = compute_recent_form(rows, season=2026, rules=PRESET_PPR)

    assert form["prior_season_games_played"] == 0
    assert form["prior_season_avg_fantasy_points"] is None
    assert form["prior_season_last3_avg_fantasy_points"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_player_form.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.player_form'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/player_form.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_player_form.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/player_form.py tests/test_player_form.py
git commit -m "feat: add request-time recent-form aggregation"
```

---

### Task 3: `scripts/refresh_stats.py` — player_stats becomes raw per-game rows + a target-season/week meta artifact

**Files:**
- Modify: `scripts/refresh_stats.py`
- Modify: `tests/test_refresh_stats.py`

**Interfaces:**
- Consumes: nothing new (still pure Polars/nflreadpy).
- Produces: `build_staged_player_stats(raw)` now also selects/renames the 13 canonical `ScoringRules` columns; `build_processed_player_stats(staged, season, week, current_player_ids) -> pl.DataFrame` now returns **one row per player per completed game** (columns: `player_id`, `player_name`, `position`, `team`, `season`, `week`, plus the 13 canonical columns) instead of pre-aggregated columns; `build_processed(staged, season, week)` additionally writes `data/processed/meta.parquet` (columns `season`, `week`, one row) — the new source of truth for "what decision is in progress," replacing the old (now-broken) trick of reading `max(week)` off `player_stats.parquet` (Task 5 updates the reader side).

**Why a meta artifact is needed (not in the original spec, found while planning):** the old `player_stats.parquet` had exactly one row per player, carrying the *target* season/week as literal columns (`pl.lit(season)`, `pl.lit(week)`). The new table holds many *historical* per-game rows spanning two real seasons, so there's no single row left to read "the target" off - `resolve_week`'s `target_week()` (Task 5) needs a dedicated place to read it.

- [ ] **Step 1: Write the failing tests**

Replace `_weekly_staged` and the `build_processed_player_stats` tests in `tests/test_refresh_stats.py`:

```python
def _weekly_staged(seasons, weeks, rec_yards, season_types=None):
    n = len(weeks)
    return pl.DataFrame({
        "player_id": ["00-1"] * n,
        "player_name": ["Test Player"] * n,
        "position": ["WR"] * n,
        "team": ["MIN"] * n,
        "season": seasons,
        "week": weeks,
        "season_type": season_types or ["REG"] * n,
        "pass_yards": [0] * n, "pass_tds": [0] * n, "pass_interceptions": [0] * n,
        "pass_2pt": [0] * n, "rush_yards": [0] * n, "rush_tds": [0] * n,
        "rush_2pt": [0] * n, "rush_attempts": [0] * n,
        "receptions": [5] * n, "rec_yards": rec_yards, "rec_tds": [0] * n,
        "rec_2pt": [0] * n, "fumbles_lost": [0] * n,
    })


def test_build_processed_player_stats_returns_one_row_per_completed_game():
    """Week 18 is the decision - only weeks strictly before it may appear."""
    staged = _weekly_staged(
        seasons=[2025] * 4, weeks=[15, 16, 17, 18], rec_yards=[50, 100, 150, 999],
    )

    result = build_processed_player_stats(staged, season=2025, week=18, current_player_ids=["00-1"])

    assert result.height == 3
    assert sorted(result["week"].to_list()) == [15, 16, 17]
    assert result.row(0, named=True)["rec_yards"] in (50, 100, 150)


def test_build_processed_player_stats_excludes_players_with_no_current_relevance():
    """Always fetching the prior season (stats_seasons) must not resurrect retirees
    or unsigned free agents - only players with a current-season game or a spot in
    this week's roster/projection universe (current_player_ids) belong in the table.
    """
    staged = _weekly_staged(seasons=[2025, 2025, 2025], weeks=[16, 17, 18], rec_yards=[50, 100, 150])

    result = build_processed_player_stats(staged, season=2026, week=1, current_player_ids=[])

    assert result.height == 0


def test_build_processed_player_stats_ignores_postseason_weeks():
    """POST week 19 outranks REG week 18 numerically - it must not appear as a completed game."""
    staged = _weekly_staged(
        seasons=[2025] * 3, weeks=[17, 18, 19], rec_yards=[50, 50, 999],
        season_types=["REG", "REG", "POST"],
    )

    result = build_processed_player_stats(staged, season=2026, week=1, current_player_ids=["00-1"])

    assert result.height == 2
    assert 19 not in result["week"].to_list()


def test_build_staged_player_stats_selects_canonical_scoring_columns():
    """The 13 ScoringRules-named columns must exist, renamed from nflverse's names."""
    raw = {
        "player_stats": pl.DataFrame({
            "player_id": ["00-1"], "player_display_name": ["Test Player"],
            "position": ["WR"], "team": ["MIN"], "opponent_team": ["GB"],
            "season": [2026], "week": [1], "season_type": ["REG"],
            "targets": [8], "receptions": [6], "carries": [1],
            "rushing_epa": [0.1], "receiving_epa": [0.2], "passing_epa": [0.0],
            "target_share": [0.3], "air_yards_share": [0.2],
            "passing_yards": [0], "passing_tds": [0], "passing_interceptions": [0],
            "passing_2pt_conversions": [0],
            "rushing_yards": [5], "rushing_tds": [0], "rushing_2pt_conversions": [0],
            "receiving_yards": [80], "receiving_tds": [1], "receiving_2pt_conversions": [0],
            "fumbles_lost_total": [0],
        }),
        "snap_counts": pl.DataFrame(schema={"pfr_player_id": pl.String, "season": pl.Int64, "week": pl.Int64, "offense_pct": pl.Float64}),
        "ff_playerids": pl.DataFrame(schema={"pfr_id": pl.String, "gsis_id": pl.String}),
        "ff_opportunity": pl.DataFrame(schema={"player_id": pl.String, "season": pl.String, "week": pl.String, "total_fantasy_points_exp": pl.Float64}),
        "ngs_receiving": pl.DataFrame(schema={"player_gsis_id": pl.String, "season": pl.Int64, "week": pl.Int64, "avg_separation": pl.Float64, "avg_yac_above_expectation": pl.Float64}),
        "ngs_rushing": pl.DataFrame(schema={"player_gsis_id": pl.String, "season": pl.Int64, "week": pl.Int64, "rush_yards_over_expected_per_att": pl.Float64}),
        "ngs_passing": pl.DataFrame(schema={"player_gsis_id": pl.String, "season": pl.Int64, "week": pl.Int64, "completion_percentage_above_expectation": pl.Float64, "aggressiveness": pl.Float64}),
    }

    staged = build_staged_player_stats(raw)
    row = staged.row(0, named=True)

    assert row["rush_attempts"] == 1  # renamed from "carries"
    assert row["rec_yards"] == 80 and row["rec_tds"] == 1
    assert row["fumbles_lost"] == 0


def test_build_processed_writes_a_target_season_week_meta_file(tmp_path, monkeypatch):
    """player_stats.parquet no longer carries a single target season/week per row -
    a dedicated meta table is the only place that survives the refactor.
    """
    monkeypatch.setattr("scripts.refresh_stats.PROCESSED_DIR", tmp_path)
    staged = {
        "player_stats": _weekly_staged(seasons=[2026], weeks=[1], rec_yards=[10]),
        "injuries": pl.DataFrame({"player_id": []}, schema={"player_id": pl.String}),
        "projections": pl.DataFrame({"player_id": []}, schema={"player_id": pl.String}),
    }

    build_processed(staged, season=2026, week=2)

    meta = pl.read_parquet(tmp_path / "meta.parquet")
    assert meta.row(0, named=True) == {"season": 2026, "week": 2}
```

Remove the now-obsolete tests that asserted pre-aggregated columns (`avg_fantasy_points_last3`, `prior_season_avg_fantasy_points`, etc.) directly on `build_processed_player_stats`'s output — that logic moved to `pipeline/player_form.py` and is covered there (Task 2).

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_refresh_stats.py -v`
Expected: FAIL (new tests reference columns/behavior that don't exist yet; old aggregation tests fail once fixtures change)

- [ ] **Step 3: Write the implementation**

In `scripts/refresh_stats.py`, replace `build_staged_player_stats`:

```python
def build_staged_player_stats(raw: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Join weekly player_stats with snap counts, xFP, and Next Gen Stats on
    (player_id, season, week). Selects the 13 canonical ScoringRules columns,
    renamed from nflverse's names, so compute_fantasy_points never needs to
    know the data's provider (docs/superpowers/specs/2026-09-19-custom-
    league-scoring-design.md).
    """
    stats = raw["player_stats"].select([
        "player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week", "season_type", "targets",
        "passing_yards", "passing_tds", "passing_interceptions", "passing_2pt_conversions",
        "rushing_yards", "rushing_tds", "rushing_2pt_conversions", "carries",
        "receptions", "receiving_yards", "receiving_tds", "receiving_2pt_conversions",
        "fumbles_lost_total",
        "rushing_epa", "receiving_epa", "passing_epa", "target_share", "air_yards_share",
    ]).rename({
        "player_display_name": "player_name",
        "passing_yards": "pass_yards", "passing_tds": "pass_tds",
        "passing_interceptions": "pass_interceptions", "passing_2pt_conversions": "pass_2pt",
        "rushing_yards": "rush_yards", "rushing_tds": "rush_tds",
        "rushing_2pt_conversions": "rush_2pt", "carries": "rush_attempts",
        "receiving_yards": "rec_yards", "receiving_tds": "rec_tds",
        "receiving_2pt_conversions": "rec_2pt", "fumbles_lost_total": "fumbles_lost",
    })

    pfr_map = _pfr_to_gsis(raw["ff_playerids"])
    snaps = (
        raw["snap_counts"]
        .rename({"pfr_player_id": "pfr_id"})
        .join(pfr_map, on="pfr_id", how="left")
        .filter(pl.col("gsis_id").is_not_null())
        .select(["gsis_id", "season", "week", "offense_pct"])
        .rename({"gsis_id": "player_id", "offense_pct": "snap_percentage"})
    )

    xfp = raw["ff_opportunity"].select([
        "player_id", "season", "week", "total_fantasy_points_exp",
    ]).with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
    ]).rename({"total_fantasy_points_exp": "xfp"})

    ngs_receiving = raw["ngs_receiving"].select([
        "player_gsis_id", "season", "week", "avg_separation", "avg_yac_above_expectation",
    ]).rename({"player_gsis_id": "player_id"})

    ngs_rushing = raw["ngs_rushing"].select([
        "player_gsis_id", "season", "week", "rush_yards_over_expected_per_att",
    ]).rename({"player_gsis_id": "player_id"})

    ngs_passing = raw["ngs_passing"].select([
        "player_gsis_id", "season", "week",
        "completion_percentage_above_expectation", "aggressiveness",
    ]).rename({"player_gsis_id": "player_id"})

    return (
        stats
        .join(snaps, on=["player_id", "season", "week"], how="left")
        .join(xfp, on=["player_id", "season", "week"], how="left")
        .join(ngs_receiving, on=["player_id", "season", "week"], how="left")
        .join(ngs_rushing, on=["player_id", "season", "week"], how="left")
        .join(ngs_passing, on=["player_id", "season", "week"], how="left")
    )
```

Replace `build_processed_player_stats` (drop `_own_last_n`, `LAST_N_WEEKS` from this file - both moved to `pipeline/player_form.py`):

```python
CANONICAL_STAT_COLUMNS = [
    "pass_yards", "pass_tds", "pass_interceptions", "pass_2pt",
    "rush_yards", "rush_tds", "rush_2pt", "rush_attempts",
    "receptions", "rec_yards", "rec_tds", "rec_2pt", "fumbles_lost",
]


def build_processed_player_stats(
    staged: pl.DataFrame, season: int, week: int, current_player_ids
) -> pl.DataFrame:
    """One row per player per completed game, carrying raw counting stats for
    request-time scoring (pipeline.scoring.compute_fantasy_points /
    pipeline.player_form.compute_recent_form) instead of pre-aggregating under
    one fixed formula.

    Still filters to weeks strictly before `week` (_completed_weeks - a
    projection for a game already in the books is not a projection) and to
    the current roster/projection universe (current_player_ids) - without
    that filter, always fetching the prior season (stats_seasons) would let
    anyone who merely *played* last season - retirees, unsigned free agents -
    leak into the table on stale, no-longer-actionable production (PR #11).
    """
    completed = _completed_weeks(staged, season, week)
    this_season = completed.filter(pl.col("season") == season)

    relevant_ids = set(this_season["player_id"].to_list()) | set(current_player_ids)
    return completed.filter(pl.col("player_id").is_in(list(relevant_ids))).select(
        ["player_id", "player_name", "position", "team", "season", "week"]
        + CANONICAL_STAT_COLUMNS
    )
```

Update `build_processed` to also write the meta table:

```python
def build_processed(staged: dict[str, pl.DataFrame], season: int, week: int) -> None:
    """Build and persist all processed/*.parquet tables from the staged tables."""
    current_player_ids = (
        staged["projections"].filter(pl.col("player_id").is_not_null())["player_id"].to_list()
    )
    processed = {
        "player_stats": build_processed_player_stats(
            staged["player_stats"], season, week, current_player_ids
        ),
        "injuries": build_processed_injuries(staged["injuries"]),
        "projections": build_processed_projections(staged["projections"]),
        # player_stats no longer carries a single target season/week per row
        # (it's many historical per-game rows now) - this is the only place
        # "what decision is in progress" survives. Read by
        # pipeline.decision_engine.target_season_week().
        "meta": pl.DataFrame({"season": [season], "week": [week]}),
    }
    for name, df in processed.items():
        _write(df, PROCESSED_DIR, name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_refresh_stats.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/refresh_stats.py tests/test_refresh_stats.py
git commit -m "feat: persist raw per-game player stats instead of pre-aggregated PPR"
```

---

### Task 4: `scripts/refresh_stats.py` — projections carry raw per-category values

**Files:**
- Modify: `scripts/refresh_stats.py`
- Modify: `tests/test_refresh_stats.py` (add; no existing projection tests to remove — none exist today)

**Interfaces:**
- Produces: `fetch_sleeper_projections(season, week)` now extracts the 13 canonical fields from Sleeper's `stats` dict (confirmed live: `pass_yd`, `pass_td`, `pass_int`, `pass_2pt`, `rush_yd`, `rush_td`, `rush_2pt`, `rush_att`, `rec`, `rec_yd`, `rec_td`, `rec_2pt`, `fum_lost`) instead of `pts_ppr`/`pts_half_ppr`/`pts_std`. `build_processed_projections(staged)` selects the canonical columns instead of `projected_points`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_refresh_stats.py (add)
def test_build_processed_projections_carries_raw_canonical_stat_columns():
    staged = pl.DataFrame({
        "player_id": ["00-1"], "player_name": ["Test Player"], "player_name_ecr": [None],
        "position": ["WR"], "pos": [None], "team": ["MIN"], "team_ecr": [None],
        "season": [2026], "week": [2],
        "pass_yards": [0.0], "pass_tds": [0.0], "pass_interceptions": [0.0], "pass_2pt": [0.0],
        "rush_yards": [2.0], "rush_tds": [0.0], "rush_2pt": [0.0], "rush_attempts": [0.5],
        "receptions": [4.4], "rec_yards": [57.0], "rec_tds": [0.4], "rec_2pt": [0.0],
        "fumbles_lost": [0.02],
        "ecr_rank": [18.0], "ecr_position_rank": ["WR9"],
    })

    result = build_processed_projections(staged)
    row = result.row(0, named=True)

    assert row["receptions"] == pytest.approx(4.4)
    assert row["rec_yards"] == pytest.approx(57.0)
    assert row["ecr_rank"] == pytest.approx(18.0)
    assert "projected_points" not in result.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_refresh_stats.py::test_build_processed_projections_carries_raw_canonical_stat_columns -v`
Expected: FAIL (`projected_points` still present / canonical columns missing)

- [ ] **Step 3: Write the implementation**

Replace `fetch_sleeper_projections`:

```python
def fetch_sleeper_projections(season: int, week: int) -> pl.DataFrame:
    """Fetch Sleeper's weekly projections for one season/week, keeping the raw
    per-category values (confirmed live: pass_yd, pass_td, pass_int, pass_2pt,
    rush_yd, rush_td, rush_2pt, rush_att, rec, rec_yd, rec_td, rec_2pt,
    fum_lost) instead of a single pre-computed point total, so request-time
    scoring (pipeline.scoring.compute_fantasy_points) can apply any tier.
    """
    url = SLEEPER_PROJECTIONS_URL.format(season=season, week=week)
    resp = requests.get(url, params={"season_type": "regular"}, timeout=30)
    resp.raise_for_status()
    rows = []
    for entry in resp.json():
        stats = entry.get("stats") or {}
        player = entry.get("player") or {}
        first, last = player.get("first_name"), player.get("last_name")
        rows.append({
            "sleeper_id": entry.get("player_id"),
            "season": season,
            "week": week,
            "player_name": f"{first} {last}".strip() if first or last else None,
            "position": player.get("position"),
            "team": entry.get("team"),
            "pass_yards": stats.get("pass_yd"),
            "pass_tds": stats.get("pass_td"),
            "pass_interceptions": stats.get("pass_int"),
            "pass_2pt": stats.get("pass_2pt"),
            "rush_yards": stats.get("rush_yd"),
            "rush_tds": stats.get("rush_td"),
            "rush_2pt": stats.get("rush_2pt"),
            "rush_attempts": stats.get("rush_att"),
            "receptions": stats.get("rec"),
            "rec_yards": stats.get("rec_yd"),
            "rec_tds": stats.get("rec_td"),
            "rec_2pt": stats.get("rec_2pt"),
            "fumbles_lost": stats.get("fum_lost"),
        })
    return pl.DataFrame(rows, infer_schema_length=None)
```

Replace `build_processed_projections`:

```python
def build_processed_projections(staged: pl.DataFrame) -> pl.DataFrame:
    """Flatten staged projections into raw per-category columns for request-
    time scoring, plus ECR ranking.
    """
    return staged.filter(pl.col("player_id").is_not_null()).select(
        [
            "player_id",
            pl.coalesce(["player_name", "player_name_ecr"]).alias("player_name"),
            pl.coalesce(["position", "pos"]).alias("position"),
            pl.coalesce(["team", "team_ecr"]).alias("team"),
            "season", "week",
        ]
        + CANONICAL_STAT_COLUMNS
        + [pl.lit("sleeper").alias("source"), "ecr_rank", "ecr_position_rank"]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_refresh_stats.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/refresh_stats.py tests/test_refresh_stats.py
git commit -m "feat: carry raw per-category projected stats instead of pts_ppr"
```

---

### Task 5: `pipeline/decision_engine.py` — target season/week from the meta file, `season` threaded through `decide()`

**Files:**
- Modify: `pipeline/decision_engine.py`
- Modify: `tests/test_decision_engine.py`

**Interfaces:**
- Consumes: `data/processed/meta.parquet` (Task 3).
- Produces: `target_season_week() -> tuple[int, int]` (new); `target_week() -> int` (unchanged signature, now reads the meta file); `resolve_season(season: int | None) -> int` (new); `decide(question, players=None, season=None, week=None, tables=None, news_fn=None, scoring_rules=None) -> Decision` (gains `season`, `scoring_rules`); calls `build_context(resolved_players, resolved_season, resolved_week, tables=tables, news_fn=news_fn, scoring_rules=scoring_rules)` — **note the new positional argument order**, `season` before `week` (Task 6 defines `build_context`'s new signature).

- [ ] **Step 1: Write the failing tests**

Replace the two `target_week` tests in `tests/test_decision_engine.py` and add `resolve_season` coverage:

```python
def test_target_season_week_reads_the_meta_file(monkeypatch):
    fixture = pl.DataFrame({"season": [2026], "week": [8]})
    monkeypatch.setattr("pipeline.decision_engine.Path.exists", lambda self: True)
    monkeypatch.setattr("pipeline.decision_engine.pl.read_parquet", lambda *_: fixture)
    assert target_season_week() == (2026, 8)
    assert target_week() == 8


def test_target_season_week_raises_when_meta_file_is_missing(monkeypatch):
    monkeypatch.setattr("pipeline.decision_engine.Path.exists", lambda self: False)
    with pytest.raises(DataUnavailableError):
        target_season_week()


def test_resolve_season_prefers_the_explicit_argument(monkeypatch):
    monkeypatch.setattr("pipeline.decision_engine.target_season_week", lambda: (2026, 8))
    assert resolve_season(2025) == 2025


def test_resolve_season_falls_back_to_the_meta_file(monkeypatch):
    monkeypatch.setattr("pipeline.decision_engine.target_season_week", lambda: (2026, 8))
    assert resolve_season(None) == 2026
```

Remove `test_target_week_reads_the_max_week_in_processed_stats` and `test_target_week_raises_on_empty_processed_stats` (superseded by the meta-file versions above).

Update imports at the top of the test file:

```python
from pipeline.decision_engine import (
    Decision,
    DecisionError,
    decide,
    format_decision,
    resolve_players,
    resolve_season,
    target_season_week,
    target_week,
)
```

Update `_fixture_tables()` and every `decide(...)` call site to the new `player_stats` per-game-row shape and to expect the new `build_context` call signature — this overlaps with Task 6's context_builder rewrite, so make this edit together with Task 6's fixture changes rather than twice. For this task alone, it's enough that `target_season_week`/`resolve_season` pass; the `decide(...)` tests will be finished in Task 6's step.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_decision_engine.py -k "target_season_week or resolve_season" -v`
Expected: FAIL with `ImportError`/`AttributeError`

- [ ] **Step 3: Write the implementation**

```python
def target_season_week() -> tuple[int, int]:
    """The (season, week) the current processed data targets - the decision
    in progress. Read from data/processed/meta.parquet, written by
    refresh_stats.py's build_processed().

    player_stats.parquet no longer carries a single target season/week per
    row (it holds many historical per-game rows across two real seasons), so
    this dedicated meta table is the only place it survives.
    """
    meta_path = PROCESSED_DIR / "meta.parquet"
    if not meta_path.exists():
        raise DataUnavailableError(
            "data/processed/meta.parquet is missing - run python scripts/refresh_stats.py"
        )
    row = pl.read_parquet(meta_path).row(0, named=True)
    return row["season"], row["week"]


def target_week() -> int:
    """The week half of target_season_week() - most callers only override week."""
    return target_season_week()[1]


def resolve_season(season: int | None) -> int:
    """Explicit season, else whatever the current processed data targets.

    Unlike week, there's no free-text season extraction - nobody types
    "season 2026" in a start/sit question - so the only override is passing
    `season` directly.
    """
    if season is not None:
        return season
    return target_season_week()[0]
```

Update `decide()`:

```python
def decide(
    question: str,
    players: list[str] | None = None,
    season: int | None = None,
    week: int | None = None,
    tables: dict[str, pl.DataFrame] | None = None,
    news_fn: Callable[[str | None, str], list[str]] | None = None,
    scoring_rules: ScoringRules | None = None,
) -> Decision:
    """Answer a two-player start/sit question end to end.

    `players`/`season`/`week` override what the question text says (season
    has no text-extraction path - see resolve_season); `tables` lets tests
    inject fixture DataFrames instead of reading data/processed/.
    `scoring_rules` defaults to full PPR when omitted (build_context's own
    default) - see docs/superpowers/specs/2026-09-19-custom-league-scoring-design.md.
    """
    resolved_players = resolve_players(question, players)
    resolved_week = resolve_week(question, week)
    resolved_season = resolve_season(season)
    news_fn = news_fn if news_fn is not None else retrieve_news
    context = build_context(
        resolved_players, resolved_season, resolved_week,
        tables=tables, news_fn=news_fn, scoring_rules=scoring_rules,
    )
    recommendation = run_llm(context, question)
    _check_recommendation(recommendation, resolved_players)
    return Decision(
        question=question,
        players=resolved_players,
        week=resolved_week,
        context=context,
        recommendation=recommendation,
    )
```

Add the import: `from pipeline.scoring import ScoringRules`.

Add a `--season` CLI flag in `main()`, mirroring the existing `--week` flag, and pass it through:

```python
parser.add_argument("--season", type=int, default=None)
...
decision = decide(question, players=args.players, season=args.season, week=args.week)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_decision_engine.py -k "target_season_week or resolve_season" -v`
Expected: PASS (4 tests) — the rest of this file still fails until Task 6 finishes the fixture rewrite; that's expected at this checkpoint.

- [ ] **Step 5: Commit**

```bash
git add pipeline/decision_engine.py tests/test_decision_engine.py
git commit -m "feat: read target season/week from meta.parquet, thread season through decide()"
```

---

### Task 6: `pipeline/context_builder.py` — rewrite to use `compute_recent_form`/`compute_fantasy_points`, add `season` and `scoring_rules` params

**Files:**
- Modify: `pipeline/context_builder.py`
- Modify: `tests/test_context_builder.py`
- Modify: `tests/test_decision_engine.py` (finish the fixture rewrite started in Task 5)
- Modify: `tests/test_chat_engine.py` (fixture columns only — signature changes are Task 7)

**Interfaces:**
- Consumes: `pipeline.scoring.{ScoringRules, PRESET_PPR, compute_fantasy_points}` (Task 1), `pipeline.player_form.compute_recent_form` (Task 2).
- Produces: `build_context(players: list[str], season: int, week: int, tables=None, news_fn=None, scoring_rules: ScoringRules | None = None) -> str` — **note `season` is now a required positional parameter before `week`**. `PlayerNotFoundError` now fires only when a player has zero rows anywhere in `player_stats` (not "zero rows for this specific week"), since `player_stats` is no longer week-keyed.

- [ ] **Step 1: Write the failing tests**

Full rewrite of `tests/test_context_builder.py`'s fixtures and every `build_context(...)` call to the new per-game-row shape and new signature:

```python
import polars as pl
import pytest

from pipeline.context_builder import PlayerNotFoundError, build_context
from pipeline.entity_extraction import extract_players, extract_week, known_player_names
from pipeline.scoring import PRESET_HALF_PPR, PRESET_STANDARD
from retrieval.news_retriever import NewsItem


def _game_row(player_id, name, season, week, **stats):
    base = {
        "player_id": player_id, "player_name": name, "position": "WR", "team": "MIN",
        "season": season, "week": week,
        "pass_yards": 0, "pass_tds": 0, "pass_interceptions": 0, "pass_2pt": 0,
        "rush_yards": 0, "rush_tds": 0, "rush_2pt": 0, "rush_attempts": 0,
        "receptions": 0, "rec_yards": 0, "rec_tds": 0, "rec_2pt": 0, "fumbles_lost": 0,
    }
    base.update(stats)
    return base


def _fixture_tables():
    """Jordan Love and Jared Goff, 4 games played this season (weeks 1-4),
    week 5 is the target (not yet played). Round per-game rates (5 receptions
    always, rec_yards varies) so every tier's math is easy to hand-check:
    Love = 100 rec_yards/game -> PPR 15.0, Half-PPR 12.5, Standard 10.0 per game.
    Goff = 50 rec_yards/game -> PPR 10.0 per game.
    """
    rows = [
        _game_row("00-love", "Jordan Love", 2026, w, receptions=5, rec_yards=100)
        for w in range(1, 5)
    ] + [
        _game_row("00-goff", "Jared Goff", 2026, w, receptions=5, rec_yards=50)
        for w in range(1, 5)
    ]
    player_stats = pl.DataFrame(rows)
    projections = pl.DataFrame([
        _game_row("00-love", "Jordan Love", 2026, 5, receptions=5, rec_yards=80),
        _game_row("00-goff", "Jared Goff", 2026, 5, receptions=5, rec_yards=60),
    ])
    injuries = pl.DataFrame({
        "player_id": ["00-love"], "player_name": ["Jordan Love"],
        "status": ["Questionable"], "practice_level": ["Full practice Friday"],
    })
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_formats_two_player_comparison():
    """Matches the README §7 PLAYER COMPARISON layout: header, stats, projection, injury.

    Defaults to PPR: Love's 100 rec_yards + 5 receptions/game = 10.0 + 5.0 = 15.0;
    projection (80 rec_yards + 5 receptions) = 8.0 + 5.0 = 13.0. Goff: 50 rec_yards
    + 5 receptions/game = 5.0 + 5.0 = 10.0; projection (60 + 5) = 6.0 + 5.0 = 11.0.
    """
    context = build_context(
        ["Jordan Love", "Jared Goff"], season=2026, week=5, tables=_fixture_tables()
    )

    assert context.startswith("PLAYER COMPARISON\n\n")
    assert (
        "Jordan Love:\n- This season (4 games): 15.0 season avg, 15.0 avg over last 3 games\n"
        "- Projected points: 13.0\n- Injury: Questionable -> Full practice Friday" in context
    )
    assert (
        "Jared Goff:\n- This season (4 games): 10.0 season avg, 10.0 avg over last 3 games\n"
        "- Projected points: 11.0\n- Injury: Healthy" in context
    )


def test_build_context_reflects_the_selected_scoring_tier():
    """Same fixture, different tier - the numbers must actually change.

    Love's 100 rec_yards + 5 receptions per game: PPR = 10.0 + 5*1.0 = 15.0;
    Standard = 10.0 + 5*0.0 = 10.0; Half-PPR = 10.0 + 5*0.5 = 12.5.
    """
    tables = _fixture_tables()
    ppr_context = build_context(["Jordan Love"], season=2026, week=5, tables=tables)
    standard_context = build_context(
        ["Jordan Love"], season=2026, week=5, tables=tables, scoring_rules=PRESET_STANDARD
    )
    half_context = build_context(
        ["Jordan Love"], season=2026, week=5, tables=tables, scoring_rules=PRESET_HALF_PPR
    )

    assert "15.0 season avg" in ppr_context
    assert "10.0 season avg" in standard_context
    assert "12.5 season avg" in half_context


def test_build_context_raises_for_unknown_player():
    with pytest.raises(PlayerNotFoundError):
        build_context(["Nobody Here"], season=2026, week=5, tables=_fixture_tables())


def test_build_context_omits_news_bullet_when_news_fn_is_not_passed():
    context = build_context(
        ["Jordan Love", "Jared Goff"], season=2026, week=5, tables=_fixture_tables()
    )
    assert "Recent news" not in context


def test_build_context_adds_a_news_bullet_per_snippet_when_news_fn_returns_some():
    def news_fn(player_id, player_name):
        if player_name != "Jordan Love":
            return []
        return [NewsItem(
            title="Packers stay aggressive", snippet="Packers plan to stay aggressive...",
            link="https://example.com/a", source="ESPN", published_at="2026-09-01T00:00:00+00:00",
        )]

    context = build_context(
        ["Jordan Love", "Jared Goff"], season=2026, week=5,
        tables=_fixture_tables(), news_fn=news_fn,
    )
    assert '- Recent news:\n  - "Packers plan to stay aggressive..."' in context
    assert "Recent news" not in context.split("Jared Goff:")[1]


def _partial_current_season_tables(games_this_season=1, prior_games=4):
    """A player with a thin current-season sample - last season is still
    relevant. 60 rec_yards + 4 receptions/game (PPR default) = 6.0 + 4.0 = 10.0
    per game, both seasons, so every branch's expected number is the same
    round 10.0 regardless of how many games are in the window.
    """
    rows = [
        _game_row("00-nabers", "Malik Nabers", 2026, w, receptions=4, rec_yards=60)
        for w in range(1, 1 + games_this_season)
    ] + [
        _game_row("00-nabers", "Malik Nabers", 2025, w, receptions=4, rec_yards=60)
        for w in range(1, 1 + prior_games)
    ]
    player_stats = pl.DataFrame(rows)
    projections = pl.DataFrame({"player_id": [], "player_name": [], "week": []}, schema={
        "player_id": pl.String, "player_name": pl.String, "week": pl.Int64,
    })
    injuries = pl.DataFrame({"player_id": [], "player_name": [], "status": [], "practice_level": []})
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def test_build_context_shows_last_season_separately_when_this_season_is_thin():
    context = build_context(
        ["Malik Nabers"], season=2026, week=2, tables=_partial_current_season_tables()
    )
    assert "- This season (1 game): 10.0 avg fantasy points" in context
    assert "- Last season (4 games): 10.0 season avg, 10.0 avg over final 3 games" in context
    assert "- Most recent game played (Week 1, 2026): 10.0 pts" in context


def test_build_context_omits_last_season_once_three_current_season_games_exist():
    context = build_context(
        ["Malik Nabers"], season=2026, week=4,
        tables=_partial_current_season_tables(games_this_season=3),
    )
    assert "Last season" not in context
    assert "Most recent game played" not in context


def test_build_context_shows_only_last_season_when_no_games_played_yet():
    context = build_context(
        ["Malik Nabers"], season=2026, week=1,
        tables=_partial_current_season_tables(games_this_season=0),
    )
    assert "- This season: no games played yet" in context
    assert "- Last season (4 games): 10.0 season avg, 10.0 avg over final 3 games" in context


def test_build_context_omits_last_season_line_when_player_has_none():
    context = build_context(
        ["Malik Nabers"], season=2026, week=2,
        tables=_partial_current_season_tables(prior_games=0),
    )
    assert "Last season" not in context
    assert "- Most recent game played (Week 1, 2026): 10.0 pts" in context


def test_extract_week_finds_week_number_in_free_text():
    assert extract_week("Should I start Jordan Love in week 5?") == 5
    assert extract_week("who do I play in Week12") == 12
    assert extract_week("no week mentioned here") is None


def test_extract_players_matches_known_names_in_order_of_appearance():
    known_names = ["Jordan Love", "Colston Loveland", "Jared Goff"]
    text = "Should I start Jared Goff or Jordan Love this week?"
    assert extract_players(text, known_names=known_names) == ["Jared Goff", "Jordan Love"]


def test_extract_players_ignores_substring_collisions():
    known_names = ["Jordan Love", "Colston Loveland"]
    text = "Is Colston Loveland going to see more targets?"
    assert extract_players(text, known_names=known_names) == ["Colston Loveland"]


def test_extract_players_matches_names_ending_in_punctuation():
    known_names = ["Marvin Harrison Jr."]
    text = "Start Marvin Harrison Jr. in week 5"
    assert extract_players(text, known_names=known_names) == ["Marvin Harrison Jr."]


def test_known_player_names_drops_null_rows(monkeypatch):
    fixture = pl.DataFrame({"player_name": ["Jordan Love", None, "Jared Goff"]})
    monkeypatch.setattr("pipeline.entity_extraction.pl.read_parquet", lambda *_: fixture)
    assert known_player_names() == ["Jordan Love", "Jared Goff"]
```

(`test_build_context_joins_on_player_id_not_name`-style id-mismatch coverage is retained implicitly — every fixture above already keys by `player_id`, matching real-data shape, so a dedicated duplicate isn't needed; drop that old test rather than port it, since the id-based join code path in `_match_player` is unchanged from before and is exercised by every test here.)

Now finish `tests/test_decision_engine.py`'s fixtures (started in Task 5):

```python
def _fixture_tables():
    player_stats = pl.DataFrame([
        {"player_id": "00-love", "player_name": "Jordan Love", "season": 2026, "week": w,
         "pass_yards": 0, "pass_tds": 0, "pass_interceptions": 0, "pass_2pt": 0,
         "rush_yards": 0, "rush_tds": 0, "rush_2pt": 0, "rush_attempts": 0,
         "receptions": 1, "rec_yards": 195, "rec_tds": 0, "rec_2pt": 0, "fumbles_lost": 0}
        for w in range(1, 5)
    ] + [
        {"player_id": "00-goff", "player_name": "Jared Goff", "season": 2026, "week": w,
         "pass_yards": 0, "pass_tds": 0, "pass_interceptions": 0, "pass_2pt": 0,
         "rush_yards": 0, "rush_tds": 0, "rush_2pt": 0, "rush_attempts": 0,
         "receptions": 1, "rec_yards": 125, "rec_tds": 0, "rec_2pt": 0, "fumbles_lost": 0}
        for w in range(1, 5)
    ])
    projections = pl.DataFrame({
        "player_id": ["00-love", "00-goff"], "player_name": ["Jordan Love", "Jared Goff"],
        "week": [5, 5],
        "pass_yards": [0, 0], "pass_tds": [0, 0], "pass_interceptions": [0, 0], "pass_2pt": [0, 0],
        "rush_yards": [0, 0], "rush_tds": [0, 0], "rush_2pt": [0, 0], "rush_attempts": [0, 0],
        "receptions": [1, 1], "rec_yards": [161, 133], "rec_tds": [0, 0], "rec_2pt": [0, 0],
        "fumbles_lost": [0, 0],
    })
    injuries = pl.DataFrame({
        "player_id": ["00-love"], "player_name": ["Jordan Love"],
        "status": ["Questionable"], "practice_level": ["Full practice Friday"],
    })
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}
```

Every `decide(...)` call in that file already passes `week=5`/`week=9`/etc. by keyword and `tables=_fixture_tables()` — those keep working unchanged since `season` has a sensible resolution path (`resolve_season(None)` → `target_season_week()`), **except** tests must now also monkeypatch `target_season_week` wherever they currently monkeypatch `target_week`, and pass `season=2026` explicitly wherever the old tests relied on an implicit season. Update:
- `test_decide_falls_back_to_target_week_when_question_has_none`: monkeypatch `pipeline.decision_engine.target_season_week` returning `(2026, 5)` instead of `pipeline.decision_engine.target_week` returning `5`.
- `test_target_week_reads_the_max_week_in_processed_stats` / `test_target_week_raises_on_empty_processed_stats`: already replaced in Task 5.
- All other `decide(...)` calls: add `season=2026` explicitly (the fixture's only season), since with no live `meta.parquet` in tests, an unmocked `resolve_season(None)` would raise `DataUnavailableError`.

Finally, update `tests/test_chat_engine.py`'s `_fixture_tables()` to the same per-game-row shape (column names only — `chat()`'s signature changes in Task 7, so this file's `chat(...)` calls stay as-is here and get `season=`/updated in Task 7).

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_context_builder.py tests/test_decision_engine.py -v`
Expected: FAIL (old `build_context`/`_format_player` signatures don't accept `season`)

- [ ] **Step 3: Write the implementation**

```python
# pipeline/context_builder.py
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
    form = compute_recent_form(player_rows, season, scoring_rules)

    proj = _match_player(tables["projections"], name, player_id).filter(pl.col("week") == week)
    projected = (
        compute_fantasy_points(proj.row(0, named=True), scoring_rules) if proj.height else None
    )

    lines = [f"{name}:", *_format_recent_form(form)]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_context_builder.py tests/test_decision_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/context_builder.py tests/test_context_builder.py tests/test_decision_engine.py
git commit -m "feat: compute recent form and projections from raw stats at request time"
```

---

### Task 7: `pipeline/chat_engine.py` — thread `season`/`scoring_rules` through `chat()`

**Files:**
- Modify: `pipeline/chat_engine.py`
- Modify: `tests/test_chat_engine.py`

**Interfaces:**
- Consumes: `pipeline.decision_engine.resolve_season` (Task 5), `pipeline.scoring.ScoringRules` (Task 1).
- Produces: `chat(message, mentioned_players=None, season=None, week=None, history=None, tables=None, news_fn=None, scoring_rules=None) -> ChatResult` (gains `season`, `scoring_rules`; `ChatResult` itself is unchanged).

- [ ] **Step 1: Write the failing tests**

Update `tests/test_chat_engine.py`'s `_fixture_tables()` to the per-game-row shape (same pattern as Task 6's context_builder fixtures) and add:

```python
def _fixture_tables():
    """Round per-game rates so the math is easy to hand-check: Love = 100
    rec_yards + 5 receptions/game -> PPR 15.0/game. Goff = 50 rec_yards + 5
    receptions -> PPR 10.0. Nix = 70 rec_yards + 5 receptions -> PPR 12.0.
    """
    player_stats = pl.DataFrame([
        {"player_id": pid, "player_name": name, "season": 2026, "week": w,
         "pass_yards": 0, "pass_tds": 0, "pass_interceptions": 0, "pass_2pt": 0,
         "rush_yards": 0, "rush_tds": 0, "rush_2pt": 0, "rush_attempts": 0,
         "receptions": 5, "rec_yards": rec_yards, "rec_tds": 0, "rec_2pt": 0, "fumbles_lost": 0}
        for pid, name, rec_yards in [
            ("00-love", "Jordan Love", 100), ("00-goff", "Jared Goff", 50), ("00-nix", "Bo Nix", 70),
        ]
        for w in range(1, 5)
    ])
    projections = pl.DataFrame({
        "player_id": ["00-love", "00-goff", "00-nix"],
        "player_name": ["Jordan Love", "Jared Goff", "Bo Nix"],
        "week": [5, 5, 5],
        "pass_yards": [0, 0, 0], "pass_tds": [0, 0, 0], "pass_interceptions": [0, 0, 0],
        "pass_2pt": [0, 0, 0], "rush_yards": [0, 0, 0], "rush_tds": [0, 0, 0],
        "rush_2pt": [0, 0, 0], "rush_attempts": [0, 0, 0],
        "receptions": [5, 5, 5], "rec_yards": [80, 60, 70], "rec_tds": [0, 0, 0],
        "rec_2pt": [0, 0, 0], "fumbles_lost": [0, 0, 0],
    })
    injuries = pl.DataFrame({
        "player_id": ["00-love"], "player_name": ["Jordan Love"],
        "status": ["Questionable"], "practice_level": ["Full practice Friday"],
    })
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}
```

Add `season=2026` to every `chat(...)` call in this file (all currently pass `week=5` by keyword, so this is additive), and one new test:

```python
def test_chat_passes_scoring_rules_through_to_context(stub_run_chat_llm):
    from pipeline.scoring import PRESET_STANDARD

    chat(
        "Should I start Jordan Love or Jared Goff?",
        mentioned_players=["Jordan Love", "Jared Goff"],
        season=2026, week=5,
        tables=_fixture_tables(), news_fn=lambda *_: [],
        scoring_rules=PRESET_STANDARD,
    )
    # Love's PPR season avg would be 15.0 (10.0 rec_yards + 5.0 reception credit).
    # Standard drops the reception credit entirely - if scoring_rules were being
    # silently ignored, "15.0" would still show up; it must not.
    assert "15.0" not in stub_run_chat_llm["context"]
    assert "10.0 season avg" in stub_run_chat_llm["context"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_chat_engine.py -v`
Expected: FAIL (`chat()` doesn't accept `season`/`scoring_rules` yet)

- [ ] **Step 3: Write the implementation**

```python
from pipeline.decision_engine import DecisionError, resolve_season, resolve_week
from pipeline.scoring import ScoringRules

...

def chat(
    message: str,
    mentioned_players: list[str] | None = None,
    season: int | None = None,
    week: int | None = None,
    history: list[dict[str, str]] | None = None,
    tables: dict[str, pl.DataFrame] | None = None,
    news_fn: Callable[[str | None, str], list[NewsItem]] | None = None,
    scoring_rules: ScoringRules | None = None,
) -> ChatResult:
    """Answer any fantasy-football question about the resolved players.

    `scoring_rules` defaults to full PPR when omitted (build_context's own
    default) - see docs/superpowers/specs/2026-09-19-custom-league-scoring-design.md.
    """
    mentioned_players = mentioned_players or []
    history = history or []
    players = resolve_chat_players(message, mentioned_players)
    resolved_week = resolve_week(message, week)
    resolved_season = resolve_season(season)

    news_fn = news_fn if news_fn is not None else retrieve_news
    collected: list[NewsItem] = []

    def _tracking_news_fn(player_id: str | None, player_name: str) -> list[NewsItem]:
        items = news_fn(player_id, player_name)
        collected.extend(items)
        return items

    context = build_context(
        players, resolved_season, resolved_week,
        tables=tables, news_fn=_tracking_news_fn, scoring_rules=scoring_rules,
    )
    answer = run_chat_llm(context, message, history)
    _check_recommendation(answer.recommendation, players)

    seen_links: set[str] = set()
    sources: list[NewsItem] = []
    for item in collected:
        if item.link not in seen_links:
            seen_links.add(item.link)
            sources.append(item)

    return ChatResult(
        answer=answer.answer,
        sources=sources,
        players_discussed=players,
        recommendation=answer.recommendation,
        week=resolved_week,
        context=context,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_chat_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/chat_engine.py tests/test_chat_engine.py
git commit -m "feat: thread season and scoring_rules through chat()"
```

---

### Task 8: `llm/interface.py` — parse a free-text custom scoring description

**Files:**
- Modify: `llm/interface.py`

**Interfaces:**
- Consumes: `pipeline.scoring.ScoringRules` (Task 1).
- Produces: `parse_custom_scoring_rules(base: ScoringRules, description: str) -> ScoringRules`.

**Testing note:** matching this repo's existing convention, `run_llm`/`run_chat_llm` have no dedicated unit tests (they're only exercised indirectly via `decision_engine`/`chat_engine` tests that stub them out entirely) — real Anthropic API calls aren't mocked anywhere in this codebase. `parse_custom_scoring_rules` follows the same pattern: no unit test here; Task 9's `POST /scoring-rules` tests stub this function out, and Task 11's live end-to-end verification exercises it for real.

- [ ] **Step 1: Write the implementation**

```python
from pipeline.scoring import ScoringRules

...

def parse_custom_scoring_rules(base: ScoringRules, description: str) -> ScoringRules:
    """Turn a free-text description of league-scoring modifications into a
    full ScoringRules, seeded from `base` (docs/superpowers/specs/2026-09-19-
    custom-league-scoring-design.md). Runs once, at settings-save time
    (POST /scoring-rules) - never per chat message.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = anthropic.Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    prompt = (
        "A fantasy football league's base scoring rules are (points per unit):\n"
        f"{base.model_dump_json(indent=2)}\n\n"
        "The user describes how their league's rules differ from this base:\n"
        f'"{description}"\n\n'
        "Return the full set of scoring rules with only the described "
        "categories changed - every field the description doesn't mention "
        "must keep its base value exactly."
    )
    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=(
            "You configure fantasy football scoring rules from a user's "
            "description. Only change what the user actually describes."
        ),
        messages=[{"role": "user", "content": prompt}],
        output_format=ScoringRules,
    )
    return response.parsed_output
```

(No test-runner step for this task — see the testing note above. Verified live in Task 11.)

- [ ] **Step 2: Commit**

```bash
git add llm/interface.py
git commit -m "feat: parse custom scoring rule descriptions via the LLM"
```

---

### Task 9: `api/main.py` — `POST /scoring-rules` endpoint, `ChatRequest` gains `season`/`scoring_rules`

**Files:**
- Modify: `api/main.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `pipeline.scoring.{ScoringRules, PRESET_PPR, PRESET_HALF_PPR, PRESET_STANDARD}` (Task 1), `llm.interface.parse_custom_scoring_rules` (Task 8), `pipeline.chat_engine.chat` (Task 7, already imported).
- Produces: `POST /scoring-rules` (request: `{base, base_hint, custom_description}`, response: `{rules: ScoringRules}`); `ChatRequest.season: int | None`, `ChatRequest.scoring_rules: ScoringRules | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api.py (add)
from pipeline.scoring import PRESET_HALF_PPR, PRESET_PPR, PRESET_STANDARD, ScoringRules


def test_post_scoring_rules_returns_the_matching_preset_without_an_llm_call(monkeypatch):
    def _fail_if_called(*_a, **_k):
        raise AssertionError("parse_custom_scoring_rules must not be called for a built-in preset")

    monkeypatch.setattr("api.main.parse_custom_scoring_rules", _fail_if_called)

    response = client.post("/scoring-rules", json={"base": "half_ppr"})

    assert response.status_code == 200
    assert response.json()["rules"]["receptions"] == pytest.approx(0.5)


def test_post_scoring_rules_custom_calls_the_parser_with_the_chosen_base_hint(monkeypatch):
    calls = {}

    def fake_parse(base, description):
        calls.update(base=base, description=description)
        return base.model_copy(update={"receptions": 0.5})

    monkeypatch.setattr("api.main.parse_custom_scoring_rules", fake_parse)

    response = client.post("/scoring-rules", json={
        "base": "custom", "base_hint": "ppr", "custom_description": "catches are 0.5",
    })

    assert response.status_code == 200
    assert response.json()["rules"]["receptions"] == pytest.approx(0.5)
    assert calls["base"] == PRESET_PPR
    assert calls["description"] == "catches are 0.5"


def test_post_scoring_rules_custom_without_a_description_is_a_400():
    response = client.post("/scoring-rules", json={"base": "custom"})
    assert response.status_code == 400


def test_post_chat_defaults_scoring_rules_to_none_when_omitted(stub_chat):
    client.post("/chat", json={"message": "How's Jordan Love looking?", "mentioned_players": ["Jordan Love"]})
    assert stub_chat["scoring_rules"] is None


def test_post_chat_passes_through_an_explicit_scoring_rules_payload(stub_chat):
    payload = PRESET_STANDARD.model_dump()
    client.post("/chat", json={
        "message": "How's Jordan Love looking?",
        "mentioned_players": ["Jordan Love"],
        "scoring_rules": payload,
    })
    assert stub_chat["scoring_rules"] == ScoringRules(**payload)
```

Update `stub_chat`'s `fake_chat` signature to also capture `season`/`scoring_rules`:

```python
@pytest.fixture
def stub_chat(monkeypatch):
    calls = {}

    def fake_chat(message, mentioned_players=None, season=None, week=None, history=None, scoring_rules=None, **_):
        calls.update(
            message=message, mentioned_players=mentioned_players, season=season,
            week=week, history=history, scoring_rules=scoring_rules,
        )
        return _fixture_chat_result()

    monkeypatch.setattr("api.main.chat", fake_chat)
    return calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_api.py -v`
Expected: FAIL (`/scoring-rules` doesn't exist; `ChatRequest` rejects `scoring_rules`)

- [ ] **Step 3: Write the implementation**

Add imports:

```python
from typing import Annotated, Literal

from llm.interface import parse_custom_scoring_rules
from pipeline.scoring import PRESET_HALF_PPR, PRESET_PPR, PRESET_STANDARD, ScoringRules
```

Add the preset lookup and request/response models:

```python
_SCORING_PRESETS = {"ppr": PRESET_PPR, "half_ppr": PRESET_HALF_PPR, "standard": PRESET_STANDARD}


class ScoringRulesRequest(BaseModel):
    base: Literal["ppr", "half_ppr", "standard", "custom"]
    base_hint: Literal["ppr", "half_ppr", "standard"] = "ppr"
    custom_description: Annotated[str, StringConstraints(max_length=MAX_QUESTION_LENGTH)] | None = None


class ScoringRulesResponse(BaseModel):
    rules: ScoringRules
```

Add the endpoint (rate-limited like `/chat`, since `base == "custom"` also costs a paid LLM call):

```python
@app.post("/scoring-rules", response_model=ScoringRulesResponse)
@limiter.limit(CHAT_RATE_LIMIT)
def scoring_rules_endpoint(request: Request, rules_request: ScoringRulesRequest) -> ScoringRulesResponse:
    """Resolve a scoring tier to its weights - a preset directly, or a Custom
    description parsed once by the LLM. The frontend saves the *result* to
    localStorage and sends it with every /chat call, so this never runs per
    chat message (README §14 / custom-league-scoring design spec).
    """
    if rules_request.base != "custom":
        return ScoringRulesResponse(rules=_SCORING_PRESETS[rules_request.base])
    if not rules_request.custom_description:
        raise HTTPException(
            status_code=400, detail="custom_description is required when base is 'custom'."
        )
    rules = parse_custom_scoring_rules(
        _SCORING_PRESETS[rules_request.base_hint], rules_request.custom_description
    )
    return ScoringRulesResponse(rules=rules)
```

Update `ChatRequest`:

```python
class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(max_length=MAX_QUESTION_LENGTH)]
    mentioned_players: list[Annotated[str, StringConstraints(max_length=MAX_PLAYER_NAME_LENGTH)]] = (
        Field(default_factory=list, max_length=MAX_MENTIONED_PLAYERS)
    )
    season: int | None = Field(
        default=None, description="Defaults to the season the processed data targets"
    )
    week: int | None = Field(
        default=None, description="Defaults to the week the processed data describes"
    )
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    scoring_rules: ScoringRules | None = Field(
        default=None, description="Defaults to full PPR when omitted"
    )
```

Update `chat_endpoint`'s call to `chat()`:

```python
result = chat(
    chat_request.message,
    mentioned_players=chat_request.mentioned_players,
    season=chat_request.season,
    week=chat_request.week,
    history=[turn.model_dump() for turn in chat_request.history],
    scoring_rules=chat_request.scoring_rules,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat: add POST /scoring-rules and thread scoring_rules through /chat"
```

---

### Task 10: Frontend — scoring settings toggle, `localStorage`, wired into `/chat`

**Files:**
- Create: `frontend/src/ScoringSettings.jsx`
- Modify: `frontend/src/api.js`
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/App.css` (minimal styling for the new control)

**Interfaces:**
- Produces: `fetchScoringRules({base, baseHint, customDescription}) -> Promise<{rules}>`, `fetchChat` gains `season`/`scoringRules` params, `loadStoredScoringRules()`/`loadStoredScoringBase()` (exported from `ScoringSettings.jsx`), `<ScoringSettings onRulesChange={fn} />`.

No frontend test suite exists in this repo (consistent with the chat-assistant plan) — this task is verified manually via `npm run dev` (Step 4).

- [ ] **Step 1: Add `fetchScoringRules` and extend `fetchChat` in `frontend/src/api.js`**

```js
export async function fetchScoringRules({ base, baseHint, customDescription }) {
  const response = await fetch(`${API_BASE_URL}/scoring-rules`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      base,
      base_hint: baseHint ?? 'ppr',
      custom_description: customDescription ?? null,
    }),
  })
  if (!response.ok) {
    throw new Error(await parseErrorDetail(response))
  }
  return response.json()
}

export async function fetchChat({ message, mentionedPlayers, week, history, scoringRules }) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      mentioned_players: mentionedPlayers ?? [],
      week: week ?? null,
      history: history ?? [],
      scoring_rules: scoringRules ?? null,
    }),
  })
  if (!response.ok) {
    throw new Error(await parseErrorDetail(response))
  }
  return response.json()
}
```

- [ ] **Step 2: Create `frontend/src/ScoringSettings.jsx`**

```jsx
import { useState } from 'react'
import { fetchScoringRules } from './api'

const STORAGE_RULES_KEY = 'ffw_scoring_rules'
const STORAGE_BASE_KEY = 'ffw_scoring_base'
const LABELS = { ppr: 'PPR', half_ppr: 'Half-PPR', standard: 'Standard', custom: 'Custom' }

export function loadStoredScoringRules() {
  try {
    const raw = localStorage.getItem(STORAGE_RULES_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

function loadStoredScoringBase() {
  return localStorage.getItem(STORAGE_BASE_KEY) || 'ppr'
}

function ScoringSettings({ onRulesChange }) {
  const [base, setBase] = useState(loadStoredScoringBase())
  const [baseHint, setBaseHint] = useState('ppr')
  const [description, setDescription] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  async function save(nextBase) {
    setSaving(true)
    setError(null)
    try {
      const { rules } = await fetchScoringRules({
        base: nextBase,
        baseHint,
        customDescription: nextBase === 'custom' ? description : null,
      })
      localStorage.setItem(STORAGE_RULES_KEY, JSON.stringify(rules))
      localStorage.setItem(STORAGE_BASE_KEY, nextBase)
      setBase(nextBase)
      onRulesChange(rules)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="scoring-settings">
      <span className="scoring-settings-label">Scoring</span>
      <div className="scoring-options">
        {Object.keys(LABELS).map((option) => (
          <button
            key={option}
            type="button"
            className={base === option ? 'scoring-option active' : 'scoring-option'}
            onClick={() => (option === 'custom' ? setBase('custom') : save(option))}
            disabled={saving}
          >
            {LABELS[option]}
          </button>
        ))}
      </div>
      {base === 'custom' && (
        <div className="scoring-custom">
          <select value={baseHint} onChange={(e) => setBaseHint(e.target.value)}>
            <option value="ppr">Based on PPR</option>
            <option value="half_ppr">Based on Half-PPR</option>
            <option value="standard">Based on Standard</option>
          </select>
          <textarea
            placeholder="Describe how your league differs, e.g. 'catches are 0.5 points'"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <button type="button" onClick={() => save('custom')} disabled={saving || !description}>
            Save
          </button>
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  )
}

export default ScoringSettings
```

- [ ] **Step 3: Wire it into `frontend/src/App.jsx`**

```jsx
import { useEffect, useRef, useState } from 'react'
import { fetchChat, fetchPlayers } from './api'
import ChatInput from './ChatInput'
import MessageBubble from './MessageBubble'
import ScoringSettings, { loadStoredScoringRules } from './ScoringSettings'
import './App.css'

function App() {
  const [players, setPlayers] = useState([])
  const [playersError, setPlayersError] = useState(null)
  const [week, setWeek] = useState('')
  const [scoringRules, setScoringRules] = useState(loadStoredScoringRules())
  const [messages, setMessages] = useState([])
  const [submitting, setSubmitting] = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    fetchPlayers().then(setPlayers).catch((err) => setPlayersError(err.message))
  }, [])

  useEffect(() => {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    scrollRef.current?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' })
  }, [messages])

  async function handleSend(message, mentionedPlayers) {
    const history = messages.map(({ role, content }) => ({ role, content }))
    setMessages((prev) => [...prev, { role: 'user', content: message }])
    setSubmitting(true)
    try {
      const response = await fetchChat({
        message, mentionedPlayers,
        week: week ? Number(week) : null,
        history, scoringRules,
      })
      setMessages((prev) => [...prev, {
        role: 'assistant', content: response.answer, sources: response.sources,
        recommendation: response.recommendation, context: response.context,
      }])
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: err.message, error: true }])
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="app">
      <header>
        <div className="header-title">
          <h1>Fantasy Football Wizard</h1>
          <p className="subtitle">Ask about start/sit, trades, or any player.</p>
        </div>
        <ScoringSettings onRulesChange={setScoringRules} />
        <label className="week-label" title="Optional — defaults to the upcoming week">
          Week
          <input
            type="number" min="1" max="22" placeholder="upcoming"
            value={week} onChange={(e) => setWeek(e.target.value)}
          />
        </label>
      </header>

      {playersError && <p className="error">Couldn't load players: {playersError}</p>}

      <div className="message-list">
        {messages.map((message, i) => <MessageBubble key={i} {...message} />)}
        <div ref={scrollRef} />
      </div>

      <ChatInput players={players} onSend={handleSend} disabled={submitting} />
    </div>
  )
}

export default App
```

- [ ] **Step 4: Manual verification**

Run: `cd frontend && npm run dev` (with the backend running separately per the README's local-dev instructions).
Expected: the header shows a "Scoring" control with PPR/Half-PPR/Standard/Custom buttons. Clicking a preset immediately updates `localStorage` (verify in DevTools → Application → Local Storage → `ffw_scoring_rules`). Selecting Custom reveals the base-hint dropdown and description box; Save calls the backend once and stores the resolved weights. Refreshing the page keeps the previously-saved tier active (button stays highlighted, `localStorage` still populated). Sending a chat message includes `scoring_rules` in the network request payload (verify in DevTools → Network).

- [ ] **Step 5: Add minimal styling to `frontend/src/App.css`**

```css
.scoring-settings {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.scoring-options {
  display: flex;
  gap: 0.25rem;
}
.scoring-option {
  padding: 0.25rem 0.6rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  background: transparent;
  cursor: pointer;
}
.scoring-option.active {
  background: #2b6cb0;
  color: white;
  border-color: #2b6cb0;
}
.scoring-custom {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  max-width: 320px;
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/ScoringSettings.jsx frontend/src/api.js frontend/src/App.jsx frontend/src/App.css
git commit -m "feat: add scoring settings toggle to the frontend"
```

---

### Task 11: Full regression, live end-to-end verification, docs update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Run the full backend test suite and lint**

Run: `source .venv/bin/activate && python -m pytest -q && ruff check .`
Expected: all tests pass; no new lint errors (the 2 pre-existing unrelated ones from before this branch, if still present, are out of scope).

- [ ] **Step 2: Regenerate real data and verify the new pipeline end-to-end**

```bash
source .venv/bin/activate
python -m scripts.refresh_stats
python3 -c "
from pipeline.context_builder import build_context
from pipeline.scoring import PRESET_PPR, PRESET_STANDARD, PRESET_HALF_PPR
import polars as pl

meta = pl.read_parquet('data/processed/meta.parquet').row(0, named=True)
season, week = meta['season'], meta['week']
for label, rules in [('PPR', PRESET_PPR), ('Half-PPR', PRESET_HALF_PPR), ('Standard', PRESET_STANDARD)]:
    print(f'--- {label} ---')
    print(build_context(['Jordan Love'], season=season, week=week, scoring_rules=rules))
    print()
"
```
Expected: three context blocks with genuinely different `season avg`/`avg over last 3 games` numbers per tier (PPR > Half-PPR > Standard, assuming Love has any receptions - if not, pick a receiver from that week's real data instead).

- [ ] **Step 3: Live-verify the full `/scoring-rules` + `/chat` flow, including Custom**

```bash
source .venv/bin/activate && uvicorn api.main:app --reload &
sleep 2
curl -s -X POST http://localhost:8000/scoring-rules -H 'Content-Type: application/json' \
  -d '{"base": "custom", "base_hint": "ppr", "custom_description": "catches are worth 0.5 points instead of 1"}' | python3 -m json.tool
kill %1
```
Expected: a full `ScoringRules` JSON object with `receptions: 0.5` and every other field matching `PRESET_PPR`'s values unchanged. If the LLM changes an unrelated field, that's a real bug — tighten `parse_custom_scoring_rules`'s prompt (Task 8) and re-verify, don't proceed with a broken parse.

- [ ] **Step 4: Update `CLAUDE.md`**

Update the Phase 1/Phase 2 status lines (currently describing the `_ppr`-suffixed field approach from PR #11) to describe the new architecture: request-time scoring computation, the four tiers, `pipeline/scoring.py`, `pipeline/player_form.py`, and the `data/processed/meta.parquet` artifact. Add a line noting `POST /scoring-rules` under the chat-assistant/API section.

- [ ] **Step 5: Update `README.md`**

Update §7's "Example Context" to show the new field names (no `_ppr` suffix) and add a one-line note that the figures reflect whichever scoring tier was selected (replacing the PR #11 note that said "always PPR"). Add a short new subsection documenting the scoring tiers and `POST /scoring-rules`.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document custom league scoring"
```

---

## Self-Review Notes

- **Spec coverage:** every section of the design spec (scoring.py, player_form.py, refresh_stats.py rewrites for both player_stats and projections, context_builder rewrite, chat_engine/decision_engine threading, LLM parsing, API endpoint, frontend) maps to a task above.
- **Gap found and fixed during planning (not in the original spec):** `build_context` needs a `season` parameter, and a new `data/processed/meta.parquet` artifact is needed to resolve it — the old trick of reading `max(week)` off `player_stats.parquet` breaks once that table holds many historical rows instead of one row per player for a single target week. This is threaded through Tasks 3, 5, 6, 7, and 9.
- **Type/name consistency verified:** `ScoringRules` field names are identical across Task 1 (definition), Task 3/4 (staging renames), Task 6 (context_builder), Task 8 (LLM parsing), and Task 9 (API) - all 13 fields, same names, everywhere.
- **No placeholders:** every step has real code, real assertions, or a real shell command.
