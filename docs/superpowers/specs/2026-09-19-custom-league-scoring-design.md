# Custom league scoring — design

**Status**: approved for implementation
**Date**: 2026-09-19

## Summary

The app currently hard-codes full-PPR scoring everywhere (`fantasy_points_ppr`
from nflverse for historical stats, `pts_ppr` from Sleeper for projections).
Add a scoring-rules toggle — **PPR / Half-PPR / Standard / Custom** — so every
number the agent shows or reasons over (recent form, season averages,
projections) reflects the scoring format the user actually plays under.
Custom lets the user describe modifications on top of one of the three base
tiers in their own words (e.g. "PPR, but catches are 0.5"); the description is
parsed into structured weights once, not on every chat message.

The scoring choice is a value sent by the frontend with each request (stored
client-side), not a server-side global — this app has no auth or per-user
accounts today (CLAUDE.md's Project Goal: a personal tool, not a multi-tenant
product), so a single global config would leak across any two visitors of the
same deployment. Keeping the setting client-held avoids that without adding
auth.

## Why

The user's own league scores a non-standard rule (1 point per carry) that no
built-in PPR tier represents, and pointed out mid-discussion that generic PPR
math actively produces wrong advice for their real decisions — the whole
point of this app is decisions the user will act on in their real league. A
live ESPN/Sleeper league-API connector was considered and rejected for this
specific problem: it would solve a different thing (auto-discovering rosters
and matchups), and requires OAuth/session-based auth for ESPN in particular.
Once the user is willing to state their own rules, no external league API is
needed — the agent just needs to compute fantasy points itself from raw
per-game stat counts nflverse/Sleeper already provide, instead of trusting a
fixed formula baked into someone else's `fantasy_points`/`pts_ppr` column.

## Architecture

### Today

- `scripts/refresh_stats.py`'s `build_processed_player_stats()` pre-aggregates
  one row per player at **refresh time**: `avg_fantasy_points_ppr_last3`,
  `avg_fantasy_points_ppr_season`, `prior_season_*_ppr`, `last_game_*_ppr` —
  all derived from nflverse's fixed `fantasy_points_ppr` column.
- `build_processed_projections()` keeps a single `projected_points` column,
  aliased directly from Sleeper's `pts_ppr`.
- `pipeline/context_builder.py`'s `_format_recent_form()` reads those
  pre-aggregated PPR columns directly; there is no notion of "which scoring
  format" anywhere in the pipeline.

This doesn't extend to a per-request scoring choice: pre-aggregating "the"
fantasy-point average at refresh time assumes one fixed formula. Precomputing
four (or infinite, for Custom) versions nightly doesn't scale and still
wouldn't cover arbitrary Custom weights decided after the last refresh.

### New

**One computation path for all four tiers.** PPR/Half-PPR/Standard are just
specific weight values; Custom is the same math with different weights. Move
fantasy-point computation from refresh time to **request time**, over raw
per-game counting stats, computed only for the 1–3 players actually being
discussed (verified cheap: 20,612 staged rows / 0.35 MB total across two
seasons today — filtering to a few players and summing weighted columns in
polars is single-digit milliseconds, noise next to the LLM call that already
dominates `/chat` latency).

#### `pipeline/scoring.py` (new)

```python
class ScoringRules(BaseModel):
    pass_yards: float = 0.0          # points per passing yard
    pass_tds: float = 0.0
    pass_interceptions: float = 0.0  # negative to penalize
    pass_2pt: float = 0.0
    rush_yards: float = 0.0          # points per rushing yard
    rush_tds: float = 0.0
    rush_2pt: float = 0.0
    rush_attempts: float = 0.0       # points per carry (0 in every built-in preset;
                                      # exists so Custom can express rules like
                                      # "1 point per handoff")
    receptions: float = 0.0          # the PPR knob: 0 / 0.5 / 1.0 across the 3 presets
    rec_yards: float = 0.0           # points per receiving yard
    rec_tds: float = 0.0
    rec_2pt: float = 0.0
    fumbles_lost: float = 0.0        # negative to penalize

PRESET_STANDARD = ScoringRules(
    pass_yards=0.04, pass_tds=4, pass_interceptions=-2, pass_2pt=2,
    rush_yards=0.1, rush_tds=6, rush_2pt=2,
    rec_yards=0.1, rec_tds=6, rec_2pt=2,
    fumbles_lost=-2,
)
PRESET_HALF_PPR = PRESET_STANDARD.model_copy(update={"receptions": 0.5})
PRESET_PPR = PRESET_STANDARD.model_copy(update={"receptions": 1.0})

def compute_fantasy_points(row: dict, rules: ScoringRules) -> float:
    """Weighted sum of row[field] * rules.field over every ScoringRules field
    present in `row` (missing/null fields contribute 0 - a QB row has no
    `receptions` key and that's fine).
    """
```

Every field name in `ScoringRules` is also the **canonical column name** used
downstream — nflverse's and Sleeper's native column names are renamed to
these at the staging step (see below), so `compute_fantasy_points` never
needs to know which provider a row came from:

| `ScoringRules` field | nflverse historical column   | Sleeper projected key |
|-----------------------|-------------------------------|------------------------|
| `pass_yards`          | `passing_yards`               | `pass_yd`             |
| `pass_tds`            | `passing_tds`                 | `pass_td`             |
| `pass_interceptions`  | `passing_interceptions`       | `pass_int`            |
| `pass_2pt`            | `passing_2pt_conversions`     | `pass_2pt`            |
| `rush_yards`          | `rushing_yards`               | `rush_yd`             |
| `rush_tds`            | `rushing_tds`                 | `rush_td`             |
| `rush_2pt`            | `rushing_2pt_conversions`     | `rush_2pt`            |
| `rush_attempts`       | `carries`                     | `rush_att`            |
| `receptions`          | `receptions`                  | `rec`                 |
| `rec_yards`           | `receiving_yards`             | `rec_yd`              |
| `rec_tds`             | `receiving_tds`               | `rec_td`              |
| `rec_2pt`             | `receiving_2pt_conversions`   | `rec_2pt`             |
| `fumbles_lost`        | `fumbles_lost_total`          | `fum_lost`            |

Both source columns were confirmed present in live data during design
(nflverse's raw `player_stats` and a live Sleeper `/projections` call for a
WR and a QB sample).

#### `scripts/refresh_stats.py` changes

- `build_staged_player_stats()`: select the additional raw columns above
  (renamed to the canonical names) alongside what's already selected
  (`targets`, `receptions`→kept as-is since it matches, `carries`, snaps,
  xFP, NGS). Drop reliance on `fantasy_points`/`fantasy_points_ppr` for
  anything downstream — keep the columns in the staged table only as a
  regression-test reference (see Testing), never read by processed/context
  code.
- `build_processed_player_stats()` **simplifies**: it no longer aggregates.
  It filters to `_completed_weeks()` (unchanged) and the current
  roster/projection universe (unchanged fix from PR #11), and returns **one
  row per player per completed game** with the canonical raw-stat columns
  plus `player_id`, `player_name`, `position`, `team`, `season`, `week`. All
  of the `_own_last_n`/aggregation logic that used to live here moves to
  `pipeline/player_form.py` (below), since it now runs at request time.
- `build_staged_projections()` / `build_processed_projections()`: keep the
  raw per-category projected columns (renamed to canonical names) instead of
  collapsing to one `projected_points` column. `pts_ppr`/`pts_half_ppr`/
  `pts_std` are dropped from the processed table — they're not needed once
  `compute_fantasy_points` can derive any tier (including the three presets)
  from the same raw counts used for historical stats, keeping one formula
  path instead of "use the provider's number for presets, compute for
  Custom."

#### `pipeline/player_form.py` (new)

Houses the per-player aggregation that used to run at refresh time,
parameterized by `ScoringRules`:

```python
def compute_recent_form(
    player_rows: pl.DataFrame,  # one player's rows from processed player_stats
    season: int, week: int, rules: ScoringRules,
) -> dict:
    """Returns the same field shape context_builder already renders:
    games_played_this_season, avg_fantasy_points_last3, avg_fantasy_points_season,
    prior_season_games_played, prior_season_avg_fantasy_points,
    prior_season_last3_avg_fantasy_points, last_game_season, last_game_week,
    last_game_fantasy_points - each fantasy-point number computed via
    compute_fantasy_points(row, rules) per game row, then averaged.
    """
```

`_own_last_n` (own-last-N-games-per-player, unchanged logic from PR #11)
moves here from `refresh_stats.py`.

#### `pipeline/context_builder.py` changes

- `build_context(players, week, tables=None, news_fn=None, scoring_rules=None)`
  — `scoring_rules` defaults to `PRESET_PPR` (today's behavior, unchanged for
  any caller that doesn't pass one).
- `_format_player()` filters `player_stats` to the named player's `player_id`
  across all rows (not a single week-keyed row), calls
  `player_form.compute_recent_form(...)`, and renders the exact same
  season-separated format from PR #11 (`_format_recent_form` keeps its
  branching logic — 0 / 1-2 / 3+ this-season games — but reads from the dict
  `compute_recent_form` returns instead of pre-aggregated table columns).
- Projection lookup: `compute_fantasy_points(proj_row, scoring_rules)`
  instead of reading a `projected_points` column.
- Field names drop the `_ppr` suffix added in PR #11
  (`avg_fantasy_points_last3`, not `avg_fantasy_points_ppr_last3`) — there's
  now exactly one set of numbers, computed under whichever rules were
  selected, not parallel std/PPR columns.

#### `pipeline/chat_engine.py` / `pipeline/decision_engine.py`

Both gain a `scoring_rules: ScoringRules | None = None` parameter, threaded
straight through to `build_context()`.

#### `llm/interface.py` — Custom parsing

New function, parses a free-text description into a full resolved
`ScoringRules`, seeded from one of the three presets:

```python
def parse_custom_scoring_rules(base: ScoringRules, description: str) -> ScoringRules:
    """One LLM call (client.messages.parse, output_format=ScoringRules).
    Prompt gives the model `base`'s weights as JSON plus `description`, and
    instructs it to return the full ScoringRules with only the described
    categories changed - every other field stays at `base`'s value.
    """
```

This runs **once**, at settings-save time, not per chat message.

#### `api/main.py`

New endpoint:

```
POST /scoring-rules
Request:  { "base": "ppr" | "half_ppr" | "standard" | "custom",
            "base_hint": "ppr" | "half_ppr" | "standard" | null,
            "custom_description": str | null }
Response: { "rules": ScoringRules }
```

For `base != "custom"`, `base_hint`/`custom_description` are ignored and the
matching preset is returned directly — no LLM call. For `base == "custom"`,
`base_hint` selects which preset seeds the description (defaults to `"ppr"`
if omitted — the app's long-standing default) and `custom_description` is
required; the endpoint calls
`parse_custom_scoring_rules(PRESET_<base_hint>, custom_description)`.

`ChatRequest` (existing) gains `scoring_rules: ScoringRules | None = None`.
Light validation on `ScoringRules` fields (`Field(ge=-10, le=10)`) guards
against garbage values reaching the LLM context as a giant multiplier.

#### Frontend

New settings control (toggle: PPR / Half-PPR / Standard / Custom; Custom
reveals a base-hint sub-choice + a text field for the description, with a
"Save" action that calls `POST /scoring-rules` once). The **resolved**
`ScoringRules` object (never the raw description) is stored in
`localStorage` and attached to every `/chat` request. No setting saved yet →
omit the field → backend defaults to PPR, matching current behavior.

## Error handling

- `POST /scoring-rules` with `base == "custom"` and no `custom_description` →
  400.
- LLM output validation failure during custom parsing → 400, frontend keeps
  the previously-saved rules (or the PPR default) rather than silently
  clearing the setting.
- `ScoringRules` field out of the `[-10, 10]` range on `/chat` → 422 (Pydantic
  validation, standard FastAPI behavior) — the frontend never constructs an
  out-of-range payload itself, so this only fires on a malformed direct API
  call.

## Testing

- `tests/test_scoring.py` (new): `compute_fantasy_points` against fixture
  rows with hand-computed expected totals for each preset; a **consistency
  check** comparing `compute_fantasy_points(row, PRESET_STANDARD)` /
  `PRESET_PPR` against nflverse's own `fantasy_points`/`fantasy_points_ppr`
  columns on the same fixture rows (documents that this reimplements the
  well-known standard formula, not an approximation) — allow a small
  tolerance rather than requiring bit-exact equality, since nflverse's own
  fumble/2pt handling isn't publicly guaranteed identical.
- `tests/test_refresh_stats.py`: `build_processed_player_stats` tests rewrite
  to assert the new per-game-row shape (this is the fourth revision of this
  function this week; keep the existing season-boundary/roster-filter test
  *behavior*, just on the new output shape).
- `tests/test_context_builder.py`: `_format_recent_form`/`_format_player`
  tests rewrite to build fixture per-game rows and pass an explicit
  `scoring_rules` (mostly `PRESET_PPR`, plus one test each for
  `PRESET_STANDARD`/`PRESET_HALF_PPR` to prove the tier actually changes the
  rendered numbers).
- `tests/test_chat_engine.py` / `tests/test_decision_engine.py`: add a
  `scoring_rules` passthrough test each.
- `tests/test_api.py`: new tests for `POST /scoring-rules` (each preset,
  custom with a stubbed `parse_custom_scoring_rules`, missing description
  400) and `/chat` accepting/defaulting `scoring_rules`.
- Frontend: manual verification via dev server (no frontend test suite
  exists today, consistent with current project conventions per the chat
  assistant spec).

## Non-goals (this round)

- **Kicker/DST scoring** — nflverse's data has clean distance-bucketed kicker
  columns (`fg_made_20_29`, etc.) so this is plausible later, but it's a
  different (non-linear, bucketed) rule shape than the linear per-unit model
  here. Deferred.
- **Ambient mid-chat scoring detection** — a scoring change only takes effect
  through the explicit `POST /scoring-rules` settings flow, never by parsing
  an offhand mention inside an unrelated start/sit question.
- **Multi-user auth / per-account settings** — the scoring choice is a
  client-held value sent per request, not server-side global state; this
  avoids the cross-user leak concern without adding accounts. If the app
  ever becomes genuinely multi-user, that's a separate, larger project.
- **One-off "what if" exploration without changing the saved setting** — not
  built; changing the toggle changes the standing setting.
- Draft strategy/ADP, waiver-wire optimization, DFS/betting advice,
  full-season simulations — unaffected, still permanently out of scope.

## Rollout

Implement fully (data pipeline + new modules + API + frontend), verify
locally end-to-end (a real `/chat` question under each of the four tiers,
confirming the numbers actually change), then resume Phase 7 deployment work
— unaffected by this change beyond needing a fresh data refresh once live.
