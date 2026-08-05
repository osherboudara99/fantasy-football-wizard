# Progress Log

Append-only, one entry per completed phase.

## 2026-07-19 — Phase 2: Context builder

- Shipped `pipeline/entity_extraction.py` (regex player-name/week parsing against
  the known-names universe from `data/processed/player_stats.parquet`) and
  `pipeline/context_builder.py` (`build_context(players, week)` assembling the
  §7 PLAYER COMPARISON block from `player_stats`/`projections`/`injuries`
  processed Parquet). "Recent news" bullet from §7's example is intentionally
  omitted until Phase 6 (RAG) wires it in — documented in the module docstring.
- Done-check: PASS — `build_context(["Player A","Player B"], week)` returns the
  §7 format; verified both via `tests/test_context_builder.py` (13 tests) and a
  manual fixture-data call by an independent checker subagent.
- Tests: `uv run python -m pytest -q` → 13 passed. Lint: `uv run ruff check .`
  → clean. (Note: bare `uv run pytest` fails to import `pipeline`/`scripts` —
  no conftest.py/package on sys.path; use `python -m pytest` instead.)
- Checker pass: independent subagent confirmed done-check + ran a medium-effort
  code review. No blocking bugs. Two non-blocking notes for future hardening:
  (1) `_format_player` has no null guard before `:.1f` formatting on
  avg/projected points; (2) `injuries.parquet` has no `week` column, so
  `_format_injury` matches on player name only — harmless today because all
  three processed tables refresh together, but latent if injuries are ever
  refreshed on their own cadence (README §11).
- Approx LLM $ spent this phase: $0 (no LLM calls — regex/Pandas only).

## 2026-08-05 — Phase 3: Decision engine end-to-end

- Shipped `pipeline/decision_engine.py`: `decide(question, players=None, week=None)`
  resolves players/week (explicit args → entity extraction → data's latest week),
  builds the §7 context, calls `run_llm()`, and returns a `Decision` (question,
  players, week, context, recommendation). CLI:
  `python -m pipeline.decision_engine "Should I start X or Y in week 18?" [--show-context]`.
- Done-check: PASS — real terminal run on 2025 week 18 data
  ("T.J. Hockenson or Chris Godwin Jr.") returned START Godwin / BENCH Hockenson
  at 98% confidence, citing Hockenson's Out status; independently re-run by a
  fresh checker subagent with its own player pair.
- Tests: `python -m pytest -q` → 30 passed. Lint: `ruff check .` → clean.
- Checker pass found four real defects, all fixed before marking the phase done:
  1. Sleeper ships ~20% of its `gsis_id`s whitespace-padded, so the full join in
     `build_staged_injuries` split those players into two rows — one real report
     plus a Sleeper row defaulting to "Healthy". T.J. Hockenson (Out) rendered as
     "Healthy". Fixed by stripping the ids at staging.
  2. `context_builder` joined projections/injuries by `player_name` while
     `player_id` sits on all three tables — nflverse display names ("Kenneth
     Walker III") don't match Sleeper's ("Kenneth Walker"), silently dropping 86
     players' projections. Now keyed on `player_id`, and a real injury report
     always outranks a "Healthy" duplicate.
  3. `week 0` in a question is falsy, so `extract_week(...) or latest_week()`
     silently answered for a different week. Split into `resolve_week()`.
  4. Nothing checked the LLM's answer against the question — a hallucinated name
     or 187% confidence rendered fine. Added `_check_recommendation()` plus
     `ge=0, le=1` on `Recommendation.confidence`.
- Known limitation (not fixed, flagged to the user): `latest_week()` returns the
  most recently *completed* week, because that's what `refresh_stats.py` resolves
  and fetches projections for. In-season this answers about a game already
  played; asking for the upcoming week raises `PlayerNotFoundError`. Fixing it
  means changing Phase 1's week resolution — a scope decision, not a Phase 3 bug.
- Approx LLM $ spent this phase: ~$0.02 (3 real `claude-haiku-4-5` calls: 2 by
  the builder, 1 by the checker).
