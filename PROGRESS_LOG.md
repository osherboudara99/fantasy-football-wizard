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
