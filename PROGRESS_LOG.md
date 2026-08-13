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
- Approx LLM $ spent this phase: ~$0.02 (3 real `claude-haiku-4-5` calls: 2 by
  the builder, 1 by the checker).

## 2026-08-05 — Follow-up: week resolution now targets the upcoming week

Raised by the Phase 3 checker as a known limitation, then fixed on user request
(same branch, before opening the PR).

- Problem: `refresh_stats.resolve_season_week()` resolved the most recently
  *completed* week, so the whole pipeline described a game already played —
  recent-form averages included the very week being "projected", and asking about
  the upcoming week raised `PlayerNotFoundError`. Backwards for a start/sit tool.
- Fix: `resolve_target_week()` picks the earliest regular-season week with an
  unplayed game (mid-week that's the week in progress; in the off-season it's week
  1 of the next season). `build_processed_player_stats()` now aggregates over weeks
  strictly *before* the target, chronologically rather than numerically, so an
  early-season target carries recent form over from the prior season;
  `season_type == "REG"` is filtered explicitly so POST weeks 19+ can't sort ahead
  of a new season's week 1. `stats_seasons()` pulls the prior season only when the
  window actually reaches back. `decision_engine.latest_week()` → `target_week()`.
- Also needed: nflverse publishes nothing for a season until it starts, so
  targeting the upcoming week 404s (or trips nflreadpy's season-range guard) on
  every per-season table. `load_by_season()` loads season by season and skips the
  unpublished ones; the official injury report falls back to an empty, correctly
  typed table (Sleeper's live dump still carries `injury_status`), and schedules
  load via `seasons=True` + filter since they're published before a season starts.
- Verified with a real refresh on 2026-08-05: target resolved to 2026 week 1,
  form carried from 2025 weeks 16-18, 946 non-null Sleeper projections for the
  upcoming week, injuries from Sleeper (81 Questionable, 18 PUP, 2 IR). End-to-end
  CLI run with no `--week`: "Bijan Robinson or Jahmyr Gibbs?" → Week 1 comparison,
  START Bijan at 72%.
- Tests: `python -m pytest -q` → 33 passed. Lint: `ruff check .` → clean.
- Approx LLM $ spent: ~$0.01 (1 real `claude-haiku-4-5` call).

## 2026-08-05 — Phase 4: FastAPI backend

- Shipped `api/main.py`: `POST /recommendation` (two players + optional week and
  free-text question → the `Recommendation` fields plus the resolved week, players,
  and the context the answer was built from, for README §10's debug view),
  `GET /players` for the frontend picker, and `GET /health` for Cloud Run's probe.
  `.env` is loaded at startup and CORS origins come from `FRONTEND_ORIGINS`
  (default: Vite on localhost:5173). The API owns no logic — it validates input,
  calls `decide()`, and maps pipeline errors to status codes.
- Done-check: PASS — `curl -X POST localhost:8000/recommendation -d
  '{"players":["Bijan Robinson","Jahmyr Gibbs"]}'` returned 200 with a valid
  recommendation JSON; independently re-run by a fresh checker subagent with its
  own player pair, which also confirmed the returned `context` matches what
  `build_context()` produces byte-for-byte.
- Tests: `python -m pytest -q` → 48 passed. Lint: `ruff check .` → clean.
- Checker pass found four real defects, all fixed:
  1. `question` was unbounded and goes straight into the prompt — a 520K-char
     question reached the LLM (~130K input tokens, ~$0.13 for one request, ~26x
     the per-recommendation budget). Now capped at 500 chars, names at 100.
  2. `["X", "X"]` passed validation, spent a paid call, and could return
     `start == bench` — `_check_recommendation` compares *sets*, so a one-element
     set matched a one-element set. `resolve_players` now requires two distinct
     names (fixes the CLI path too).
  3. A failed refresh surfaced as `400` echoing a server filesystem path and a
     runbook command. Added `DataUnavailableError` → `503` with a generic message,
     so outages are server errors and alerting can see them.
  4. Tests that asserted status codes without proving the pipeline wasn't entered,
     and a 400 test that never checked the detail — tightened.
- Verified clean by the checker: no secret ever reaches a response body (including
  a simulated provider error carrying a fake key — response was a bare 500), CORS
  is not accidentally permissive (`allow_credentials` stays False), and endpoints
  are sync `def` so blocking parquet/LLM work runs in the threadpool.
- Known gap, deliberate: **no auth or rate limiting**. Harmless locally, but
  `POST /recommendation` spends the owner's API credits per call, so this must be
  addressed in Phase 7 before the service gets a public URL. `/docs`, `/redoc`,
  and `/openapi.json` are open too (no secrets in them, but they advertise the
  paid endpoint's schema).
- Approx LLM $ spent this phase: ~$0.02 (2 real `claude-haiku-4-5` calls: 1 by the
  builder, 1 by the checker; all input-bound tests reject before the LLM).

## 2026-08-06 — Phase 5: React frontend

- Shipped `frontend/`: Vite + React (JS, not TS) app. `src/api.js` wraps
  `GET /players` and `POST /recommendation` (`VITE_API_BASE_URL` env var,
  defaults to `http://localhost:8000`, matching the FastAPI CORS default
  origin `http://localhost:5173`). `src/App.jsx` holds two `<select>` player
  pickers (each disables the other's current pick so the two players can't
  match), an optional free-text question (500-char cap, mirrors the API's
  limit) and an optional week override, and calls the API on submit.
  `src/ResultCard.jsx` renders start/bench, a confidence bar, key/risk
  factor lists, and a collapsible `<details>` debug view of the raw context
  string the recommendation was built from (README §10's optional debug
  view). Removed the Vite template's default counter/hero boilerplate.
- Done-check: PASS — `npm run build` and `npx oxlint` both clean; a real
  `POST /recommendation` round-trip through the browser-facing CORS path
  (`Origin: http://localhost:5173`) returned a valid recommendation matching
  `ResultCard.jsx`'s expected shape byte-for-byte. The Claude-in-Chrome
  browser extension wasn't connected this session, so the visual pass was
  done by the user directly in a browser tab (both dev servers running
  locally) rather than by an automated agent — confirmed working.
- Tests: no JS test framework was added (none existed in the repo to match,
  and the done-check is a manual end-to-end flow, not unit-testable
  business logic — the frontend has none, by design). `npm run build`
  clean, `npx oxlint` clean. Backend suite unaffected:
  `python -m pytest -q` → 48 passed, `ruff check .` → clean.
- User feedback after the done-check passed (2026-08-06), explicitly
  deferred until after Phase 7 deployment rather than reworked now:
  1. The `<select>` player pickers don't scale — too many players to scan;
     wants a searchable/typeahead input instead.
  2. Wants defense/DST options so team defenses can be compared, not just
     offensive skill players — no current data source in the pipeline is
     wired for DST, needs scoping.
  3. Wants a league scoring config (e.g. reception points) so projections
     reflect the user's actual league rules instead of a fixed default.
  4. Wants personal branding on the page: name plus LinkedIn/GitHub links.
  Logged in README §10 as "Post-deployment backlog." Decision: ship Phase 5
  as-is, proceed through the remaining roadmap phases, revisit this list
  after Phase 7 has a deployed URL to iterate against.
- Approx LLM $ spent this phase: ~$0.01-0.02 (1 real `claude-haiku-4-5` call
  during the builder's CORS/end-to-end check, plus at least 1 more from the
  user's manual browser test).

## 2026-08-12 — Phase 6: News RAG

- Shipped `scripts/refresh_news.py` (fetches ESPN/Yahoo/RotoBaller RSS, tags each
  item with known player(s) it mentions via `pipeline.entity_extraction`'s regex
  matcher, drops items with no known player or no parseable pubDate, keeps items
  from the last `--max-age-days` days (default 10, per README §4.1), joins
  `player_id` from `data/processed/player_stats.parquet` -> `data/{raw,processed}/news.parquet`;
  run as `python -m scripts.refresh_news` since it imports `pipeline`).
  `embeddings/build_embeddings.py` (embeds tagged rows with `sentence-transformers`
  `all-MiniLM-L6-v2` and upserts into a Chroma `news` collection at
  `embeddings/chroma_db/`, id'd `{player_id}:{link}` so reruns upsert rather than
  duplicate; embedding only happens here, never in the API request path, per
  README §13). `retrieval/news_retriever.py`'s `retrieve_news(player_id,
  player_name, k=3, max_age_days=7)` does a Chroma metadata-filtered
  (`player_id` + `published_ts >= cutoff`) semantic query and returns `[]`
  (never raises) if nothing's indexed yet. `pipeline/context_builder.build_context()`
  gained an optional `news_fn` param that adds the §7 "Recent news" bullet only
  when passed - existing callers with no `news_fn` see byte-identical output, so
  Phase 2/3 tests needed zero changes to their assertions; `decide()` wires in
  the real retriever by default (tests override with a stub to stay hermetic).
- Done-check: PASS - a real end-to-end run (`refresh_stats` -> `refresh_news` ->
  `build_embeddings` -> `decide()`, 2026 week 1 real data) produced a context
  with 3 "Recent news" bullets for Aaron Rodgers (real ESPN/Yahoo headlines about
  his Lambeau Field return) and 1 for Bo Nix, and the LLM's actual recommendation
  cited one in its risk factors ("Bo Nix's 2026 season described as critical -
  could indicate uncertainty about consistency"); independently re-run by a fresh
  checker subagent with its own player pair.
- Tests: `python -m pytest -q` -> 73 passed (7 new: `test_refresh_news.py`,
  `test_news_retriever.py`, `test_build_embeddings.py`, plus context-builder news
  wiring cases). `ruff check .` -> clean. All new Chroma-touching tests use
  `chromadb.EphemeralClient()` with fixture embeddings/stub encoders - no real
  model load or network call in the suite.
- Bugs found and fixed during the builder's own real-data run (not by the
  checker - caught before requesting review):
  1. `chromadb.EphemeralClient()` shares its underlying store across instances
     within one process, so `test_build_embeddings.py` and `test_news_retriever.py`
     both writing to a collection literally named "news" leaked documents
     between test files depending on run order. Fixed by giving every test a
     unique collection name (`news-test-<uuid>`), monkeypatching
     `build_embeddings`'s module-level `COLLECTION_NAME` where needed.
  2. `tag_and_filter`'s recency cutoff (`datetime.now(timezone.utc)`, tz-aware)
     compared against a `published_at` column declared as tz-naive
     `Datetime("us")` raised a polars `SchemaError` on any real (or fixture)
     tz-aware timestamp. Fixed by declaring the column `Datetime("us", "UTC")`.
  3. RSS titles came through with literal double-encoded HTML entities (e.g.
     `Aaron Rodgers &quot;bummed&quot;...`) because only `description` was
     cleaned, not `title`. Both now go through the same `_clean_text()`
     (BeautifulSoup markup strip + `html.unescape`), verified against the real
     ESPN feed.
- Checker pass: independently re-ran the done-check against real live data with
  its own player pair (Christian McCaffrey vs. Chase Brown, not the builder's
  Aaron Rodgers/Bo Nix) - PASS, confirmed reproducible rather than a one-off.
  Ran the full pytest suite 7 times (including `-p no:randomly`) to rule out
  order-dependent flakiness in the Chroma test isolation fix - none found.
  Found one real defect, fixed before marking the phase done:
  4. `tag_and_filter`'s `id_map.unique(subset=["player_name"])` arbitrarily kept
     one `player_id` when two different real players share a display name -
     verified live in the current `player_stats.parquet` ("Byron Young": a DT
     and an LB with different ids; "Jaylon Jones" likewise). Any article about
     either would have been silently misattributed to whichever id survived the
     dedup, with no error. Fixed by dropping tags for any name that maps to more
     than one `player_id` instead of guessing - consistent with this app's
     existing "clear failure over a silent guess" stance (`PlayerNotFoundError`,
     the hallucinated-recommendation check) and CLAUDE.md's "never implement
     fuzzy player-name matching." Narrow blast radius (2 of 1447 names in live
     data) but same silent-mismatched-join bug class flagged in Phases 3-4's
     checker passes, so treated as blocking rather than deferred.
- Tests after the fix: `python -m pytest -q` -> 74 passed (1 new regression
  test for the ambiguous-name-drop behavior). `ruff check .` -> clean.
- Approx LLM $ spent this phase: ~$0.02 (1 real `claude-haiku-4-5` call each
  from the builder's and the checker's end-to-end done-check runs).
