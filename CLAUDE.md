# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

A conversational decision-support tool for fantasy football. It combines structured NFL stats/projections with RAG over fantasy news to answer start/sit, flex, and trade questions in a chat interface, with cited sources. (Revised 2026-09-09 — originally scoped as "not a chatbot"; the user decided a chat interface covering more question types is the direction going forward. Design: `docs/superpowers/specs/2026-09-09-chat-assistant-design.md`.)

**Non-goals (v1)**: draft strategy (deferred — needs new ADP/rankings data, explicitly saved for a later task, not dropped), waiver wire optimization, DFS/betting advice, full-season simulations. Trade evaluation is now in scope (removed from non-goals 2026-09-09) — it reuses the existing stats/news context with no dedicated value model.

## Source of Truth

`README.md` is the canonical build plan: its **Development Roadmap** table defines the execution order (phases 0-8 with done-checks), and sections 0-14 hold the per-topic detail (data ingestion, embeddings, retrieval, prompt engineering, LLM inference, React UI, FastAPI backend, refresh strategy). When planning or implementing features, follow the roadmap order unless the user redirects. If other docs conflict, `README.md` wins unless the user explicitly overrides it.

**The repo was reset to a clean slate in July 2026** (prior skeleton code lives only in git history). Only `llm/interface.py` exists as code; `data/`, `embeddings/`, `retrieval/`, `pipeline/`, `api/`, `frontend/`, `scripts/` described in README section 2 do not exist yet — check before assuming a file exists.

## Development Roadmap (status)

Authoritative sequence and done-checks live in README → Development Roadmap. Current status:

- **Phase 0 — Environment & LLM core: ~done** (`llm/interface.py`, `.env.example`; done-check needs a real API key)
- **Phase 1 — Structured data ingestion: done** (`scripts/refresh_stats.py`; `python -m scripts.refresh_stats` refreshes `data/{raw,staged,processed}/*.parquet` for the **upcoming** week — `resolve_target_week()` picks the earliest week with an unplayed game, stats aggregate over weeks before it, projections/injuries are fetched for it; `tests/test_refresh_stats.py` covers the aggregation/overlay logic. **Reworked 2026-09-20 (custom league scoring)**: `data/processed/player_stats.parquet` is no longer pre-aggregated into one row per player under a single fixed formula — it now holds many historical **per-game rows** (raw stat counts: `pass_yards`, `receptions`, `rec_tds`, etc., under canonical names shared across nflverse and Sleeper), because fantasy points depend on which scoring tier the caller picks and can no longer be baked in at refresh time. A new `data/processed/meta.parquet` (one row: `season`, `week`) is the only place "which week is the upcoming decision for" survives now that `player_stats.parquet` spans many weeks — read by `pipeline.decision_engine.target_season_week()`. Recent-form aggregation (this-season/last-3/prior-season averages, never blending across the season boundary — the season-boundary fix from the 2026-09-19 bug report still holds) moved out of this script and into `pipeline/player_form.py`'s `compute_recent_form()`, which runs at request time against whichever `ScoringRules` the caller selected, via `pipeline/scoring.py`'s `compute_fantasy_points()`. Four scoring tiers: `PRESET_PPR`, `PRESET_HALF_PPR`, `PRESET_STANDARD` (all in `pipeline/scoring.py`), plus arbitrary **Custom** rules (LLM-parsed from a free-text description — see the chat-assistant/API line below). This supersedes the old `_ppr`-suffixed-field approach from PR #11, which baked one fixed (PPR) formula into the processed data at refresh time)
- **Phase 2 — Context builder: done** (`pipeline/entity_extraction.py`, `pipeline/context_builder.py`; `build_context(players, season, week, scoring_rules)` returns the §7 format minus the "Recent news" bullet, deferred to Phase 6; `tests/test_context_builder.py` covers extraction + assembly. **Updated 2026-09-19**: `_format_recent_form()` never blends seasons — 3+ this-season games shows only this-season lines; fewer shows this-season (or "no games played yet") plus a separate "Last season" line and the single most recent game played, each clearly labeled with its own game count; last season drops out entirely once there are 3+ this-season games. **Reworked 2026-09-20**: `build_context` takes explicit `season` (resolved from `data/processed/meta.parquet`, since `player_stats.parquet` is no longer single-week) and `scoring_rules` params; all per-player averages/projections it reports are computed on demand from raw per-game rows via `pipeline/player_form.compute_recent_form()` under the caller's chosen tier, so the same players/week can produce three different numeric context blocks depending on which `ScoringRules` was passed in. **Fixed 2026-09-20 (past-week questions)**: `compute_recent_form()` now takes `week` and bounds "this season" to games at or before it, so a question about a played-but-not-most-recent week (e.g. "why did he score well in week 3" asked in week 6) no longer silently pulls in later games that hadn't happened yet from that week's perspective — a no-op for the normal "ask about the upcoming week" case. When the asked-about week was actually played, the context adds that week's own real stat line plus an explicit note that historical projections/news aren't retained for past weeks (`llm/interface.py`'s `RECENCY_WEIGHTING_GUIDANCE` instructs the model to state that limitation rather than answer as if it had full period-accurate context). Historical projections and wider news retention remain an explicit non-goal, not silently missing. **Fixed 2026-09-21 (future-week questions)**: the symmetric gap - asking about a week beyond the target (further out than the one week Sleeper projections are ever fetched for, e.g. "week 10" when the target is week 5) previously just silently omitted the projection, reading like an ordinary unprojected player rather than "this week isn't covered yet". `pipeline.decision_engine.is_future_query()` compares the resolved season/week against `target_season_week()` and, when true, adds an explicit note (`context_builder._format_future_note`) that no projection exists for that week and the numbers shown are current-form trends only; `RECENCY_WEIGHTING_GUIDANCE` instructs the model to state that limitation rather than presenting the trend as a week-specific prediction)
- **Phase 3 — Decision engine end-to-end: done** (`pipeline/decision_engine.py`; `decide(question, players=None, season=None, week=None, scoring_rules=None)` returns a `Decision` — resolved players/week, the context sent to the LLM, and the validated `Recommendation`; CLI: `python -m pipeline.decision_engine "Should I start X or Y in week 18?"`; `tests/test_decision_engine.py` covers orchestration with a stubbed LLM; `target_week()` defaults to the week the processed data describes, i.e. the upcoming one)
- **Phase 4 — FastAPI backend: done, superseded by the chat assistant below** (`api/main.py`; `POST /recommendation` originally, replaced 2026-09 by `POST /chat` — see the chat assistant line; `GET /players`, `GET /health` unchanged; CORS origins via `FRONTEND_ORIGINS`. **No auth or rate limiting yet** — the user decided Aug 2026 that rate limiting is required before the service gets a public URL; options are tabled in README §14, now scoped to `/chat`)
- **Phase 5 — React frontend: done, superseded by the chat assistant below** (`frontend/` Vite + React app; the original player-picker form was replaced 2026-09 by a chat UI — see the chat assistant line. **Remaining post-deployment backlog** (user request, 2026-08-06): defense/DST comparison support and personal branding/links — see README §10; typeahead search was superseded by `@`-mention search)
- **Phase 6 — News RAG: done** (`scripts/refresh_news.py` fetches ESPN/Yahoo/RotoBaller RSS, tags each item with known player(s) via `pipeline.entity_extraction`, keeps items from the last `--max-age-days` (default 10) days, writes `data/{raw,processed}/news.parquet`; run as `python -m scripts.refresh_news` — it imports `pipeline`, so a bare script path won't resolve. When a display name maps to more than one `player_id` in `player_stats.parquet` (e.g. two real "Byron Young"s), that name's tags are dropped rather than guessed — no fuzzy/context-based disambiguation from RSS text. `embeddings/build_embeddings.py` embeds tagged rows with `sentence-transformers` (`all-MiniLM-L6-v2`) and upserts into a Chroma `news` collection at `embeddings/chroma_db/` (gitignored), one entry per article-player pair, keyed `{player_id}:{link}` so reruns upsert instead of duplicating; run as `python -m embeddings.build_embeddings`. `retrieval/news_retriever.py`'s `retrieve_news(player_id, player_name, k=3, max_age_days=7)` does a metadata-filtered (`player_id` + `published_ts >= cutoff`) semantic query, returning `[]` (never raising) if nothing's been embedded yet. `pipeline/context_builder.build_context()` gained an optional `news_fn` param that adds the §7 "Recent news" bullet only when passed — pre-Phase-6 callers with no `news_fn` are byte-identical to before; `pipeline/decision_engine.decide()` wires in the real `retrieve_news` by default)
- **Chat assistant (inserted before Phase 7, decided 2026-09-09): done** (merged 2026-09-16, PR #8) — `POST /chat` (`pipeline/chat_engine.py`) answers start/sit, trade, and single-player questions for any number of `@`-mentioned or extracted players, returning prose + cited `sources` (`retrieval.news_retriever.NewsItem`) + an optional `recommendation` only when the question is start/sit-shaped; `frontend/` is now a chat UI (`ChatInput` with `@`-mention autocomplete, `MessageBubble`, `SourceList`, `DecisionCard`). Full design in `docs/superpowers/specs/2026-09-09-chat-assistant-design.md`, plan in `docs/superpowers/plans/2026-09-09-chat-assistant.md`. Draft strategy/ADP/rankings still explicitly excluded — later task. **Custom league scoring (2026-09-20, done)**: `POST /scoring-rules` (`api/main.py`) takes a base tier (`ppr`/`half_ppr`/`standard`/`custom`) plus, for `custom`, a free-text `custom_description` (e.g. "catches are worth 0.5 points instead of 1") and a `base_hint` tier to start from; for `custom` it calls `llm/interface.py`'s `parse_custom_scoring_rules()` (LLM-based, starts from the hinted preset and only overrides the fields the description actually changes) and returns a full `ScoringRules` JSON object. `POST /chat` accepts an optional `scoring_rules` body field threaded through `decide()`/`build_context()`/`compute_recent_form()` so recommendations reflect the caller's league scoring instead of always assuming PPR; the frontend has a scoring-tier toggle that calls `/scoring-rules` and attaches the result to subsequent `/chat` requests.
- **Phase 7 — Deployment (Cloud Run + Cloudflare Pages): in progress**
  - Done: `Dockerfile`/`.dockerignore` (lean API image); dependency split (base vs. `refresh` extra); GCS-backed data loading (`pipeline/gcs_sync.py`'s `sync_from_gcs()` downloads `data/processed/*.parquet` + `embeddings/chroma_db/` from a `GCS_BUCKET` env var at FastAPI startup via a lifespan hook — no-op when unset, so local dev is unaffected; missing/undownloaded files now surface as a clean `FileNotFoundError`→503 instead of a raw 500, on both `/chat` and `/players`)
  - Done: rate limiting + docs lockdown (`api/main.py`; `slowapi` limits `POST /chat` to `CHAT_RATE_LIMIT` = 30/hour per caller IP via `get_remote_address`, exceeding it returns 429; `/docs`, `/redoc`, `/openapi.json` disabled via `_docs_config()` whenever `GCS_BUCKET` is set. Cloud Armor as an additional edge-level layer is still a Cloud Run-deploy-time follow-up, not required to go live — README §14)
  - Not started: actual Cloud Run deploy; Cloud Run Jobs + Cloud Scheduler for refresh; Cloudflare Pages frontend deploy. None of these have touched real GCP/Cloudflare resources yet.
- **Phase 8 — Evaluation: not started**

When a phase's done-check passes, update its status line here and in the README roadmap table.

## Architecture

This is a **backend-orchestrated LLM pipeline**, not a prompt-only app:
- Structured data (stats, projections, injuries) is queried deterministically via Pandas/Polars — never put tabular lookups through RAG or the LLM.
- Unstructured data (news) is retrieved via embeddings + vector search (Chroma), filtered by player name and recency.
- The LLM only reasons over already-assembled context and returns explanations — it does not orchestrate retrieval or fetch data itself.
- Context assembly (structured + retrieved news → LLM-ready comparison text) lives in plain Python/Pydantic, kept out of the LLM for debuggability.
- **UI is React** (`frontend/`, Vite); it talks only to the **FastAPI backend** (`api/`). All orchestration, data access, and LLM calls stay in Python — the frontend never sees the Anthropic API key.

## LLM (decided July 2026: Anthropic API)

- **Provider**: Claude API via the `anthropic` SDK. Chosen over OpenAI for this workload: best structured-reasoning quality per dollar at the cheap tier, guaranteed-schema structured outputs, and one ecosystem with Claude Code. Do not reintroduce `openai` or `llama-cpp-python`.
- **Model**: `claude-haiku-4-5` default ($1/$5 per MTok — ~half a cent per recommendation), overridable via `ANTHROPIC_MODEL` env var; `claude-sonnet-5` is the drop-in upgrade for harder reasoning. Use exact model IDs, no date suffixes.
- **Structured outputs**: `llm/interface.py` uses `client.messages.parse(..., output_format=Recommendation)` — the API guarantees the response validates against the Pydantic `Recommendation` schema (start, bench, confidence, key_factors, risk_factors). No JSON-repair fallback is needed; don't add one.
- **Auth**: `ANTHROPIC_API_KEY` from `.env` (gitignored, loaded via `python-dotenv`); `.env.example` is the committed placeholder template — copy it to `.env`. `run_llm()` reads the key explicitly via `os.getenv` and fails fast with a clear `RuntimeError` if unset — keep that check. Note: the user's Claude Pro / ChatGPT Plus subscriptions do NOT cover API usage — API is pay-per-token separately.

## Data Sources

Canonical free sources were vetted July 2026 and are tabled in README section 1.5 — consult it before adding any data dependency. The short version:

- **nflreadpy (nflverse) is the backbone** for all structured data: `load_player_stats()`, `load_injuries()` (official reports), `load_snap_counts()`, `load_nextgen_stats()`, `load_ff_opportunity()` (target share / expected fantasy points), `load_depth_charts()`, `load_ff_rankings()` (FantasyPros consensus), `load_schedules()` (includes Vegas spread/total). Prefer adding one of these over introducing a new dependency or scraper.
- **Sleeper API** (`api.sleeper.app` / `api.sleeper.com`) is the second pillar: free, read-only, **no API key**. Player dump with real-time `injury_status` (`GET /v1/players/nfl`, ~5MB — fetch at most daily and cache), weekly projections (undocumented but stable: `projections/nfl/{season}/{week}`), trending adds/drops, and full league/roster/matchup data. Stay under 1000 calls/min.
- **Player identity**: join across sources with `nflreadpy.load_ff_playerids()` (maps Sleeper/ESPN/Yahoo/GSIS/PFR IDs). Never implement fuzzy player-name matching.
- **News for RAG**: use free RSS/JSON feeds (ESPN `espn.com/espn/rss/nfl/news`, Yahoo Sports NFL RSS, RotoBaller feeds). Scraping HTML is a last resort.
- **Avoid**: paid APIs (Fantasy Nerds, SportsDataIO, MySportsFeeds). The ESPN hidden fantasy API is unofficial/fragile — fallback only.

## Commands

Environment (uv-managed, `pyproject.toml` is the dependency source of truth):
```powershell
uv venv --python 3.11
uv sync --extra refresh --extra dev   # local dev: everything, including refresh scripts + tests
.\.venv\Scripts\Activate.ps1
```
Bare `uv sync` (no extras) installs only what the FastAPI request path needs - no
`nflreadpy`/`pandas`/`sentence-transformers`/torch. That's what `Dockerfile` uses
for the API image (README §13's "no torch in the API image"); it's too lean to run
`scripts/refresh_*.py`, `embeddings/build_embeddings.py`, or the full test suite.

Run:
```powershell
python llm\interface.py            # smoke-tests the LLM interface (needs ANTHROPIC_API_KEY in .env)
python -m scripts.refresh_news     # RSS -> data/processed/news.parquet (tagged, recency-filtered)
python -m embeddings.build_embeddings  # embeds tagged news into Chroma (embeddings/chroma_db/)
uvicorn api.main:app --reload      # FastAPI backend (once api/ exists)
```
Frontend (once `frontend/` exists): `npm run dev` from `frontend/` (Vite).

Lint/test:
```powershell
ruff check .   # dev dependency
pytest         # place tests under tests/test_*.py; prefer small deterministic tests for data transforms
```

## Deployment Target

Frontend on **Cloudflare Pages** (free static hosting of the Vite build); backend as **Dockerized FastAPI on GCP Cloud Run** (scale-to-zero); data refresh via **Cloud Run Jobs + Cloud Scheduler** writing to GCS. Two constraints shape backend code: Cloud Run is stateless (Parquet + Chroma data must load from GCS, never assume local disk persists), and the request path must stay light (embedding via `sentence-transformers`/torch happens only in refresh jobs, never in the API container — keeps images small and cold starts fast). See README section 13.

## Conventions

- Python `>=3.10`, 4-space indentation, standard PEP 8 (`snake_case` functions/vars, `PascalCase` classes).
- Data pipelines write Parquet through named stages `raw` → `staged` → `processed` under `data/` (see README section 3); keep pipeline classes chainable.
- Commit messages: short, descriptive summaries (e.g., "updated packages, minor readme update").
- Local secrets/config go in `.env` (gitignored) — never commit them, and never expose them to the React frontend.
