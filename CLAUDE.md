# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

A **decision-support system** (not a chatbot) for fantasy football start/sit and flex decisions. It combines structured NFL stats/projections with RAG over fantasy news to produce explained recommendations.

**Non-goals (v1)**: draft strategy, trades, waiver wire optimization, DFS/betting advice, full-season simulations.

## Source of Truth

`README.md` is the canonical build plan: its **Development Roadmap** table defines the execution order (phases 0-8 with done-checks), and sections 0-14 hold the per-topic detail (data ingestion, embeddings, retrieval, prompt engineering, LLM inference, React UI, FastAPI backend, refresh strategy). When planning or implementing features, follow the roadmap order unless the user redirects. If other docs conflict, `README.md` wins unless the user explicitly overrides it.

**The repo was reset to a clean slate in July 2026** (prior skeleton code lives only in git history). Only `llm/interface.py` exists as code; `data/`, `embeddings/`, `retrieval/`, `pipeline/`, `api/`, `frontend/`, `scripts/` described in README section 2 do not exist yet — check before assuming a file exists.

## Development Roadmap (status)

Authoritative sequence and done-checks live in README → Development Roadmap. Current status:

- **Phase 0 — Environment & LLM core: ~done** (`llm/interface.py`, `.env.example`; done-check needs a real API key)
- **Phase 1 — Structured data ingestion: not started**
- **Phase 2 — Context builder: not started**
- **Phase 3 — Decision engine end-to-end: not started**
- **Phase 4 — FastAPI backend: not started**
- **Phase 5 — React frontend: not started**
- **Phase 6 — News RAG: not started**
- **Phase 7 — Deployment (Cloud Run + Cloudflare Pages): not started**
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
uv sync
.\.venv\Scripts\Activate.ps1
```

Run:
```powershell
python llm\interface.py            # smoke-tests the LLM interface (needs ANTHROPIC_API_KEY in .env)
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
