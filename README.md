# Fantasy_Football_Wizard

## Quickstart (uv)

1. Install `uv` and ensure it is on your PATH.
2. Create the environment and install dependencies:

```powershell
uv venv --python 3.11
uv sync --extra refresh --extra dev
```

3. Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```
```zsh
source .venv/bin/activate
```

Notes:
- Dependencies are defined in `pyproject.toml`; `uv sync` regenerates `uv.lock`. The base `dependencies` list (Phase 7 on) is deliberately just what the FastAPI request path needs — no `nflreadpy`/`pandas`/`sentence-transformers`/torch, per §13's "no torch in the API image." Local dev needs the `refresh` extra too (data/news/embeddings scripts + their tests) — `dev` adds `pytest`/`ruff`/notebook tooling. The API's Docker image installs the base group only.
- Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` (optionally `ANTHROPIC_MODEL`) — never commit `.env`.

## Running the app locally

Backend (from repo root, venv active):
```zsh
uvicorn api.main:app --reload
```
Serves on `http://localhost:8000` (`GET /health`, `GET /players`, `POST /recommendation`).

Frontend (separate terminal):
```zsh
cd frontend
npm run dev
```
Serves on Vite's default (`http://localhost:5173`) and calls the backend via `VITE_API_BASE_URL` (defaults to `http://localhost:8000`); CORS on the backend already allows Vite's default origin.

If `data/{raw,staged,processed}/*.parquet` or `embeddings/chroma_db/` are missing or stale, regenerate them (needs the `refresh` extra):
```zsh
python -m scripts.refresh_stats
python -m scripts.refresh_news
python -m embeddings.build_embeddings
```

---

# Development Roadmap

Ordered execution plan. Each phase references the detailed sections below; a phase is complete when its done-check passes. Work phases in order.

| Phase | Scope | Refs | Done when |
|-------|-------|------|-----------|
| **0. Environment & LLM core** *(~done)* | uv env, deps, `llm/interface.py` on the Anthropic SDK, `.env.example` | §1, §9 | Smoke test (`python llm\interface.py`) returns a validated `Recommendation` |
| **1. Structured data ingestion** *(done)* | Rebuild `scripts/refresh_stats.py` on nflreadpy (stats, snap counts, xFP, injuries, depth charts, schedules); Sleeper player-dump + projections fetchers; Parquet in `data/` via raw→staged→processed; join on `load_ff_playerids()` | §1.5, §3 | One command refreshes all Parquet artifacts for the upcoming week |
| **2. Context builder** *(done)* | `pipeline/entity_extraction.py` (regex player/week parsing); `pipeline/context_builder.py` producing LLM-ready comparison text from Parquet | §5–§7 | `build_context(["Player A", "Player B"], week)` returns the §7 format; unit-tested with fixture data |
| **3. Decision engine end-to-end** *(done)* | `pipeline/decision_engine.py`: context builder → `run_llm()` → `Recommendation` | §8–§9 | A real two-player question answers correctly from the terminal |
| **4. FastAPI backend** *(done)* | `api/main.py` with `POST /recommendation` and `GET /players`; `.env` loaded at startup; CORS for frontend origin | §14 | `curl` returns a recommendation JSON |
| **5. React frontend** *(done)* | `frontend/` Vite app: player pickers, question input, recommendation + confidence display | §10 | Full flow works locally against FastAPI |
| **6. News RAG** *(done)* | `scripts/refresh_news.py` (RSS feeds); `embeddings/build_embeddings.py` (sentence-transformers → Chroma, player-ID + date metadata); `retrieval/news_retriever.py` (top-k, ≤7-day filter); wire into context builder | §4, §6.2 | Recommendations cite recent news |
| **7. Deployment** | Dockerfile (no torch in API image); Cloud Run service; GCS-backed data/Chroma; Cloud Run Jobs + Scheduler for refresh; Cloudflare Pages frontend; **rate limiting on `POST /recommendation`** (decided Aug 2026 — see §14) | §13, §14 | Public URL serves a recommendation |
| **8. Evaluation** | Log recommendations vs FantasyPros consensus and actual weekly outcomes | §12 | Week-over-week agreement tracking exists |


# Fantasy Football Wizard - Project Checklist

> An LLM-powered fantasy football decision assistant that uses structured NFL analytics data and retrieval-augmented generation (RAG) over real-time fantasy news to provide start/sit recommendations with explanations.

### Architectural Note

This system follows a **backend-orchestrated LLM architecture**:

- The LLM is used strictly for reasoning and explanation
- Structured data is queried deterministically via Python/Pandas
- Unstructured data is retrieved via vector search (RAG)
- All orchestration happens in the backend (not inside the LLM)

This mirrors real-world production LLM systems.

---

## 0. Project Definition

### Goal
Build a **decision-support system** (not just a chatbot) for fantasy football start/sit and flex decisions.

### Non-Goals (v1)
- Draft strategy
- Trades
- Waiver wire optimization
- DFS or betting advice
- Full-season simulations

These may be considered future extensions.

**Promising future extension**: league-aware advice via the Sleeper API (free, read-only, no auth) — pull actual rosters, matchups, and waiver trends from `GET /v1/league/{league_id}/...` so recommendations account for the user's real team instead of manually entered players.

---

## 1. Tech Stack

### Core
- **Python 3.10+**
- **LLM**: Claude API (`anthropic` SDK) — `claude-haiku-4-5` default, overridable via `ANTHROPIC_MODEL` (e.g. `claude-sonnet-5` for harder reasoning)
- **Embeddings**: `sentence-transformers`
- **Vector Database**: Chroma (local)
- **UI**: React frontend + FastAPI backend (see section 14)

### Data & Processing
- **Structured Data**: `nflreadpy` (nflverse), Sleeper API, Pandas/Polars
- **Unstructured Data**: Fantasy news via free RSS/JSON feeds (ESPN, Yahoo, RotoBaller)
- **Scheduling**: Cron jobs or manual refresh scripts

---

## 1.5 Data Sources (verified free, 2026)

All primary sources below are free and require no paid subscription. Sources marked "no key" need no signup at all.

| Data | Primary Source | Access | Notes |
|------|---------------|--------|-------|
| Weekly/seasonal player stats | `nflreadpy.load_player_stats()` | Python pkg, no key | Already in use |
| Snap counts | `nflreadpy.load_snap_counts()` | no key | PFR-sourced, 2012+ |
| Next Gen Stats (adv. passing/rushing/receiving) | `nflreadpy.load_nextgen_stats()` | no key | 2016+ |
| Opportunity / expected fantasy points | `nflreadpy.load_ff_opportunity()` | no key | Target share, air yards, xFP — 2006+ |
| Official injury reports (practice participation) | `nflreadpy.load_injuries()` | no key | 2009+, weekly cadence |
| Real-time injury status | Sleeper `GET /v1/players/nfl` | REST, no key | `injury_status` per player; ~5MB, fetch 1x/day |
| Depth charts | `nflreadpy.load_depth_charts()` | no key | 2001+ |
| Projections (weekly) | Sleeper `api.sleeper.com/projections/nfl/{season}/{week}` | REST, no key | Undocumented but stable; PPR/half/standard |
| Consensus rankings (ECR) | `nflreadpy.load_ff_rankings()` | no key | FantasyPros ECR via ffverse |
| Player ID / name normalization | `nflreadpy.load_ff_playerids()` | no key | Maps Sleeper/ESPN/Yahoo/PFR/GSIS IDs — use this instead of fuzzy name matching |
| Schedules + Vegas lines | `nflreadpy.load_schedules()` | no key | Includes spread/total — useful game-script signal |
| Trending adds/drops | Sleeper `/v1/players/nfl/trending/{add\|drop}` | REST, no key | Community waiver signal |
| News (RAG corpus) | ESPN RSS (`espn.com/espn/rss/nfl/news`), Yahoo Sports NFL RSS, RotoBaller free feeds (XML/JSON/RSS) | RSS/JSON, no key | Prefer feeds over scraping |
| My league rosters/matchups | Sleeper `/v1/league/{id}/...` | REST, no key | Read-only, no auth; rate limit <1000 calls/min |

**Secondary / fallback sources**
- FantasyPros CSV export (manual download) — projections backup
- ESPN hidden fantasy API (`fantasy.espn.com/apis/v3/games/ffl/...`) + `espn-api` Python lib — ownership %, ESPN league integration (needs cookies for private leagues); unofficial, may break without notice
- Yahoo Fantasy Sports API — official but OAuth-heavy; only if a Yahoo league must be supported

**Deliberately excluded**: Fantasy Nerds ($74.95/yr), SportsDataIO, MySportsFeeds (trial-gated) — paid; Rotoworld scraping — fragile, feeds above cover it.

---

## 2. Repository Structure

```
fantasy-football-wizard/
|-- data/                  # local artifacts (gitignored; GCS in prod)
|   |-- raw/
|   |-- staged/
|   `-- processed/
|       |-- player_stats.parquet
|       |-- projections.parquet
|       `-- injuries.parquet
|-- embeddings/
|   |-- build_embeddings.py
|   `-- chroma_db/
|-- retrieval/
|   |-- news_retriever.py
|   `-- filters.py
|-- llm/
|   |-- interface.py       # exists — Anthropic SDK + structured outputs
|   `-- prompt_templates.py
|-- pipeline/
|   |-- entity_extraction.py
|   |-- context_builder.py
|   `-- decision_engine.py
|-- api/
|   `-- main.py            # FastAPI app
|-- frontend/              # React (Vite) app
|-- scripts/
|   |-- refresh_stats.py
|   |-- refresh_news.py
|   `-- refresh_embeddings.py
|-- tests/                 # pytest, test_*.py
|-- Dockerfile             # API image (Phase 7; no torch)
|-- .env.example
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

Only `llm/interface.py` exists today — everything else is created phase-by-phase per the roadmap.


---

## 3. Structured Data Ingestion (Stats & Projections)

**Recommended Tools**
- `nflreadpy` for raw data ingestion
- `pandas` for aggregation and feature engineering
- `pyarrow` / Parquet for efficient local storage

**Why**
- Structured data requires exact filtering (week, opponent, player)
- Vector databases are intentionally NOT used for tabular data
- Pandas allows deterministic, debuggable data access

### 3.1 Player Stats

**Tool**
- `nflreadpy`

**Steps**
- Pull weekly player statistics
- Enrich with snap counts (`load_snap_counts`), opportunity/xFP (`load_ff_opportunity`), and Next Gen Stats (`load_nextgen_stats`)
- Aggregate:
  - Last 3 weeks
  - Season averages
- Store processed data as Parquet or CSV

**Key Fields**
```
player_name
position
team
week
fantasy_points
snap_percentage
targets
rush_attempts
epa
```


---

### 3.2 Fantasy Projections

**Sources**
- Sleeper projections endpoint (primary): `https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular` — free, no key, returns per-player projections for PPR/half/standard
- `nflreadpy.load_ff_rankings()` — FantasyPros expert consensus rankings (ECR)
- FantasyPros CSV export (manual fallback)

**Steps**
- Join sources on canonical player IDs via `nflreadpy.load_ff_playerids()` (maps Sleeper/ESPN/Yahoo/GSIS/PFR IDs) — avoid fuzzy name matching
- Store projections by week and source

**Key Fields**
```
player_name
week
projected_points
source
```

---

### 3.3 Injury Reports

**Sources**
- `nflreadpy.load_injuries()` — official NFL injury reports with practice participation (weekly cadence)
- Sleeper `GET /v1/players/nfl` — `injury_status` / `injury_note` per player for intra-week updates (daily cadence)

**Steps**
- Parse practice participation reports
- Overlay Sleeper's real-time status on top of the official weekly report
- Store both structured fields and raw text notes

**Key Fields**
```
player_name
status
practice_level
report_date
notes
```

---

## 4. Unstructured Data Ingestion (Fantasy News)

**Recommended Tools**
- `requests` / `httpx` for fetching news
- `BeautifulSoup` (if scraping is required)
- `sentence-transformers` for embeddings
- Chroma for vector search

**Why**
- News is unstructured, subjective, and time-sensitive
- Semantic search is required to retrieve relevant context
- Embeddings are rebuilt frequently to ensure freshness


### 4.1 News Collection

**Sources**
- ESPN NFL news RSS: `https://www.espn.com/espn/rss/nfl/news` — no key
- Yahoo Sports NFL RSS: `https://sports.yahoo.com/nfl/rss` — no key
- RotoBaller free player news feeds (XML/JSON/RSS) — player-tagged blurbs, ideal RAG input
- Sleeper trending adds/drops (`/v1/players/nfl/trending/add`) — community signal to prioritize which players' news matters this week

**Steps**
- Prefer structured feeds over HTML scraping (BeautifulSoup only as last resort)
- Fetch news from the last 7-10 days only
- Tag each item with player ID (via `load_ff_playerids`) and date
- Store raw text data

---

### 4.2 Embeddings Pipeline (RAG)

**Tools**
- `sentence-transformers`
- Chroma

**Steps**
- Chunk news articles by player and article
- Generate embeddings
- Store metadata alongside embeddings

**Metadata Example**
```
{
  "player_id": "4046",          // canonical ID via load_ff_playerids()
  "player_name": "Jordan Love",
  "date": "2026-09-01",
  "source": "ESPN RSS"
}
```

---

## 5. Entity Extraction & Query Parsing

**Recommended Tools**
- Regex + string matching for MVP
- Optional lightweight LLM call for edge cases

**Why**
- Player names and weeks are well-defined entities
- Avoids unnecessary LLM calls for simple parsing
- Improves latency and reliability


**Goal**
Extract player names, positions, and week from user input.

**Tools**
- Regex-based parsing (MVP)
- Optional: small LLM call for robustness

---

## 6. Retrieval Strategy

**Recommended Tools**
- Pandas DataFrames (initial version)
- Optional: SQLite / DuckDB for scaling

**Why**
- Structured data benefits from exact queries
- Easier to debug and validate than semantic retrieval
- Matches how analytics systems work in production


### 6.1 Structured Retrieval (No RAG)

**Tools**
- Pandas / SQL

**Steps**
- Fetch recent stats for each player
- Fetch fantasy projections
- Fetch injury status
Structured data is queried deterministically and injected into the LLM context.
---

### 6.2 Unstructured Retrieval (RAG)

**Recommended Tools**
- Chroma for vector storage
- Metadata filtering for player name + recency

**Why**
- News relevance is semantic, not keyword-based
- Metadata filtering prevents stale or unrelated context


**Tools**
- Chroma

**Steps**
- Query embeddings by player name
- Filter results by recency (<= 7 days)
- Retrieve top-k relevant news chunks

---

## 7. Context Assembly
**Recommended Tools**
- Plain Python functions
- Pydantic models (optional) for structured context objects

**Why**
- Context assembly is deterministic business logic
- Keeping it outside the LLM improves explainability
- Enables easier testing and debugging


Convert raw data into a clean, LLM-friendly comparison format.

Example Context

Recent-form lines never blend across a season boundary (fixed 2026-09-19 — see
CLAUDE.md Phase 1/2): once a player has 3+ games played this season, only
this-season numbers appear; earlier in a season, this-season and last-season
figures are shown as separate, clearly-labeled lines instead of one averaged
number. All of these figures are PPR (matching `projected_points`, §9) - never
the non-PPR fields the processed table also carries.

```
PLAYER COMPARISON

Jordan Love:
- This season (4 games): 17.9 season avg, 18.4 avg over last 3 games
- Projected points: 17.1
- Injury: Questionable -> Full practice Friday
- Recent news:
  - "Packers plan to stay aggressive..."

Malik Nabers:
- This season (1 game): 6.9 avg fantasy points
- Last season (4 games): 9.8 season avg, 10.7 avg over final 3 games
- Most recent game played (Week 1, 2026): 6.9 pts
- Projected points: 13.8
- Injury: Healthy


```


---

## 8. Prompt Engineering

**Recommended Tools**
- Prompt templates stored as versioned files
- Optional: Jinja2 for templating

**Why**
- Prompts are part of system logic
- Versioning prompts allows experimentation and rollback


Define system instructions and a structured response schema.

**Tool**
- Custom prompt templates

**System Instructions**
- Act as a fantasy football analyst
- Use only the provided data
- Explain reasoning clearly
- Explicitly assess risk

**Response Schema**

```
{
  "start": "Player A",
  "bench": "Player B",
  "confidence": 0.0,
  "key_factors": [],
  "risk_factors": []
}

```


---

## 9. LLM Inference

**Recommended Tools**
- Claude API via the `anthropic` SDK (see `llm/interface.py`)
- Structured outputs (`client.messages.parse` + Pydantic) — the API guarantees the response matches the schema, no JSON-repair fallback needed

**Why**
- `claude-haiku-4-5` ($1/$5 per MTok) is the best performance-per-dollar for this workload: short assembled context in, small structured recommendation out (~half a cent per query)
- `ANTHROPIC_MODEL=claude-sonnet-5` is a drop-in upgrade for harder reasoning
- Structured outputs eliminate malformed-response handling entirely

**Steps**
- Assemble context (section 7), call `run_llm(context, question)`
- Receive a validated `Recommendation` (start, bench, confidence, key_factors, risk_factors)
- Return it from the API layer for display

---

## 10. React Frontend

**Recommended Tools**
- React (Vite) frontend in `frontend/`
- FastAPI backend (section 14) as the only thing the frontend talks to

**Why**
- Clean separation: React handles UI only; all orchestration, data access, and LLM calls live in the Python backend
- The frontend never sees the Anthropic API key — it only calls FastAPI endpoints

UI for interacting with the fantasy assistant.
**Features**
- Player selection UI
- Natural language question input
- Recommendation display
- Confidence visualization

**Optional Enhancements**
- Source citations
- Debug view showing injected context

**Post-deployment backlog** (raised by the user after the Phase 5 done-check passed, 2026-08-06 — deliberately deferred: ship Phase 5 as-is, deploy per Phase 7, then iterate):
- Replace the `<select>` player pickers with a searchable/typeahead input — the full player list is too long to scan
- Add defense/DST options so team defenses can be compared, not just offensive skill players (needs a defense data source — none of the current nflreadpy/Sleeper tables used are wired for DST; scope during implementation)
- League scoring config (e.g. reception points, custom scoring rules) so projections/recommendations reflect the user's actual league instead of a fixed default
- Personal branding on the page: the user's name plus links to their LinkedIn and GitHub

---

## 11. Data Refresh Strategy

**Which week gets refreshed**: `refresh_stats.py` targets the **upcoming** week — the earliest regular-season week that still has an unplayed game (mid-week that's the week in progress; in the off-season it's week 1 of the next season). Stats aggregate over the weeks *before* the target, while projections and injury reports are fetched *for* it, so a recommendation is never built from the box score of the game it's projecting. Early in a season the recent-form window reaches back into the prior season, and nflverse tables that a new season hasn't published yet are skipped rather than failing the refresh.

| Data Type | Source | Refresh Frequency |
|---------|--------|------------------|
| Player Stats (+ snaps, xFP, NGS) | nflreadpy | Weekly |
| Projections | Sleeper endpoint / `load_ff_rankings` | Weekly |
| Injury Reports (official) | `load_injuries` | Daily during season |
| Injury Status (real-time) + player dump | Sleeper `/v1/players/nfl` | Daily (~5MB, don't over-fetch) |
| News | RSS/JSON feeds | Daily |
| Trending adds/drops | Sleeper | Daily |
| Embeddings | — | On news refresh |

Scheduling:
- Local dev: run refresh scripts manually or via OS scheduler
- Production: Cloud Run Jobs triggered by Cloud Scheduler (see §13) — stats weekly, news daily, embeddings rebuilt after each news refresh

---
## 12. Evaluation (Lightweight)

**Approaches**

- Compare recommendations against FantasyPros consensus
- Track agreement rates week-over-week
- Log confidence vs actual fantasy outcomes
---
## 13. Deployment

- Local dev: FastAPI (`uvicorn`) + React dev server (Vite)
- **Frontend**: Cloudflare Pages — free tier, unlimited static bandwidth, deploys the Vite build on git push
- **Backend**: Dockerized FastAPI on GCP Cloud Run — scale-to-zero, free tier (~2M requests/month) covers personal use
- **Data refresh**: Cloud Run Jobs + Cloud Scheduler (replaces local cron); artifacts written to a GCS bucket
- **State caveat**: Cloud Run is stateless/ephemeral — Parquet files and `chroma_db/` must live in GCS, not on local disk. **Implemented**: `pipeline/gcs_sync.py`'s `sync_from_gcs()` downloads them into the same local paths at FastAPI startup (a lifespan hook, gated on the `GCS_BUCKET` env var — unset in local dev, so nothing changes there). Download-on-start was chosen over a GCS FUSE volume mount because Chroma's SQLite backend needs file-locking semantics FUSE doesn't reliably support; still needs a real bucket populated by the refresh job before this can be exercised end-to-end.
- **Staleness caveat (open, not yet resolved)**: the sync only runs once, at startup. A Cloud Run instance that stays warm across requests will keep serving the data it downloaded at its last cold start, even after the scheduled refresh job (not yet built) writes newer data to the bucket — there's no in-process re-sync. This needs a decision once the refresh job + Cloud Scheduler pieces below actually exist: either the API polls the bucket periodically, or the refresh job forces a new Cloud Run revision/restart after publishing a complete snapshot. Deferred rather than guessed at now, since there's no scheduler to hook into yet.
- **Cold-start caveat**: keep `sentence-transformers` (torch) out of the request path — embed at refresh time in the Cloud Run Job; the API container should only *query* Chroma. Otherwise scale-to-zero cold starts take tens of seconds and the image balloons past 2GB.
- Fallback if statefulness gets annoying: a small always-on VM (Fly.io / Lightsail, ~$5/mo) with a persistent volume

---

## 14. FastAPI Backend (required)

With a React frontend, **FastAPI is the required backend layer** — it exposes the pipeline as HTTP endpoints the frontend consumes.

### Why FastAPI?

FastAPI is **not a replacement** for your pipeline or LLM. It exposes your backend logic as **HTTP endpoints**, allowing the React app (and later other clients — mobile, Slack bots) to interact with the Fantasy Football Wizard.

Benefits include:

- **Separation of Concerns:** React only handles UI; FastAPI handles pipeline orchestration.
- **Security:** the Anthropic API key stays server-side.
- **Reusability:** Multiple frontends can call the same backend endpoints.
- **Scalability:** Easier to deploy and scale in the cloud.
- **Debugging:** Endpoints can be tested independently with tools like Postman or `curl`.
- **Resume Value:** Demonstrates experience building a production-ready ML API.

### Rate limiting (done, code-level; Cloud Armor still pending actual deploy)

`POST /chat` spends the owner's Anthropic credits on every call, and CORS is not a
defense — it only constrains browsers, and `curl` ignores it entirely. The input caps
already in `api/main.py` (`MAX_QUESTION_LENGTH`, `MAX_PLAYER_NAME_LENGTH`) bound the
cost of a single request but not the number of requests.

**Implemented**: `slowapi` (in-process Starlette middleware) limits `POST /chat` to
`CHAT_RATE_LIMIT` (30 requests/hour per caller IP, keyed by `api/main.py`'s `_client_ip()`
— the rightmost `X-Forwarded-For` entry, since that's the one Cloud Run's GFE proxy
itself appends and a caller can't forge, not the leftmost caller-supplied one);
`/players` and `/health` are unlimited since they cost no LLM call.

**Deploy-time requirement, not yet applied (no Cloud Run service exists yet)**: the
limiter's counters are in-memory per instance, not shared across replicas. Under
Cloud Run's default autoscaling, one caller triggering the limit is itself the kind of
burst that can spin up a second instance — which starts with an empty counter and
grants that same caller another 30/hour. **Before this ships publicly, the Cloud Run
service must be deployed with `--max-instances=1`** (fine at personal-use traffic
levels) so the in-process limiter stays meaningful; if scale-out is ever needed, a
**Cloud Armor rate-limit policy** in front of Cloud Run replaces this in-process limit
entirely — see Phase 7 below.

`/docs`, `/redoc`, and `/openapi.json` are **disabled in prod**: `api/main.py`'s
`_docs_config()` passes `docs_url=None, redoc_url=None, openapi_url=None` to `FastAPI()`
whenever `GCS_BUCKET` is set (the same signal Cloud Run vs. local dev already uses
elsewhere in this file); local dev keeps them on.

### Example FastAPI Endpoint

```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.context_builder import build_context
from llm.interface import run_llm

app = FastAPI()

class RecommendationRequest(BaseModel):
    players: list[str]
    week: int

@app.post("/recommendation")
def get_recommendation(request: RecommendationRequest):
    context = build_context(request.players, request.week)
    response = run_llm(context)
    return response
```
