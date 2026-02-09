# Fantasy_Football_Wizard

## Quickstart (uv)

1. Install `uv` and ensure it is on your PATH.
2. Run the setup script:

```powershell
.\scripts\setup_uv.ps1
```

3. Activate the environment in a new shell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Notes:
- Dependencies are defined in `pyproject.toml`.
- `scripts/refresh_stats.py` currently imports `nflreadpy`, so `nflreadpy` is included in the default deps.

## Workflow (uv)

1. Create or update the environment (run when deps change):

```powershell
.\scripts\setup_uv.ps1
```

2. Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Run scripts or the app:

```powershell
python scripts\refresh_stats.py
```

Notes:
- `uv venv` creates `.venv` in the project root.
- `uv sync` installs dependencies from `pyproject.toml` into `.venv`.
- After activation, `python` and `pip` point to the project environment.


# 🧙 Fantasy Football Wizard — Project Checklist

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

---

## 1. Tech Stack

### Core
- **Python 3.10+**
- **LLM**: Phi-3-mini (via `llama-cpp-python`)
- **Embeddings**: `sentence-transformers`
- **Vector Database**: Chroma (local)
- **UI**: Streamlit

### Data & Processing
- **Structured Data**: `nfl_data_py`, CSVs, Pandas
- **Unstructured Data**: Fantasy news articles / blurbs
- **Scheduling**: Cron jobs or manual refresh scripts

---

## 2. Repository Structure

```
fantasy-football-wizard/
│
├── data/
│   ├── raw/
│   │   ├── stats/
│   │   ├── projections/
│   │   ├── injuries/
│   │   └── news/
│   ├── processed/
│   │   ├── player_stats.parquet
│   │   ├── projections.parquet
│   │   └── injuries.parquet
│
├── embeddings/
│   ├── build_embeddings.py
│   └── chroma_db/
│
├── retrieval/
│   ├── news_retriever.py
│   └── filters.py
│
├── llm/
│   ├── model_loader.py
│   ├── prompt_templates.py
│   └── inference.py
│
├── pipeline/
│   ├── entity_extraction.py
│   ├── context_builder.py
│   └── decision_engine.py
│
├── app/
│   └── streamlit_app.py
│
├── scripts/
│   ├── refresh_stats.py
│   ├── refresh_news.py
│   └── refresh_embeddings.py
│
├── README.md
└── requirements.txt
```

---

## 3. Structured Data Ingestion (Stats & Projections)

**Recommended Tools**
- `nfl_data_py` for raw data ingestion
- `pandas` for aggregation and feature engineering
- `pyarrow` / Parquet for efficient local storage

**Why**
- Structured data requires exact filtering (week, opponent, player)
- Vector databases are intentionally NOT used for tabular data
- Pandas allows deterministic, debuggable data access

### 3.1 Player Stats

**Tool**
- `nfl_data_py`

**Steps**
- Pull weekly player statistics
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
- FantasyPros CSV exports
- Sleeper API
- ESPN / Yahoo (manual CSV)

**Steps**
- Normalize player names across sources
- Store projections by week

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
- Official NFL injury reports
- FantasyPros injury feed

**Steps**
- Parse practice participation reports
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
- FantasyPros blurbs
- Rotoworld
- Beat reporter articles

**Steps**
- Fetch news from the last 7–10 days only
- Tag each item with player name and date
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
  "player": "Jordan Love",
  "date": "2024-12-01",
  "source": "FantasyPros"
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
- Filter results by recency (≤ 7 days)
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

```
PLAYER COMPARISON

Jordan Love:
- Avg fantasy points (last 3 weeks): 18.4
- Projected points: 17.1
- Injury: Questionable → Full practice Friday
- Recent news:
  - "Packers plan to stay aggressive..."

Jared Goff:
- Avg fantasy points (last 3 weeks): 12.1
- Projected points: 14.3
- Injury: Healthy
- Recent news:
  - "Cold weather may impact passing..."


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
- `llama-cpp-python` for local inference
- JSON schema validation on outputs

**Why**
- Local inference reduces cost and dependency risk
- Schema validation prevents malformed responses


**Steps**
- Load the local LLM
- Inject assembled context
- Parse structured output for display

---

## 10. Streamlit Application
**Recommended Tools**
- Streamlit for rapid UI development
- Session state for caching results
- Optional: rate limiting on inference calls

**Why**
- Streamlit accelerates iteration and demos
- UI should remain thin; business logic stays backend-only


UI for interacting with the fantasy assistant.
**Features**
- Player selection UI
- Natural language question input
- Recommendation display
- Confidence visualization

**Optional Enhancements**
- Source citations
- Debug view showing injected context

---

## 11. Data Refresh Strategy

| Data Type | Refresh Frequency |
|---------|------------------|
| Player Stats | Weekly |
| Projections | Weekly |
| Injury Reports | Daily |
| News | Daily |
| Embeddings | On news refresh |

Use cron jobs to:
- Refresh stats weekly
- Refresh news daily
- Rebuild embeddings after news updates

---
## 12. Evaluation (Lightweight)

**Approaches**

- Compare recommendations against FantasyPros consensus
- Track agreement rates week-over-week
- Log confidence vs actual fantasy outcomes
---
## 13. Deployment

- Local Streamlit deployment
- Optional cloud VM (AWS Lightsail / EC2)
- Cron jobs for data refresh

---

## 14. Optional FastAPI Integration

While the current MVP uses Streamlit as both the UI and orchestration layer, adding **FastAPI** can improve separation of concerns and scalability.

### Why Use FastAPI?

FastAPI is **not a replacement** for your pipeline or LLM. Instead, it exposes your backend logic as **HTTP endpoints**, allowing multiple clients (Streamlit, mobile apps, Slack bots, etc.) to interact with your Fantasy Football Wizard.

Benefits include:

- **Separation of Concerns:** Streamlit only handles UI; FastAPI handles pipeline orchestration.
- **Reusability:** Multiple frontends can call the same backend endpoints.
- **Scalability:** Easier to deploy and scale in the cloud.
- **Debugging:** Endpoints can be tested independently with tools like Postman or `curl`.
- **Resume Value:** Demonstrates experience building a production-ready ML API.

### Example FastAPI Endpoint

```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.context_builder import build_context
from llm.inference import run_llm

app = FastAPI()

class RecommendationRequest(BaseModel):
    players: list[str]
    week: int

@app.post("/recommendation")
def get_recommendation(request: RecommendationRequest):
    context = build_context(request.players, request.week)
    response = run_llm(context)
    return response
--
