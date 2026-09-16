# Chat Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two-player start/sit form with a general chat interface (start/sit, trade, single-player questions) that cites news sources and supports `@`-mention player search.

**Architecture:** A new `pipeline/chat_engine.py` generalizes `pipeline/decision_engine.py`'s orchestration (structured stats + news RAG → LLM) to an arbitrary number of players and a flexible answer shape, exposed as `POST /chat`. The frontend's picker form becomes a chat view built from small presentational components.

**Tech Stack:** FastAPI, Pydantic, Polars, Chroma, Anthropic SDK (`client.messages.parse`), React 19 (Vite) — no new dependencies on either side.

**Spec:** `docs/superpowers/specs/2026-09-09-chat-assistant-design.md`

## Global Constraints

- No new Python or npm dependencies — `@`-mention autocomplete and the chat UI are plain React; no new package needed for either side (AGENT_GOALS.md's dependency-approval rule).
- `mentioned_players` are exact names from `GET /players` (the existing `known_player_names()` list) — not player IDs. There is no player-ID concept at the API boundary anywhere in this codebase; don't introduce one.
- Never expose `ANTHROPIC_API_KEY` to the frontend — all LLM calls stay in `pipeline`/`llm` (CLAUDE.md).
- The frontend has no automated test suite today; frontend tasks verify via a manual dev-server check (`npm run dev` + browser), consistent with existing project convention. Backend tasks are TDD (pytest) as usual.
- `POST /recommendation` is removed once `POST /chat` exists — don't keep both.
- Draft strategy / ADP / consensus rankings are explicitly out of scope for this plan (deferred, see CLAUDE.md).
- Commit after every task (small, working increments; conventional-commit prefixes) per this user's standing `incremental-commits` preference.

---

### Task 1: Add `title` to news embedding metadata

**Files:**
- Modify: `embeddings/build_embeddings.py:41-49` (`_row_metadata`)
- Test: `tests/test_build_embeddings.py`

**Interfaces:**
- Produces: Chroma `news` collection documents now carry a `title` metadata field (string, `""` if absent) alongside the existing `player_id`/`player_name`/`source`/`link`/`published_at`/`published_ts`. Task 2 reads this field.

- [ ] **Step 1: Extend the existing metadata test**

In `tests/test_build_embeddings.py`, add an assertion to `test_build_embeddings_upserts_one_row_per_article_with_metadata`:

```python
    metadata = result["metadatas"][0]
    assert metadata["player_id"] == "00-1"
    assert metadata["player_name"] == "Jordan Love"
    assert metadata["title"] == "Jordan Love day-to-day"
    assert metadata["published_ts"] == int(datetime(2026, 8, 10, tzinfo=timezone.utc).timestamp())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_build_embeddings.py::test_build_embeddings_upserts_one_row_per_article_with_metadata -v`
Expected: FAIL with `KeyError: 'title'`

- [ ] **Step 3: Add `title` to `_row_metadata`**

In `embeddings/build_embeddings.py`, replace `_row_metadata`:

```python
def _row_metadata(row: dict) -> dict:
    published_at = row["published_at"]
    return {
        "player_id": row["player_id"] or "",
        "player_name": row["player_name"] or "",
        "title": row["title"] or "",
        "source": row["source"] or "",
        "link": row["link"] or "",
        "published_at": published_at.isoformat() if published_at is not None else "",
        "published_ts": int(published_at.timestamp()) if published_at is not None else 0,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_build_embeddings.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add embeddings/build_embeddings.py tests/test_build_embeddings.py
git commit -m "feat: embed article title into news metadata"
```

---

### Task 2: Change `retrieve_news` to return structured `NewsItem`s

**Files:**
- Modify: `retrieval/news_retriever.py`
- Test: `tests/test_news_retriever.py`

**Interfaces:**
- Consumes: Chroma metadata's `title`/`link`/`source`/`published_at` fields (Task 1).
- Produces: `NewsItem` (frozen dataclass: `title: str`, `snippet: str`, `link: str`, `source: str`, `published_at: str`) and `retrieve_news(...) -> list[NewsItem]` (was `list[str]`). Task 3 (context builder) and Task 4 (chat engine) both import `NewsItem` from here.

- [ ] **Step 1: Rewrite the test file for the new return type**

Replace `tests/test_news_retriever.py` in full:

```python
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import chromadb
import pytest

from retrieval.news_retriever import NewsItem, retrieve_news


def _stub_embed(text: str) -> list[float]:
    """Deterministic 3-dim stand-in for a real sentence-transformers encoding.

    Only used to populate the fixture collection (build_embeddings.py's job) -
    retrieve_news itself never embeds anything, see news_retriever's docstring.
    """
    return [float(len(text) % 7), float(sum(map(ord, text)) % 11), 1.0]


@pytest.fixture
def collection():
    """A uniquely-named collection per test - chromadb's EphemeralClient shares its
    underlying store across instances within a process, so a fixed name would leak
    documents between tests (and between this file and test_build_embeddings.py).
    """
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(f"news-test-{uuid4().hex}")


def _add(collection, doc_id, text, player_id, days_ago, title="", link="", source=""):
    published_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    collection.upsert(
        ids=[doc_id],
        embeddings=[_stub_embed(text)],
        documents=[text],
        metadatas=[{
            "player_id": player_id,
            "player_name": "Jordan Love",
            "title": title,
            "link": link,
            "source": source,
            "published_at": published_at.isoformat(),
            "published_ts": int(published_at.timestamp()),
        }],
    )


def test_retrieve_news_returns_empty_list_without_a_player_id(collection):
    assert retrieve_news(None, "Jordan Love", collection=collection) == []


def test_retrieve_news_returns_empty_list_when_collection_is_empty(collection):
    assert retrieve_news("00-1", "Jordan Love", collection=collection) == []


def test_retrieve_news_filters_to_the_requested_player(collection):
    _add(collection, "a", "Jordan Love news", "00-1", days_ago=1,
         title="Love news", link="https://example.com/a", source="ESPN")
    _add(collection, "b", "Jared Goff news", "00-2", days_ago=1)

    [item] = retrieve_news("00-1", "Jordan Love", collection=collection)
    assert item.snippet == "Jordan Love news"
    assert item.title == "Love news"
    assert item.link == "https://example.com/a"
    assert item.source == "ESPN"


def test_retrieve_news_filters_out_stale_articles(collection):
    _add(collection, "a", "Jordan Love fresh news", "00-1", days_ago=1)
    _add(collection, "b", "Jordan Love stale news", "00-1", days_ago=30)

    results = retrieve_news("00-1", "Jordan Love", max_age_days=7, collection=collection)
    assert [item.snippet for item in results] == ["Jordan Love fresh news"]


def test_retrieve_news_orders_most_recent_first(collection):
    _add(collection, "a", "Jordan Love three days ago", "00-1", days_ago=3)
    _add(collection, "b", "Jordan Love today", "00-1", days_ago=0)
    _add(collection, "c", "Jordan Love one day ago", "00-1", days_ago=1)

    results = retrieve_news("00-1", "Jordan Love", k=3, collection=collection)
    assert [item.snippet for item in results] == [
        "Jordan Love today", "Jordan Love one day ago", "Jordan Love three days ago"
    ]


def test_retrieve_news_respects_k_keeping_the_most_recent(collection):
    for days_ago in range(5):
        _add(collection, f"a{days_ago}", f"Jordan Love news {days_ago}d ago", "00-1", days_ago=days_ago)

    results = retrieve_news("00-1", "Jordan Love", k=2, collection=collection)
    assert [item.snippet for item in results] == ["Jordan Love news 0d ago", "Jordan Love news 1d ago"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_news_retriever.py -v`
Expected: FAIL — `ImportError: cannot import name 'NewsItem'`

- [ ] **Step 3: Add `NewsItem` and update `retrieve_news`**

Replace `retrieval/news_retriever.py` in full:

```python
"""Retrieve recent, player-tagged news items from Chroma (README §6.2).

Filters by metadata only (player_id + recency), then sorts by recency -
deliberately no query-time embedding: the candidate set is already scoped to
one player by the player_id filter, so a semantic similarity pass over it adds
little, and computing one would need sentence-transformers (torch) inside the
API request path. README §13 requires the opposite - embedding only happens in
the refresh job (embeddings/build_embeddings.py), never here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chromadb

CHROMA_DIR = Path(__file__).resolve().parent.parent / "embeddings" / "chroma_db"
COLLECTION_NAME = "news"
DEFAULT_K = 3
MAX_AGE_DAYS = 7


@dataclass(frozen=True)
class NewsItem:
    """One retrieved news article - everything the chat UI needs to cite it."""

    title: str
    snippet: str
    link: str
    source: str
    published_at: str


def _default_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve_news(
    player_id: str | None,
    player_name: str,
    k: int = DEFAULT_K,
    max_age_days: int = MAX_AGE_DAYS,
    collection=None,
) -> list[NewsItem]:
    """Top-k most recent news items for one player, newest first.

    `player_name` isn't used in the query itself - it's kept so this matches
    the `news_fn(player_id, player_name)` contract the context builder calls.
    `collection` is injectable so tests can use a fixture Chroma collection
    instead of the real on-disk index. Returns [] (never raises) when nothing
    has been embedded yet or the player has no recent tagged news.
    """
    if not player_id:
        return []

    collection = collection if collection is not None else _default_collection()
    if collection.count() == 0:
        return []

    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=max_age_days)).timestamp())
    result = collection.get(
        where={"$and": [{"player_id": player_id}, {"published_ts": {"$gte": cutoff_ts}}]},
        include=["documents", "metadatas"],
    )
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    ranked = sorted(
        zip(documents, metadatas), key=lambda pair: pair[1].get("published_ts", 0), reverse=True
    )
    return [
        NewsItem(
            title=metadata.get("title", ""),
            snippet=document,
            link=metadata.get("link", ""),
            source=metadata.get("source", ""),
            published_at=metadata.get("published_at", ""),
        )
        for document, metadata in ranked[:k]
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_news_retriever.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add retrieval/news_retriever.py tests/test_news_retriever.py
git commit -m "feat: return structured NewsItem from retrieve_news"
```

---

### Task 3: Update `context_builder` to consume `NewsItem`

**Files:**
- Modify: `pipeline/context_builder.py:60-89` (`_format_player`, type hints)
- Test: `tests/test_context_builder.py`

**Interfaces:**
- Consumes: `NewsItem` from `retrieval.news_retriever` (Task 2).
- Produces: `build_context()`'s output text is unchanged in shape (still one `"Recent news"` bullet per snippet) — only the `news_fn` contract's return type changes, from the caller's perspective.

- [ ] **Step 1: Update the affected test**

In `tests/test_context_builder.py`, add the import and replace `test_build_context_adds_a_news_bullet_per_snippet_when_news_fn_returns_some`:

```python
from retrieval.news_retriever import NewsItem
```

```python
def test_build_context_adds_a_news_bullet_per_snippet_when_news_fn_returns_some():
    def news_fn(player_id, player_name):
        if player_name != "Jordan Love":
            return []
        return [NewsItem(
            title="Packers stay aggressive",
            snippet="Packers plan to stay aggressive...",
            link="https://example.com/a",
            source="ESPN",
            published_at="2026-09-01T00:00:00+00:00",
        )]

    context = build_context(
        ["Jordan Love", "Jared Goff"], week=5, tables=_fixture_tables(), news_fn=news_fn
    )

    assert (
        "Jordan Love:\n- Avg fantasy points (last 3 weeks): 18.4\n"
        "- Projected points: 17.1\n- Injury: Questionable -> Full practice Friday\n"
        '- Recent news:\n  - "Packers plan to stay aggressive..."' in context
    )
    assert "Jared Goff:\n- Avg fantasy points (last 3 weeks): 12.1\n" \
        "- Projected points: 14.3\n- Injury: Healthy" in context
    assert "Recent news" not in context.split("Jared Goff:")[1]
```

(`test_build_context_omits_news_bullet_when_news_fn_returns_nothing` and `test_build_context_passes_player_id_to_news_fn` both use `[]`/no items and need no change.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_context_builder.py::test_build_context_adds_a_news_bullet_per_snippet_when_news_fn_returns_some -v`
Expected: FAIL with `AttributeError: 'NewsItem' object` mismatch (current code does `f'  - "{item}"'` on the whole object)

- [ ] **Step 3: Update `_format_player` to use `.snippet`**

In `pipeline/context_builder.py`, change the `Callable` import line and `_format_player`:

```python
from retrieval.news_retriever import NewsItem
```

```python
def _format_player(
    name: str,
    week: int,
    tables: dict[str, pl.DataFrame],
    news_fn: Callable[[str | None, str], list[NewsItem]] | None,
) -> str:
    """One player's block: name header, last-3-week avg, projection, injury status, news."""
    stats = tables["player_stats"].filter(
        (pl.col("player_name") == name) & (pl.col("week") == week)
    )
    if stats.height == 0:
        raise PlayerNotFoundError(f'No stats found for "{name}" in week {week}')
    stats_row = stats.row(0, named=True)
    avg_last3 = stats_row["avg_fantasy_points_last3"]
    player_id = stats_row.get("player_id")

    proj = _match_player(tables["projections"], name, player_id).filter(pl.col("week") == week)
    projected = proj.row(0, named=True)["projected_points"] if proj.height else None

    lines = [f"{name}:", f"- Avg fantasy points (last 3 weeks): {avg_last3:.1f}"]
    if projected is not None:
        lines.append(f"- Projected points: {projected:.1f}")
    lines.append(f"- Injury: {_format_injury(tables['injuries'], name, player_id)}")

    if news_fn is not None:
        news_items = news_fn(player_id, name)
        if news_items:
            lines.append("- Recent news:")
            lines.extend(f'  - "{item.snippet}"' for item in news_items)
    return "\n".join(lines)
```

Also update `build_context`'s own `news_fn` type hint the same way (`Callable[[str | None, str], list[NewsItem]] | None`).

- [ ] **Step 4: Run the full test file to verify it passes**

Run: `pytest tests/test_context_builder.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the decision engine tests too (they use the same `news_fn` contract)**

Run: `pytest tests/test_decision_engine.py -v`
Expected: PASS unchanged — its `_no_news` stub returns `[]`, unaffected by the type change.

- [ ] **Step 6: Commit**

```bash
git add pipeline/context_builder.py tests/test_context_builder.py
git commit -m "refactor: consume NewsItem.snippet in context builder"
```

---

### Task 4: Add `ChatAnswer`/`run_chat_llm` and `pipeline/chat_engine.py`

**Files:**
- Modify: `llm/interface.py`
- Create: `pipeline/chat_engine.py`
- Test: `tests/test_chat_engine.py`

**Interfaces:**
- Consumes: `build_context()` (Task 3), `retrieve_news`/`NewsItem` (Task 2), `pipeline.decision_engine.resolve_week`/`DecisionError`, `pipeline.entity_extraction.extract_players`.
- Produces:
  - `llm.interface.ChatAnswer` (Pydantic: `answer: str`, `recommendation: Recommendation | None`)
  - `llm.interface.run_chat_llm(context: str, message: str, history: list[dict[str, str]] | None = None) -> ChatAnswer`
  - `pipeline.chat_engine.NoPlayersFoundError(ValueError)`
  - `pipeline.chat_engine.ChatResult` (dataclass: `answer: str`, `sources: list[NewsItem]`, `players_discussed: list[str]`, `recommendation: Recommendation | None`, `week: int`, `context: str`)
  - `pipeline.chat_engine.resolve_chat_players(message: str, mentioned_players: list[str]) -> list[str]`
  - `pipeline.chat_engine.chat(message, mentioned_players=None, week=None, history=None, tables=None, news_fn=None) -> ChatResult`
  — Task 5 (the API layer) calls `chat()` and imports `NoPlayersFoundError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat_engine.py`:

```python
import polars as pl
import pytest

from llm.interface import ChatAnswer, Recommendation
from pipeline.chat_engine import NoPlayersFoundError, chat, resolve_chat_players
from pipeline.decision_engine import DecisionError
from retrieval.news_retriever import NewsItem


def _fixture_tables():
    player_stats = pl.DataFrame({
        "player_name": ["Jordan Love", "Jared Goff", "Bo Nix"],
        "week": [5, 5, 5],
        "avg_fantasy_points_last3": [18.4, 12.1, 15.0],
    })
    projections = pl.DataFrame({
        "player_name": ["Jordan Love", "Jared Goff", "Bo Nix"],
        "week": [5, 5, 5],
        "projected_points": [17.1, 14.3, 16.0],
    })
    injuries = pl.DataFrame({
        "player_name": ["Jordan Love"],
        "status": ["Questionable"],
        "practice_level": ["Full practice Friday"],
    })
    return {"player_stats": player_stats, "projections": projections, "injuries": injuries}


def _news_item(name, link="https://example.com/a"):
    return NewsItem(
        title=f"{name} update", snippet=f"{name} update\nDetails.",
        link=link, source="ESPN", published_at="2026-09-01T00:00:00+00:00",
    )


@pytest.fixture
def stub_run_chat_llm(monkeypatch):
    """Capture what chat() sends to the LLM without spending a real API call."""
    calls = {}

    def fake(context, message, history=None):
        calls.update(context=context, message=message, history=history)
        return ChatAnswer(answer="Start Jordan Love, he has the better matchup.")

    monkeypatch.setattr("pipeline.chat_engine.run_chat_llm", fake)
    return calls


def test_resolve_chat_players_prefers_mentions_then_extracted_text(monkeypatch):
    monkeypatch.setattr("pipeline.chat_engine.extract_players", lambda message: ["Bo Nix"])
    result = resolve_chat_players("Should I start him over Bo Nix?", ["Jordan Love"])
    assert result == ["Jordan Love", "Bo Nix"]


def test_resolve_chat_players_deduplicates_case_insensitively(monkeypatch):
    monkeypatch.setattr("pipeline.chat_engine.extract_players", lambda message: ["jordan love"])
    result = resolve_chat_players("Jordan Love?", ["Jordan Love"])
    assert result == ["Jordan Love"]


def test_resolve_chat_players_raises_when_nothing_found(monkeypatch):
    monkeypatch.setattr("pipeline.chat_engine.extract_players", lambda message: [])
    with pytest.raises(NoPlayersFoundError):
        resolve_chat_players("How's the waiver wire looking?", [])


def test_chat_builds_context_for_all_resolved_players(stub_run_chat_llm):
    result = chat(
        "Should I start Jordan Love or Jared Goff?",
        mentioned_players=["Jordan Love", "Jared Goff"],
        week=5,
        tables=_fixture_tables(),
        news_fn=lambda *_: [],
    )
    assert "Jordan Love:" in stub_run_chat_llm["context"]
    assert "Jared Goff:" in stub_run_chat_llm["context"]
    assert result.players_discussed == ["Jordan Love", "Jared Goff"]
    assert result.week == 5
    assert result.answer == "Start Jordan Love, he has the better matchup."


def test_chat_passes_history_through_to_the_llm(stub_run_chat_llm):
    history = [{"role": "user", "content": "Who's playing this week?"}]
    chat(
        "What about Bo Nix?",
        mentioned_players=["Bo Nix"],
        week=5,
        history=history,
        tables=_fixture_tables(),
        news_fn=lambda *_: [],
    )
    assert stub_run_chat_llm["history"] == history


def test_chat_collects_and_deduplicates_sources_across_players(stub_run_chat_llm):
    shared = _news_item("shared", link="https://example.com/shared")

    def news_fn(player_id, player_name):
        return [shared] if player_name in ("Jordan Love", "Jared Goff") else []

    result = chat(
        "Should I start Jordan Love or Jared Goff?",
        mentioned_players=["Jordan Love", "Jared Goff"],
        week=5,
        tables=_fixture_tables(),
        news_fn=news_fn,
    )
    assert result.sources == [shared]


def test_chat_passes_through_a_recommendation_when_the_llm_returns_one(monkeypatch):
    rec = Recommendation(
        start="Jordan Love", bench="Jared Goff", confidence=0.8,
        key_factors=["Better matchup"], risk_factors=[],
    )
    monkeypatch.setattr(
        "pipeline.chat_engine.run_chat_llm",
        lambda *_a, **_k: ChatAnswer(answer="Start Jordan Love.", recommendation=rec),
    )
    result = chat(
        "Should I start Jordan Love or Jared Goff?",
        mentioned_players=["Jordan Love", "Jared Goff"],
        week=5,
        tables=_fixture_tables(),
        news_fn=lambda *_: [],
    )
    assert result.recommendation == rec


def test_chat_leaves_recommendation_none_for_a_general_question(stub_run_chat_llm):
    result = chat(
        "How many points will Jordan Love score?",
        mentioned_players=["Jordan Love"],
        week=5,
        tables=_fixture_tables(),
        news_fn=lambda *_: [],
    )
    assert result.recommendation is None


def test_chat_rejects_a_recommendation_naming_an_undiscussed_player(monkeypatch):
    rec = Recommendation(
        start="Somebody Else", bench="Jared Goff", confidence=0.8,
        key_factors=[], risk_factors=[],
    )
    monkeypatch.setattr(
        "pipeline.chat_engine.run_chat_llm",
        lambda *_a, **_k: ChatAnswer(answer="x", recommendation=rec),
    )
    with pytest.raises(DecisionError):
        chat(
            "Should I start Jordan Love or Jared Goff?",
            mentioned_players=["Jordan Love", "Jared Goff"],
            week=5,
            tables=_fixture_tables(),
            news_fn=lambda *_: [],
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_chat_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.chat_engine'`

- [ ] **Step 3: Add `ChatAnswer` and `run_chat_llm` to `llm/interface.py`**

Append to `llm/interface.py` (after the existing `Recommendation` class and before `run_llm`):

```python
class ChatAnswer(BaseModel):
    answer: str = Field(description="Prose answer to the user's fantasy football question")
    recommendation: Recommendation | None = Field(
        default=None,
        description=(
            "A start/bench recommendation, populated only when the question is a "
            "start/sit or flex-style comparison; left unset for any other question."
        ),
    )


CHAT_SYSTEM_PROMPT = (
    "You are a fantasy football analyst answering questions in a chat interface. "
    "Use only the provided context — do not rely on outside knowledge of players. "
    "Explain your reasoning clearly. If, and only if, the question is a start/sit "
    "or flex-style comparison between players, also fill in `recommendation` with "
    "a start/bench pick, a confidence between 0 and 1, and key/risk factors; for "
    "any other question (a single player's outlook, a trade evaluation, or a "
    "general question), leave `recommendation` unset and answer only in `answer`."
)


def run_chat_llm(
    context: str,
    message: str,
    history: list[dict[str, str]] | None = None,
) -> ChatAnswer:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = anthropic.Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    messages = [dict(turn) for turn in (history or [])]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"})

    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=CHAT_SYSTEM_PROMPT,
        messages=messages,
        output_format=ChatAnswer,
    )
    return response.parsed_output
```

- [ ] **Step 4: Create `pipeline/chat_engine.py`**

```python
"""Chat engine: generalizes decision_engine.decide() beyond a fixed two-player
start/sit question (docs/superpowers/specs/2026-09-09-chat-assistant-design.md).

Any number of @-mentioned or free-text-named players; the response only
carries a start/bench recommendation when the LLM judges the question to be
shaped that way, and always carries the news sources it drew on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl

from llm.interface import ChatAnswer, Recommendation, run_chat_llm
from pipeline.context_builder import build_context
from pipeline.decision_engine import DecisionError, resolve_week
from pipeline.entity_extraction import extract_players
from retrieval.news_retriever import NewsItem, retrieve_news


class NoPlayersFoundError(ValueError):
    """Raised when neither @-mentions nor free text name any known player."""


@dataclass
class ChatResult:
    """A chat answer plus everything that produced it, for the API/UI to render."""

    answer: str
    sources: list[NewsItem]
    players_discussed: list[str]
    recommendation: Recommendation | None
    week: int
    context: str


def resolve_chat_players(message: str, mentioned_players: list[str]) -> list[str]:
    """@-mentions first (reliable, no fuzzy matching), then any additional names
    found in the free text, de-duplicated case-insensitively with mentions first.
    """
    seen: set[str] = set()
    resolved: list[str] = []
    for name in [*mentioned_players, *extract_players(message)]:
        key = name.strip().casefold()
        if key not in seen:
            seen.add(key)
            resolved.append(name)
    if not resolved:
        raise NoPlayersFoundError(
            "No known players mentioned - @-mention a player or name them in your message."
        )
    return resolved


def _check_recommendation(recommendation: Recommendation | None, players: list[str]) -> None:
    """A recommendation may only name players actually discussed this turn.

    Subset, not exact-set equality (unlike decision_engine's two-player check):
    chat can discuss 3+ players and still recommend a start/bench pair from
    among them.
    """
    if recommendation is None:
        return
    answered = {recommendation.start.strip().casefold(), recommendation.bench.strip().casefold()}
    known = {name.strip().casefold() for name in players}
    if not answered <= known:
        raise DecisionError(
            f"LLM recommended players not among those discussed: {sorted(answered - known)}"
        )


def chat(
    message: str,
    mentioned_players: list[str] | None = None,
    week: int | None = None,
    history: list[dict[str, str]] | None = None,
    tables: dict[str, pl.DataFrame] | None = None,
    news_fn: Callable[[str | None, str], list[NewsItem]] | None = None,
) -> ChatResult:
    """Answer any fantasy-football question about the resolved players.

    `tables`/`news_fn` are injectable for tests, matching decision_engine.decide().
    `news_fn` defaults to the real Chroma-backed retriever; every NewsItem it
    returns is collected (de-duplicated by link) into the response's sources.
    """
    mentioned_players = mentioned_players or []
    history = history or []
    players = resolve_chat_players(message, mentioned_players)
    resolved_week = resolve_week(message, week)

    news_fn = news_fn if news_fn is not None else retrieve_news
    collected: list[NewsItem] = []

    def _tracking_news_fn(player_id: str | None, player_name: str) -> list[NewsItem]:
        items = news_fn(player_id, player_name)
        collected.extend(items)
        return items

    context = build_context(players, resolved_week, tables=tables, news_fn=_tracking_news_fn)
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

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_chat_engine.py -v`
Expected: PASS (all 9 tests)

- [ ] **Step 6: Run the full backend test suite to check for regressions**

Run: `pytest -v`
Expected: PASS (existing `test_decision_engine.py`, `test_context_builder.py`, `test_news_retriever.py`, `test_build_embeddings.py` all still pass; `test_api.py` still targets `/recommendation` until Task 5)

- [ ] **Step 7: Manually smoke-test `run_chat_llm` against the real API**

This exercises the nested-optional-model structured output (`Recommendation | None`) against the real Anthropic API before anything depends on it — run once with `ANTHROPIC_API_KEY` set:

```bash
python -c "
from dotenv import load_dotenv
load_dotenv()
from llm.interface import run_chat_llm
print(run_chat_llm('Player A: 18 pts avg. Player B: 12 pts avg.', 'Who should I start?'))
print(run_chat_llm('Player A: 18 pts avg.', 'How many points will Player A score?'))
"
```
Expected: first call's `recommendation` is populated; second call's `recommendation` is `None`. If Anthropic rejects the schema, adjust `ChatAnswer` (e.g. drop the union default) before proceeding — do not move to Task 5 until this passes.

- [ ] **Step 8: Commit**

```bash
git add llm/interface.py pipeline/chat_engine.py tests/test_chat_engine.py
git commit -m "feat: add chat engine and ChatAnswer LLM output"
```

---

### Task 5: Add `POST /chat`, remove `POST /recommendation`

**Files:**
- Modify: `api/main.py` (full rewrite of request/response models and routes)
- Modify: `tests/test_api.py` (full rewrite for `/chat`)

**Interfaces:**
- Consumes: `pipeline.chat_engine.chat`/`NoPlayersFoundError`/`ChatResult` (Task 4), `pipeline.context_builder.PlayerNotFoundError`, `pipeline.decision_engine.DataUnavailableError`/`DecisionError`, `pipeline.entity_extraction.known_player_names`.
- Produces: `POST /chat` (see spec's request/response JSON shape); `GET /players` and `GET /health` unchanged. Task 6 (`frontend/src/api.js`) calls this endpoint.

- [ ] **Step 1: Replace `tests/test_api.py` in full**

```python
import pytest
from fastapi.testclient import TestClient

from api.main import MAX_HISTORY_TURNS, MAX_MENTIONED_PLAYERS, MAX_QUESTION_LENGTH, app
from llm.interface import Recommendation
from pipeline.chat_engine import ChatResult, NoPlayersFoundError
from pipeline.context_builder import PlayerNotFoundError
from pipeline.decision_engine import DataUnavailableError, DecisionError
from retrieval.news_retriever import NewsItem

client = TestClient(app)


def _fixture_chat_result(recommendation=None):
    return ChatResult(
        answer="Start Jordan Love, he has the better matchup.",
        sources=[NewsItem(
            title="Love update", snippet="Love update\nDetails.", link="https://example.com/a",
            source="ESPN", published_at="2026-09-01T00:00:00+00:00",
        )],
        players_discussed=["Jordan Love", "Jared Goff"],
        recommendation=recommendation,
        week=5,
        context="PLAYER COMPARISON\n\nJordan Love:\n- Avg fantasy points (last 3 weeks): 18.4",
    )


@pytest.fixture
def stub_chat(monkeypatch):
    """Replace the pipeline call so the endpoint is tested without an API call."""
    calls = {}

    def fake_chat(message, mentioned_players=None, week=None, history=None, **_):
        calls.update(
            message=message, mentioned_players=mentioned_players, week=week, history=history
        )
        return _fixture_chat_result()

    monkeypatch.setattr("api.main.chat", fake_chat)
    return calls


def test_post_chat_returns_the_answer_as_json(stub_chat):
    response = client.post(
        "/chat", json={"message": "Should I start Jordan Love?", "mentioned_players": ["Jordan Love"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Start Jordan Love, he has the better matchup."
    assert body["players_discussed"] == ["Jordan Love", "Jared Goff"]
    assert body["sources"][0]["link"] == "https://example.com/a"
    assert body["recommendation"] is None
    assert body["week"] == 5
    assert body["context"].startswith("PLAYER COMPARISON")


def test_post_chat_includes_a_recommendation_when_present(monkeypatch):
    rec = Recommendation(
        start="Jordan Love", bench="Jared Goff", confidence=0.8, key_factors=["x"], risk_factors=[]
    )
    monkeypatch.setattr("api.main.chat", lambda *_a, **_k: _fixture_chat_result(recommendation=rec))
    response = client.post(
        "/chat", json={"message": "Start/sit?", "mentioned_players": ["Jordan Love", "Jared Goff"]}
    )

    body = response.json()
    assert body["recommendation"]["start"] == "Jordan Love"
    assert body["recommendation"]["confidence"] == 0.8


def test_post_chat_passes_mentioned_players_week_and_history(stub_chat):
    client.post("/chat", json={
        "message": "What about now?",
        "mentioned_players": ["Jordan Love"],
        "week": 5,
        "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    })
    assert stub_chat["mentioned_players"] == ["Jordan Love"]
    assert stub_chat["week"] == 5
    assert stub_chat["history"] == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}
    ]


def test_post_chat_defaults_mentioned_players_and_history_to_empty(stub_chat):
    client.post("/chat", json={"message": "How's the waiver wire?"})
    assert stub_chat["mentioned_players"] == []
    assert stub_chat["history"] == []


@pytest.fixture
def forbid_chat(monkeypatch):
    """Assert the pipeline is never entered - a rejected request costs no LLM call."""
    def fail(*_a, **_k):
        raise AssertionError("chat() should not be called for an invalid request")

    monkeypatch.setattr("api.main.chat", fail)


def test_post_chat_rejects_an_overlong_message(forbid_chat):
    response = client.post("/chat", json={"message": "a" * (MAX_QUESTION_LENGTH + 1)})
    assert response.status_code == 422


def test_post_chat_rejects_too_many_mentioned_players(forbid_chat):
    response = client.post("/chat", json={
        "message": "flex options?",
        "mentioned_players": [f"Player {i}" for i in range(MAX_MENTIONED_PLAYERS + 1)],
    })
    assert response.status_code == 422


def test_post_chat_rejects_too_much_history(forbid_chat):
    history = [{"role": "user", "content": "hi"}] * (MAX_HISTORY_TURNS + 1)
    response = client.post("/chat", json={"message": "hi", "history": history})
    assert response.status_code == 422


def test_post_chat_rejects_an_invalid_history_role(forbid_chat):
    response = client.post("/chat", json={"message": "hi", "history": [{"role": "system", "content": "x"}]})
    assert response.status_code == 422


def test_no_players_found_is_a_400(monkeypatch):
    def raise_no_players(*_a, **_k):
        raise NoPlayersFoundError("No known players mentioned")

    monkeypatch.setattr("api.main.chat", raise_no_players)
    response = client.post("/chat", json={"message": "How's the waiver wire?"})
    assert response.status_code == 400


def test_missing_data_is_a_503_that_hides_server_paths(monkeypatch):
    """A failed refresh is an outage, not the caller's fault - and shouldn't leak paths."""
    def raise_unavailable(*_a, **_k):
        raise DataUnavailableError(
            "data/processed/player_stats.parquet is empty - run python scripts/refresh_stats.py"
        )

    monkeypatch.setattr("api.main.chat", raise_unavailable)
    response = client.post(
        "/chat", json={"message": "Should I start Jordan Love?", "mentioned_players": ["Jordan Love"]}
    )
    assert response.status_code == 503
    assert "parquet" not in response.json()["detail"]


def test_unknown_player_is_a_404_not_a_500(monkeypatch):
    def raise_not_found(*_a, **_k):
        raise PlayerNotFoundError('No stats found for "Nobody Here" in week 5')

    monkeypatch.setattr("api.main.chat", raise_not_found)
    response = client.post(
        "/chat", json={"message": "Should I start Nobody Here?", "mentioned_players": ["Nobody Here"]}
    )
    assert response.status_code == 404
    assert "Nobody Here" in response.json()["detail"]


def test_decision_error_is_a_400(monkeypatch):
    def raise_decision_error(*_a, **_k):
        raise DecisionError("LLM answered about someone else")

    monkeypatch.setattr("api.main.chat", raise_decision_error)
    response = client.post(
        "/chat", json={"message": "Should I start Jordan Love?", "mentioned_players": ["Jordan Love"]}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "LLM answered about someone else"


def test_get_players_returns_the_known_name_universe(monkeypatch):
    monkeypatch.setattr("api.main.known_player_names", lambda: ["Jordan Love", "Jared Goff"])
    response = client.get("/players")

    assert response.status_code == 200
    assert response.json() == ["Jordan Love", "Jared Goff"]


def test_health_check():
    assert client.get("/health").json() == {"status": "ok"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `api.main` has no `/chat` route yet, and no `chat`/`MAX_HISTORY_TURNS`/`MAX_MENTIONED_PLAYERS` symbols.

- [ ] **Step 3: Replace `api/main.py` in full**

```python
"""FastAPI layer over the chat pipeline (README §14, chat design spec).

Run locally:
    uvicorn api.main:app --reload

The API owns no logic of its own: it validates input, calls `chat()`, and maps
pipeline errors onto status codes. Everything else - data access, context
assembly, the Anthropic key - stays server-side in the pipeline modules.
"""
from __future__ import annotations

import os

from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints

from pipeline.chat_engine import NoPlayersFoundError, chat
from pipeline.context_builder import PlayerNotFoundError
from pipeline.decision_engine import DataUnavailableError, DecisionError
from pipeline.entity_extraction import known_player_names

# The frontend is a separate origin in dev (Vite on 5173) and in prod (Cloudflare
# Pages), so the allowed origins have to be configurable per deployment.
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

# Every request that reaches chat() costs a paid LLM call whose price scales with
# the input. Cap the message, each history turn, each player name, and the number
# of mentions/history turns so a caller can't turn one request into a six-figure-
# token bill.
MAX_QUESTION_LENGTH = 500
MAX_PLAYER_NAME_LENGTH = 100
MAX_MENTIONED_PLAYERS = 10
MAX_HISTORY_TURNS = 20

load_dotenv()

app = FastAPI(title="Fantasy Football Wizard", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", DEFAULT_ORIGINS).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(max_length=MAX_QUESTION_LENGTH)]


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(max_length=MAX_QUESTION_LENGTH)]
    mentioned_players: list[Annotated[str, StringConstraints(max_length=MAX_PLAYER_NAME_LENGTH)]] = (
        Field(default_factory=list, max_length=MAX_MENTIONED_PLAYERS)
    )
    week: int | None = Field(
        default=None, description="Defaults to the week the processed data describes"
    )
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class SourceItem(BaseModel):
    title: str
    link: str
    source: str
    published_at: str


class RecommendationPayload(BaseModel):
    start: str
    bench: str
    confidence: float
    key_factors: list[str]
    risk_factors: list[str]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    players_discussed: list[str]
    recommendation: RecommendationPayload | None
    week: int
    # The assembled context travels with the answer so the UI can show what the
    # answer was actually based on (README §10's debug view).
    context: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe - also what Cloud Run hits before serving traffic."""
    return {"status": "ok"}


@app.get("/players")
def players() -> list[str]:
    """Every player the processed data knows about, for the frontend's @-mention search."""
    return known_player_names()


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Answer any fantasy-football question about the mentioned/named players."""
    try:
        result = chat(
            request.message,
            mentioned_players=request.mentioned_players,
            week=request.week,
            history=[turn.model_dump() for turn in request.history],
        )
    except NoPlayersFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PlayerNotFoundError as exc:
        # Unknown player or a week the data doesn't cover: the caller's input is
        # wrong, not the server - 404 rather than a 500 stack trace.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        # The refresh job hasn't populated data/ - nothing the caller can fix, and
        # the underlying message names server paths, so don't echo it back.
        raise HTTPException(
            status_code=503, detail="Player data is unavailable; try again later."
        ) from exc
    except DecisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceItem(title=s.title, link=s.link, source=s.source, published_at=s.published_at)
            for s in result.sources
        ],
        players_discussed=result.players_discussed,
        recommendation=(
            RecommendationPayload(**result.recommendation.model_dump())
            if result.recommendation is not None
            else None
        ),
        week=result.week,
        context=result.context,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full backend test suite**

Run: `pytest -v`
Expected: PASS across the whole suite.

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat: add POST /chat, remove POST /recommendation"
```

---

### Task 6: Frontend — `fetchChat` in `api.js`

**Files:**
- Modify: `frontend/src/api.js`

**Interfaces:**
- Produces: `fetchChat({ message, mentionedPlayers, week, history }) -> Promise<ChatResponse>` (JSON shape matches Task 5's `ChatResponse`). Task 11 (`App.jsx`) calls this. `fetchPlayers` is unchanged.

- [ ] **Step 1: Replace `frontend/src/api.js` in full**

```javascript
// Thin client over the FastAPI backend (../../api/main.py). No orchestration or
// business logic here - just request/response shaping for the endpoints the UI
// needs.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function parseErrorDetail(response) {
  try {
    const body = await response.json()
    return body.detail || response.statusText
  } catch {
    return response.statusText
  }
}

export async function fetchPlayers() {
  const response = await fetch(`${API_BASE_URL}/players`)
  if (!response.ok) {
    throw new Error(await parseErrorDetail(response))
  }
  return response.json()
}

export async function fetchChat({ message, mentionedPlayers, week, history }) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      mentioned_players: mentionedPlayers ?? [],
      week: week ?? null,
      history: history ?? [],
    }),
  })
  if (!response.ok) {
    throw new Error(await parseErrorDetail(response))
  }
  return response.json()
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/api.js
git commit -m "feat: add fetchChat client, remove fetchRecommendation"
```

(No automated frontend tests exist in this project; `fetchChat` is exercised manually in Task 13.)

---

### Task 7: Frontend — rename `ResultCard.jsx` to `DecisionCard.jsx`

**Files:**
- Create: `frontend/src/DecisionCard.jsx`
- Delete: `frontend/src/ResultCard.jsx`

**Interfaces:**
- Produces: `export default function DecisionCard({ start, bench, confidence, key_factors, risk_factors })` — a pure presentational component with no `week`/`context` props (those move to `MessageBubble` in Task 10, since a chat message wraps more than one card).

- [ ] **Step 1: Create `frontend/src/DecisionCard.jsx`**

```jsx
function ConfidenceBar({ confidence }) {
  const pct = Math.round(confidence * 100)
  return (
    <div className="confidence">
      <div className="confidence-label">
        <span>Confidence</span>
        <span>{pct}%</span>
      </div>
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default function DecisionCard({ start, bench, confidence, key_factors, risk_factors }) {
  return (
    <div className="decision-card">
      <div className="verdict">
        <div className="verdict-start">
          <span className="verdict-tag">START</span>
          <span className="verdict-name">{start}</span>
        </div>
        <div className="verdict-bench">
          <span className="verdict-tag">BENCH</span>
          <span className="verdict-name">{bench}</span>
        </div>
      </div>

      <ConfidenceBar confidence={confidence} />

      {key_factors?.length > 0 && (
        <div className="factors">
          <h3>Key factors</h3>
          <ul>
            {key_factors.map((factor) => (
              <li key={factor}>{factor}</li>
            ))}
          </ul>
        </div>
      )}

      {risk_factors?.length > 0 && (
        <div className="factors risk">
          <h3>Risk factors</h3>
          <ul>
            {risk_factors.map((factor) => (
              <li key={factor}>{factor}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Delete the old file**

```bash
git rm frontend/src/ResultCard.jsx
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/DecisionCard.jsx
git commit -m "refactor: rename ResultCard to DecisionCard, drop week/context props"
```

---

### Task 8: Frontend — `SourceList.jsx`

**Files:**
- Create: `frontend/src/SourceList.jsx`

**Interfaces:**
- Produces: `export default function SourceList({ sources })` where `sources` is `ChatResponse.sources` (`{ title, link, source, published_at }[]`). Renders nothing when `sources` is empty. Consumed by `MessageBubble` (Task 10).

- [ ] **Step 1: Create `frontend/src/SourceList.jsx`**

```jsx
export default function SourceList({ sources }) {
  if (!sources || sources.length === 0) return null

  return (
    <details className="source-list">
      <summary>Sources ({sources.length})</summary>
      <ul>
        {sources.map((item) => (
          <li key={item.link}>
            <a href={item.link} target="_blank" rel="noreferrer">
              {item.title}
            </a>
            {item.source && <span className="source-meta"> — {item.source}</span>}
            {item.published_at && (
              <span className="source-meta">
                {' '}
                ({new Date(item.published_at).toLocaleDateString()})
              </span>
            )}
          </li>
        ))}
      </ul>
    </details>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/SourceList.jsx
git commit -m "feat: add SourceList component for cited news"
```

---

### Task 9: Frontend — `ChatInput.jsx` with `@`-mention autocomplete

**Files:**
- Create: `frontend/src/ChatInput.jsx`

**Interfaces:**
- Consumes: `players: string[]` (from `fetchPlayers()`, threaded in by `App.jsx`).
- Produces: `export default function ChatInput({ players, onSend, disabled })`. Calls `onSend(message: string, mentionedPlayers: string[])` on submit, then clears its own internal text/mention state. `App.jsx` (Task 11) owns no input state itself — it only receives the finished `(message, mentionedPlayers)` pair.

- [ ] **Step 1: Create `frontend/src/ChatInput.jsx`**

```jsx
import { useMemo, useState } from 'react'

// Matches an "@" that starts a mention token currently being typed - i.e. the
// last "@" in the text with no whitespace between it and the cursor.
function activeMentionQuery(text) {
  const at = text.lastIndexOf('@')
  if (at === -1) return null
  const afterAt = text.slice(at + 1)
  if (/\s/.test(afterAt)) return null
  return { start: at, query: afterAt }
}

export default function ChatInput({ players, onSend, disabled }) {
  const [text, setText] = useState('')
  const [mentionedPlayers, setMentionedPlayers] = useState([])

  const mention = useMemo(() => activeMentionQuery(text), [text])
  const suggestions = useMemo(() => {
    if (!mention) return []
    const query = mention.query.toLowerCase()
    return players.filter((name) => name.toLowerCase().includes(query)).slice(0, 8)
  }, [mention, players])

  function selectMention(name) {
    const before = text.slice(0, mention.start)
    setText(`${before}@${name} `)
    setMentionedPlayers((prev) => (prev.includes(name) ? prev : [...prev, name]))
  }

  function handleSubmit(event) {
    event.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed, mentionedPlayers)
    setText('')
    setMentionedPlayers([])
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      {suggestions.length > 0 && (
        <ul className="mention-suggestions">
          {suggestions.map((name) => (
            <li key={name}>
              <button type="button" onClick={() => selectMention(name)}>
                {name}
              </button>
            </li>
          ))}
        </ul>
      )}
      <textarea
        value={text}
        placeholder="Ask about start/sit, trades, or any player... (@ to mention)"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e)
          }
        }}
        disabled={disabled}
      />
      <button type="submit" disabled={disabled || !text.trim()}>
        {disabled ? 'Thinking…' : 'Send'}
      </button>
    </form>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/ChatInput.jsx
git commit -m "feat: add ChatInput with @-mention autocomplete"
```

---

### Task 10: Frontend — `MessageBubble.jsx`

**Files:**
- Create: `frontend/src/MessageBubble.jsx`

**Interfaces:**
- Consumes: `DecisionCard` (Task 7), `SourceList` (Task 8).
- Produces: `export default function MessageBubble({ role, content, sources, recommendation, context, error })`. `role` is `"user"` or `"assistant"`. `error` (bool) renders the bubble in an error style with no sources/decision card. Consumed by `App.jsx` (Task 11), one per entry in its `messages` array.

- [ ] **Step 1: Create `frontend/src/MessageBubble.jsx`**

```jsx
import DecisionCard from './DecisionCard'
import SourceList from './SourceList'

export default function MessageBubble({ role, content, sources, recommendation, context, error }) {
  const isUser = role === 'user'

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'} ${error ? 'message-error' : ''}`}>
      <p className="message-content">{content}</p>

      {recommendation && <DecisionCard {...recommendation} />}
      {!isUser && !error && <SourceList sources={sources} />}

      {!isUser && !error && context && (
        <details className="debug-view">
          <summary>Show injected context</summary>
          <pre>{context}</pre>
        </details>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/MessageBubble.jsx
git commit -m "feat: add MessageBubble for chat message rendering"
```

---

### Task 11: Frontend — rewrite `App.jsx` as the chat shell

**Files:**
- Modify: `frontend/src/App.jsx`

**Interfaces:**
- Consumes: `fetchPlayers`/`fetchChat` (Task 6), `ChatInput` (Task 9), `MessageBubble` (Task 10).
- Produces: the app's top-level component; no other file depends on it.

- [ ] **Step 1: Replace `frontend/src/App.jsx` in full**

```jsx
import { useEffect, useRef, useState } from 'react'
import { fetchChat, fetchPlayers } from './api'
import ChatInput from './ChatInput'
import MessageBubble from './MessageBubble'
import './App.css'

function App() {
  const [players, setPlayers] = useState([])
  const [playersError, setPlayersError] = useState(null)
  const [week, setWeek] = useState('')
  const [messages, setMessages] = useState([])
  const [submitting, setSubmitting] = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    fetchPlayers()
      .then(setPlayers)
      .catch((err) => setPlayersError(err.message))
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleSend(message, mentionedPlayers) {
    const history = messages.map(({ role, content }) => ({ role, content }))
    setMessages((prev) => [...prev, { role: 'user', content: message }])
    setSubmitting(true)
    try {
      const response = await fetchChat({
        message,
        mentionedPlayers,
        week: week ? Number(week) : null,
        history,
      })
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          recommendation: response.recommendation,
          context: response.context,
        },
      ])
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: err.message, error: true }])
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Fantasy Football Wizard</h1>
        <p className="subtitle">Ask about start/sit, trades, or any player.</p>
        <label className="week-label">
          Week <span className="optional">(optional, defaults to upcoming)</span>
          <input type="number" min="1" max="22" value={week} onChange={(e) => setWeek(e.target.value)} />
        </label>
      </header>

      {playersError && <p className="error">Couldn't load players: {playersError}</p>}

      <div className="message-list">
        {messages.map((message, i) => (
          <MessageBubble key={i} {...message} />
        ))}
        <div ref={scrollRef} />
      </div>

      <ChatInput players={players} onSend={handleSend} disabled={submitting} />
    </div>
  )
}

export default App
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat: rewrite App as a chat shell"
```

---

### Task 12: Frontend — visual design pass

**Files:**
- Modify: `frontend/src/App.css`

- [ ] **Step 1: Invoke the `frontend-design` skill** for guidance on typography, spacing, and color choices before touching CSS — this project's current styling is a plain form layout being replaced with a message-list/chat-bubble layout, and deserves an intentional visual pass rather than default browser styling.

- [ ] **Step 2: Rewrite `frontend/src/App.css`** to style: the header (title/subtitle/week input), `.message-list` (scrollable column), `.message-user`/`.message-assistant` bubbles (distinct alignment/background per the frontend-design skill's guidance), `.decision-card` (carried over from the old `.result-card` styles), `.source-list`, `.mention-suggestions` (positioned dropdown above/below the textarea), `.chat-input` (textarea + send button row), and `.message-error`. Reuse color/spacing values already established for `.decision-card`'s internals (`.verdict`, `.confidence-track`, `.factors`) from the current stylesheet rather than inventing a second palette.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.css
git commit -m "style: redesign chat UI layout and bubbles"
```

---

### Task 13: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Start the backend**

```bash
source .venv/bin/activate
uvicorn api.main:app --reload
```

- [ ] **Step 2: Start the frontend**

```bash
cd frontend && npm run dev
```

- [ ] **Step 3: Exercise the golden path in a browser**

Open the Vite dev URL. Type `@` and confirm the autocomplete dropdown filters as you type; select a player and confirm a token is inserted. Ask a two-player start/sit question (`@Player A vs @Player B, who should I start?`) and confirm: the answer renders as prose, a `DecisionCard` appears with a start/bench pick and confidence bar, a "Sources" section appears when the mentioned players have recent news, and the debug `<details>` shows the injected context.

- [ ] **Step 4: Exercise a general (non-start/sit) question**

Ask a single-player question (e.g. `How many points will @Player A score this week?`). Confirm the answer renders as prose with no `DecisionCard`.

- [ ] **Step 5: Exercise multi-turn history**

Ask a follow-up question referencing the prior answer (e.g. "What about his matchup next week?") without re-mentioning the player. Confirm the assistant's reply shows it retained context from the earlier turn (this is the actual test of `history` being threaded through correctly — a follow-up that only makes sense with context is the only way to observe it working).

- [ ] **Step 6: Exercise error handling**

Send a message with no `@`-mention and no known player name in the text (e.g. "How's the weather?"). Confirm a chat-bubble error appears (400) rather than a broken UI state.

- [ ] **Step 7: Refresh the page**

Confirm the conversation resets (frontend-held history only, per the approved design — no persistence is expected).

If any step fails, fix the relevant task's code, re-run that task's automated tests if applicable, and re-verify here before considering the plan complete.
