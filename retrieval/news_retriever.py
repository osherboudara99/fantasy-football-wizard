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
