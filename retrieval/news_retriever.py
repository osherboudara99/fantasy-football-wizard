"""Retrieve recent, player-tagged news snippets from Chroma (README §6.2).

Metadata filtering (player_id + recency) narrows the candidate set before
semantic similarity ranks the top-k - keeps retrieval both relevant and fresh.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).resolve().parent.parent / "embeddings" / "chroma_db"
COLLECTION_NAME = "news"
DEFAULT_K = 3
MAX_AGE_DAYS = 7
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _default_embed_query(text: str) -> list[float]:
    """Lazily load the shared sentence-transformers model, then embed one query string."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model.encode([text]).tolist()[0]


def _default_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve_news(
    player_id: str | None,
    player_name: str,
    k: int = DEFAULT_K,
    max_age_days: int = MAX_AGE_DAYS,
    collection=None,
    embed_query: Callable[[str], list[float]] | None = None,
) -> list[str]:
    """Top-k recent news headlines/snippets for one player, most relevant first.

    `collection`/`embed_query` are injectable so tests can use a fixture Chroma
    collection and a stub encoder instead of the real on-disk index/model.
    Returns [] (never raises) when nothing has been embedded yet or the player
    has no recent tagged news - Phase 6 is additive, so a build that hasn't run
    the embeddings refresh must still produce a working (just news-free) context.
    """
    if not player_id:
        return []

    collection = collection if collection is not None else _default_collection()
    if collection.count() == 0:
        return []

    embed_query = embed_query if embed_query is not None else _default_embed_query
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=max_age_days)).timestamp())

    result = collection.query(
        query_embeddings=[embed_query(f"recent news about {player_name}")],
        n_results=k,
        where={"$and": [{"player_id": player_id}, {"published_ts": {"$gte": cutoff_ts}}]},
    )
    documents = result.get("documents") or [[]]
    return documents[0] if documents else []
