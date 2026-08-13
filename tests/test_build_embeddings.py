from datetime import datetime, timezone
from uuid import uuid4

import chromadb
import polars as pl
import pytest

import embeddings.build_embeddings as build_embeddings_module
from embeddings.build_embeddings import build_embeddings


class _StubModel:
    """Deterministic stand-in for SentenceTransformer.encode - no real model load."""

    def encode(self, texts):
        return [[float(len(t) % 7), float(sum(map(ord, t)) % 11), 1.0] for t in texts]


@pytest.fixture
def isolated_collection_name(monkeypatch):
    """A uniquely-named collection per test - chromadb's EphemeralClient shares its
    underlying store across instances within a process, so a fixed name would leak
    documents between tests (and between this file and test_news_retriever.py).
    """
    name = f"news-test-{uuid4().hex}"
    monkeypatch.setattr(build_embeddings_module, "COLLECTION_NAME", name)
    return name


def _news_fixture():
    return pl.DataFrame({
        "player_id": ["00-1"],
        "player_name": ["Jordan Love"],
        "title": ["Jordan Love day-to-day"],
        "description": ["Packers QB is questionable."],
        "link": ["https://example.com/love"],
        "source": ["ESPN"],
        "published_at": [datetime(2026, 8, 10, tzinfo=timezone.utc)],
    })


def test_build_embeddings_returns_zero_for_empty_news(isolated_collection_name):
    client = chromadb.EphemeralClient()
    count = build_embeddings(
        news=pl.DataFrame(schema=_news_fixture().schema), client=client, model=_StubModel()
    )
    assert count == 0


def test_build_embeddings_upserts_one_row_per_article_with_metadata(isolated_collection_name):
    client = chromadb.EphemeralClient()
    count = build_embeddings(news=_news_fixture(), client=client, model=_StubModel())
    assert count == 1

    collection = client.get_or_create_collection(isolated_collection_name)
    assert collection.count() == 1
    result = collection.get(ids=["00-1:https://example.com/love"], include=["documents", "metadatas"])
    assert result["documents"][0] == "Jordan Love day-to-day\nPackers QB is questionable."
    metadata = result["metadatas"][0]
    assert metadata["player_id"] == "00-1"
    assert metadata["player_name"] == "Jordan Love"
    assert metadata["published_ts"] == int(datetime(2026, 8, 10, tzinfo=timezone.utc).timestamp())


def test_build_embeddings_upsert_is_idempotent_on_rerun(isolated_collection_name):
    client = chromadb.EphemeralClient()
    build_embeddings(news=_news_fixture(), client=client, model=_StubModel())
    build_embeddings(news=_news_fixture(), client=client, model=_StubModel())

    collection = client.get_or_create_collection(isolated_collection_name)
    assert collection.count() == 1
