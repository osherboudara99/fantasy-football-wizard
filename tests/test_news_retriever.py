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
