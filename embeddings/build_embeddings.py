"""Embed tagged news into Chroma for retrieval (README §4.2).

Usage:
    python -m embeddings.build_embeddings

Reads data/processed/news.parquet (written by scripts/refresh_news.py), embeds
each tagged article-player row with sentence-transformers, and upserts it into
the `news` collection at embeddings/chroma_db/. Embedding only ever runs here,
never in the API request path (README §13) - keeps torch out of the API image
and the request path fast.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import polars as pl
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "news"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def log(msg: str) -> None:
    """Print a progress message prefixed with the module name."""
    print(f"[build_embeddings] {msg}")


def _row_id(row: dict) -> str:
    """Stable id so re-running the refresh upserts instead of duplicating."""
    return f"{row['player_id']}:{row['link']}"


def _row_text(row: dict) -> str:
    """Chunk = one article's title + description, per player it's tagged with."""
    return f"{row['title']}\n{row['description']}".strip()


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


def build_embeddings(
    news: pl.DataFrame | None = None,
    client=None,
    model: SentenceTransformer | None = None,
) -> int:
    """Embed every tagged news row and upsert it into the Chroma `news` collection.

    Returns the number of rows embedded. `news`/`client`/`model` are injectable
    so tests can pass a fixture DataFrame, an in-memory Chroma client, and a stub
    encoder instead of touching disk or downloading a real model.
    """
    news = news if news is not None else pl.read_parquet(PROCESSED_DIR / "news.parquet")
    if news.height == 0:
        log("no tagged news to embed")
        return 0

    client = client if client is not None else chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = model if model is not None else SentenceTransformer(EMBEDDING_MODEL)

    rows = news.to_dicts()
    ids = [_row_id(r) for r in rows]
    documents = [_row_text(r) for r in rows]
    embeddings = model.encode(documents)
    embeddings = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
    metadatas = [_row_metadata(r) for r in rows]

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    log(f"embedded {len(rows)} news rows into '{COLLECTION_NAME}'")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build_embeddings()


if __name__ == "__main__":
    main()
