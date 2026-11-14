"""Embed tagged news into Chroma for retrieval (README §4.2).

Usage:
    python -m embeddings.build_embeddings

Reads data/processed/news.parquet (written by scripts/refresh_news.py), embeds
each tagged article-player row, and upserts it into the `news` collection at
embeddings/chroma_db/. Embedding only ever runs here, never in the API request
path (README §13).

Uses chromadb's bundled ONNX embedding function (same all-MiniLM-L6-v2 model,
via onnxruntime) rather than sentence-transformers - onnxruntime is already a
transitive dep of chromadb itself, so this needs no extra heavy dependency and
no torch, keeping the refresh job's own image lean too (it previously pulled
in torch just for this).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import polars as pl
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "news"


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
    embedding_fn=None,
) -> int:
    """Embed every tagged news row and upsert it into the Chroma `news` collection.

    Returns the number of rows embedded. `news`/`client`/`embedding_fn` are
    injectable so tests can pass a fixture DataFrame, an in-memory Chroma
    client, and a stub encoder instead of touching disk or downloading a real
    model. `embedding_fn` takes a list of strings and returns a list of vectors
    (chromadb's `EmbeddingFunction` call signature).
    """
    news = news if news is not None else pl.read_parquet(PROCESSED_DIR / "news.parquet")
    if news.height == 0:
        log("no tagged news to embed")
        return 0

    client = client if client is not None else chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    embedding_fn = embedding_fn if embedding_fn is not None else DefaultEmbeddingFunction()

    rows = news.to_dicts()
    ids = [_row_id(r) for r in rows]
    documents = [_row_text(r) for r in rows]
    # chromadb's EmbeddingFunction protocol returns a list of numpy arrays (one
    # per document) - upsert() accepts that natively, no float conversion needed
    # (wrapping each one in list() would unpack it into a list of np.float32
    # scalars, which upsert's own validation rejects).
    embeddings = embedding_fn(documents)
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
