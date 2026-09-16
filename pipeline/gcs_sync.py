"""Download processed data + the Chroma index from GCS at container startup
(README §13). Cloud Run has no persistent local disk, so the processed
Parquet tables and the news embeddings index have to come from a bucket
instead of the local data/ and embeddings/ directories used in dev.

Only called when GCS_BUCKET is set (api/main.py's startup hook) - local dev
is completely unaffected.
"""
from __future__ import annotations

from google.cloud import storage

from pipeline.context_builder import PROCESSED_DIR
from retrieval.news_retriever import CHROMA_DIR

PROCESSED_FILES = ["player_stats.parquet", "projections.parquet", "injuries.parquet"]
CHROMA_PREFIX = "embeddings/chroma_db/"


def log(msg: str) -> None:
    print(f"[gcs_sync] {msg}")


def sync_from_gcs(bucket_name: str, client: storage.Client | None = None) -> None:
    """Download the processed tables and the Chroma index from `bucket_name`.

    Mirrors data/processed/*.parquet and embeddings/chroma_db/ under the same
    relative paths in the bucket, so the refresh job and the API agree on
    layout. `client` is injectable so tests don't need a real GCS bucket.
    """
    client = client if client is not None else storage.Client()
    bucket = client.bucket(bucket_name)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for filename in PROCESSED_FILES:
        bucket.blob(f"data/processed/{filename}").download_to_filename(
            str(PROCESSED_DIR / filename)
        )
        log(f"downloaded data/processed/{filename}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    blobs = [b for b in bucket.list_blobs(prefix=CHROMA_PREFIX) if not b.name.endswith("/")]
    for blob in blobs:
        destination = CHROMA_DIR / blob.name.removeprefix(CHROMA_PREFIX)
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
    log(f"downloaded {len(blobs)} chroma_db files")
