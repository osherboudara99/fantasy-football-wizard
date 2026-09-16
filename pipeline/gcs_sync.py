"""Download processed data + the Chroma index from GCS at container startup
(README §13). Cloud Run has no persistent local disk, so the processed
Parquet tables and the news embeddings index have to come from a bucket
instead of the local data/ and embeddings/ directories used in dev.

Only called when GCS_BUCKET is set (api/main.py's startup hook) - local dev
is completely unaffected.
"""
from __future__ import annotations

from pathlib import Path

from google.cloud import storage

from pipeline.context_builder import PROCESSED_DIR
from retrieval.news_retriever import CHROMA_DIR

PROCESSED_FILES = ["player_stats.parquet", "projections.parquet", "injuries.parquet"]
CHROMA_PREFIX = "embeddings/chroma_db/"


def log(msg: str) -> None:
    print(f"[gcs_sync] {msg}")


class UnsafeBlobPathError(ValueError):
    """Raised when a blob name would resolve outside CHROMA_DIR."""


def _resolve_chroma_destination(blob_name: str) -> Path:
    """Map a `CHROMA_PREFIX`-relative blob name to a path under CHROMA_DIR.

    A bucket blob is untrusted input - a `..` component or an absolute-looking
    suffix (e.g. `embeddings/chroma_db/../../etc/passwd`) could otherwise
    resolve outside CHROMA_DIR and let anyone with bucket-write access
    overwrite arbitrary files in the container.
    """
    relative = blob_name.removeprefix(CHROMA_PREFIX)
    destination = (CHROMA_DIR / relative).resolve()
    if destination != CHROMA_DIR.resolve() and CHROMA_DIR.resolve() not in destination.parents:
        raise UnsafeBlobPathError(f"blob {blob_name!r} resolves outside CHROMA_DIR")
    return destination


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
        destination = _resolve_chroma_destination(blob.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
    log(f"downloaded {len(blobs)} chroma_db files")
