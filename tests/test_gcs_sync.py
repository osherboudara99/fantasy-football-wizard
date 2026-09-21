from pathlib import Path

import pytest

from pipeline.gcs_sync import UnsafeBlobPathError, sync_from_gcs


class _FakeBlob:
    def __init__(self, name, content=b"data"):
        self.name = name
        self.content = content

    def download_to_filename(self, path):
        Path(path).write_bytes(self.content)


class _FakeBucket:
    def __init__(self, blob_names):
        self._blobs = {name: _FakeBlob(name, content=name.encode()) for name in blob_names}

    def blob(self, name):
        return self._blobs[name]

    def list_blobs(self, prefix):
        return [blob for name, blob in self._blobs.items() if name.startswith(prefix)]


class _FakeClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        return self._bucket


def test_sync_from_gcs_downloads_processed_parquet_files(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.gcs_sync.PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr("pipeline.gcs_sync.CHROMA_DIR", tmp_path / "chroma_db")
    bucket = _FakeBucket([
        "data/processed/player_stats.parquet",
        "data/processed/projections.parquet",
        "data/processed/injuries.parquet",
        "data/processed/meta.parquet",
    ])

    sync_from_gcs("my-bucket", client=_FakeClient(bucket))

    for filename in ["player_stats.parquet", "projections.parquet", "injuries.parquet", "meta.parquet"]:
        assert (tmp_path / "processed" / filename).read_bytes() == f"data/processed/{filename}".encode()


def test_sync_from_gcs_downloads_the_full_chroma_directory_tree(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.gcs_sync.PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr("pipeline.gcs_sync.CHROMA_DIR", tmp_path / "chroma_db")
    bucket = _FakeBucket([
        "data/processed/player_stats.parquet",
        "data/processed/projections.parquet",
        "data/processed/injuries.parquet",
        "data/processed/meta.parquet",
        "embeddings/chroma_db/chroma.sqlite3",
        "embeddings/chroma_db/abcd-1234/data_level0.bin",
    ])

    sync_from_gcs("my-bucket", client=_FakeClient(bucket))

    assert (tmp_path / "chroma_db" / "chroma.sqlite3").exists()
    assert (tmp_path / "chroma_db" / "abcd-1234" / "data_level0.bin").exists()


def test_sync_from_gcs_uses_the_named_bucket(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.gcs_sync.PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr("pipeline.gcs_sync.CHROMA_DIR", tmp_path / "chroma_db")
    bucket = _FakeBucket([
        "data/processed/player_stats.parquet",
        "data/processed/projections.parquet",
        "data/processed/injuries.parquet",
        "data/processed/meta.parquet",
    ])
    seen = {}

    class _TrackingClient(_FakeClient):
        def bucket(self, name):
            seen["bucket_name"] = name
            return super().bucket(name)

    sync_from_gcs("my-bucket", client=_TrackingClient(bucket))
    assert seen["bucket_name"] == "my-bucket"


def test_sync_from_gcs_rejects_a_blob_name_that_escapes_chroma_dir(tmp_path, monkeypatch):
    """A blob outside our control (compromised bucket, bad refresh job) must not
    be able to write outside CHROMA_DIR via a `..` component in its name.
    """
    monkeypatch.setattr("pipeline.gcs_sync.PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr("pipeline.gcs_sync.CHROMA_DIR", tmp_path / "chroma_db")
    bucket = _FakeBucket([
        "data/processed/player_stats.parquet",
        "data/processed/projections.parquet",
        "data/processed/injuries.parquet",
        "data/processed/meta.parquet",
        "embeddings/chroma_db/../../escaped.txt",
    ])

    with pytest.raises(UnsafeBlobPathError):
        sync_from_gcs("my-bucket", client=_FakeClient(bucket))

    assert not (tmp_path / "escaped.txt").exists()
