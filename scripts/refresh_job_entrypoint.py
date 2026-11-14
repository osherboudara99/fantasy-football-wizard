"""Entrypoint for the Cloud Run refresh job (README §13's refresh pipeline).

Runs end-to-end: refresh stats -> refresh news -> rebuild embeddings -> push
the results to GCS_BUCKET -> force the API's Cloud Run service onto a fresh
revision so a warm instance doesn't keep serving what it had at its own last
cold start (pipeline/cloud_run_restart.py).

Each step is the same module the documented local commands run
(`python -m scripts.refresh_stats`, etc.), as its own subprocess so a failure
anywhere stops the job rather than uploading a partial/inconsistent
snapshot. Image: Dockerfile.refresh.
"""
from __future__ import annotations

import os
import subprocess
import sys

from pipeline.cloud_run_restart import restart_cloud_run_service
from pipeline.gcs_sync import sync_to_gcs


def log(msg: str) -> None:
    print(f"[refresh_job] {msg}")


def _run(module: str) -> None:
    log(f"running {module}")
    subprocess.run([sys.executable, "-m", module], check=True)


def main() -> None:
    bucket = os.environ["GCS_BUCKET"]
    service = os.environ["CLOUD_RUN_SERVICE"]
    region = os.environ["CLOUD_RUN_REGION"]
    project = os.environ["GCP_PROJECT"]

    _run("scripts.refresh_stats")
    _run("scripts.refresh_news")
    _run("embeddings.build_embeddings")

    log(f"uploading processed data + chroma index to gs://{bucket}")
    sync_to_gcs(bucket)

    log(f"restarting Cloud Run service {service!r} in {region} ({project})")
    restart_cloud_run_service(project, region, service)
    log("done")


if __name__ == "__main__":
    main()
