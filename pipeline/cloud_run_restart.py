"""Force the API's Cloud Run service onto a new revision after a refresh.

sync_from_gcs() (gcs_sync.py) only runs once, at container startup. A warm
instance never re-syncs on its own, so without this a scheduled refresh
writes a fresh snapshot to GCS that no running instance ever picks up - the
staleness caveat from the deployment design. Touching an annotation is a
no-op change to the service's own config, but Cloud Run treats any template
change as a new revision, which cold-starts and re-syncs from GCS.

Called by scripts/refresh_job_entrypoint.py after sync_to_gcs() succeeds.
"""
from __future__ import annotations

import time

from google.cloud import run_v2


def log(msg: str) -> None:
    print(f"[cloud_run_restart] {msg}")


def restart_cloud_run_service(
    project: str, region: str, service: str, client: run_v2.ServicesClient | None = None
) -> None:
    """Touch an annotation on `service` so Cloud Run rolls a new revision."""
    client = client if client is not None else run_v2.ServicesClient()
    name = f"projects/{project}/locations/{region}/services/{service}"

    current = client.get_service(name=name)
    current.template.annotations["ffw-refresh-ts"] = str(int(time.time()))

    operation = client.update_service(service=current)
    operation.result()
    log(f"rolled a new revision for {name}")
