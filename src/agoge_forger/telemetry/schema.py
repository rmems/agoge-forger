"""F0 correlation envelope shared with blackwell-kernel-lab ``bkl.f0_correlation.v1``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "bkl.f0_correlation.v1"
RECORD_MARKER = "agoge_marker"
RECORD_PROFILE_WINDOW = "agoge_profile_window"
RECORD_PROFILE_REQUEST = "agoge_profile_window_request"
COLLECTOR_ID = "agoge-forger"
TORCH_BACKEND = "torch"
KNOWN_BACKENDS = ("torch", "nsight_systems", "nsight_compute", "cupti")
UNPINNED_REVISION = "unpinned"
TRAIN_PHASE = "train"


@dataclass(frozen=True)
class EnvelopeIdentity:
    run_id: str
    hostname: str
    gpu: Mapping[str, Any]
    collector_version: str
    cadence_ms: int | None = None


def utc_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def measurement(value: float | None, unit: str, status: str) -> dict[str, Any]:
    if status != "ok":
        return {"value": None, "unit": unit, "status": status}
    if value is None or not math.isfinite(float(value)):
        return {"value": None, "unit": unit, "status": "unavailable"}
    return {"value": float(value), "unit": unit, "status": "ok"}


def loss_measurement(logs: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not logs or "loss" not in logs:
        return None
    try:
        return measurement(float(logs["loss"]), "1", "ok")
    except (TypeError, ValueError):
        return {"value": None, "unit": "1", "status": "unavailable"}


def envelope(*, record_kind: str, identity: EnvelopeIdentity, monotonic_ns: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_kind": record_kind,
        "agoge_run_id": identity.run_id,
        "host": {"hostname": identity.hostname},
        "gpu": dict(identity.gpu),
        "timestamp_utc": utc_z(),
        "monotonic_ns": monotonic_ns,
        "collector": {"id": COLLECTOR_ID, "version": identity.collector_version},
        "cadence_ms": identity.cadence_ms,
    }
