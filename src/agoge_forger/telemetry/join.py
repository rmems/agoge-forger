"""Join Agoge markers, profile-window summaries, and BKL GPU samples on CPU."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .join_identity import gpu_match
from .join_window import join_profile_window

__all__ = ["join_profile_window", "join_samples", "load_jsonl"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def join_samples(
    markers: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Latest marker with monotonic_ns <= sample, same run/host/GPU identity."""

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for marker in markers:
        buckets.setdefault(_bucket(marker), []).append(marker)
    for bucket in buckets.values():
        bucket.sort(key=_marker_sort_key)
    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sample in samples:
        chosen = _latest_marker(buckets.get(_bucket(sample), []), sample)
        if chosen is not None:
            joined.append((chosen, sample))
    return joined


def _bucket(record: dict[str, Any]) -> tuple[str, str]:
    host = record["host"]["hostname"]
    return str(record["agoge_run_id"]), str(host)


def _marker_sort_key(marker: dict[str, Any]) -> tuple[int, datetime]:
    stamp = datetime.fromisoformat(str(marker["timestamp_utc"]).replace("Z", "+00:00"))
    return int(marker["monotonic_ns"]), stamp


def _latest_marker(markers: list[dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any] | None:
    sample_ns = int(sample["monotonic_ns"])
    chosen: dict[str, Any] | None = None
    for marker in markers:
        if int(marker["monotonic_ns"]) > sample_ns:
            break
        if gpu_match(marker.get("gpu"), sample.get("gpu")):
            chosen = marker
    return chosen
