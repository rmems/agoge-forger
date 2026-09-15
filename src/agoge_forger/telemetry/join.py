"""Join Agoge markers, profile-window summaries, and BKL GPU samples on CPU."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


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


def join_profile_window(
    summary: Mapping[str, Any],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    window_id = summary.get("profile_window_id")
    run_id = summary.get("agoge_run_id")
    start_ns = summary.get("monotonic_ns_start")
    end_ns = summary.get("monotonic_ns_end")
    matched: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("agoge_run_id") != run_id:
            continue
        sample_ns = sample.get("monotonic_ns")
        if not isinstance(sample_ns, int):
            continue
        if isinstance(start_ns, int) and sample_ns < start_ns:
            continue
        if isinstance(end_ns, int) and sample_ns > end_ns:
            continue
        ref = sample.get("profile_window_ref")
        if ref is not None and ref != window_id:
            continue
        matched.append(sample)
    return matched


def _bucket(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record["agoge_run_id"]), str(record["host"]["hostname"]))


def _marker_sort_key(marker: dict[str, Any]) -> tuple[int, datetime]:
    stamp = datetime.fromisoformat(str(marker["timestamp_utc"]).replace("Z", "+00:00"))
    return (int(marker["monotonic_ns"]), stamp)


def _latest_marker(markers: list[dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any] | None:
    sample_ns = int(sample["monotonic_ns"])
    chosen: dict[str, Any] | None = None
    for marker in markers:
        if int(marker["monotonic_ns"]) > sample_ns:
            break
        if _gpu_match(marker.get("gpu"), sample.get("gpu")):
            chosen = marker
    return chosen


def _gpu_match(marker_gpu: Any, sample_gpu: Any) -> bool:
    if not isinstance(marker_gpu, dict) or not isinstance(sample_gpu, dict):
        return False
    marker_uuid = _id(marker_gpu.get("uuid"))
    sample_uuid = _id(sample_gpu.get("uuid"))
    if marker_uuid is not None and sample_uuid is not None:
        return marker_uuid == sample_uuid
    marker_pci = _id(marker_gpu.get("pci_bus_id"))
    sample_pci = _id(sample_gpu.get("pci_bus_id"))
    if marker_pci is not None and sample_pci is not None:
        return marker_pci == sample_pci
    return False


def _id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
