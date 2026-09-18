"""Aggregate profiler events into compact family blocks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .summarize_events import profiler_table
from .summarize_names import transfer_direction


def group_cuda_kernels(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    families: dict[str, str] = {}
    for event in events:
        if event["device"] != "cuda":
            continue
        buckets[str(event["name"])].extend([float(event["duration_us"])] * int(event["count"]))
        families[str(event["name"])] = str(event["family"])
    ranked = sorted(buckets.items(), key=lambda item: -sum(item[1]))
    return [
        {
            "name": name,
            "family": families[name],
            "count": len(durations),
            "duration_us": percentiles(durations),
        }
        for name, durations in ranked[:32]
    ]


def duration_block(events: list[dict[str, Any]], device: str, status: str) -> dict[str, Any]:
    durations = [float(event["duration_us"]) for event in events if event["device"] == device]
    if status != "ok":
        return {"status": status, "value": None, "unit": "us", "count": 0}
    if not durations:
        return {"status": "unavailable", "value": None, "unit": "us", "count": 0}
    stats = percentiles(durations)
    stats["status"] = "ok"
    stats["count"] = len(durations)
    return stats


def family_block(events: list[dict[str, Any]], family: str, status: str) -> dict[str, Any]:
    durations = [float(event["duration_us"]) for event in events if event["family"] == family]
    if not durations:
        return {
            "status": "unavailable" if status == "ok" else status,
            "count": 0,
            "duration_us": None,
        }
    return {"status": "ok", "count": len(durations), "duration_us": percentiles(durations)}


def transfer_block(events: list[dict[str, Any]], cuda_status: str) -> dict[str, Any]:
    h2d: list[float] = []
    d2h: list[float] = []
    for event in events:
        if event["family"] != "transfer":
            continue
        bucket = d2h if transfer_direction(str(event["name"])) == "d2h" else h2d
        bucket.append(float(event["duration_us"]))
    if not h2d and not d2h:
        return {
            "status": "unavailable" if cuda_status == "ok" else cuda_status,
            "h2d_us": None,
            "d2h_us": None,
        }
    return {
        "status": "ok",
        "h2d_us": percentiles(h2d) if h2d else None,
        "d2h_us": percentiles(d2h) if d2h else None,
    }


def observable_path(events: list[dict[str, Any]], family: str) -> dict[str, Any]:
    names = sorted({str(event["name"]) for event in events if event["family"] == family})
    if not names:
        return {"status": "unavailable", "path": None, "names": []}
    return {"status": "ok", "path": names[0], "names": names[:8]}


def top_cpu(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event["device"] != "cpu":
            continue
        buckets[str(event["name"])].append(float(event["duration_us"]))
    ranked = sorted(buckets.items(), key=lambda item: -sum(item[1]))
    return [
        {"name": name, "count": len(durations), "duration_us": percentiles(durations)}
        for name, durations in ranked[:16]
    ]


def memory_block(events: list[dict[str, Any]], profiler: Any) -> dict[str, Any]:
    counters = memory_counters(profiler)
    if counters["status"] == "ok":
        return counters
    return family_block(events, "allocator", "ok" if events else "unavailable")


def memory_counters(profiler: Any) -> dict[str, Any]:
    raw = _profiler_memory_table(profiler)
    cpu_hits = _memory_hits(raw, ("cpu_memory_usage", "self_cpu_memory_usage"))
    device_hits = _memory_hits(
        raw, ("device_memory_usage", "cuda_memory_usage", "self_device_memory_usage")
    )
    if not cpu_hits and not device_hits:
        return {"status": "unavailable", "count": 0, "duration_us": None}
    return {
        "status": "ok",
        "count": len(cpu_hits) + len(device_hits),
        "cpu_bytes": _stats(cpu_hits, "B") if cpu_hits else None,
        "device_bytes": _stats(device_hits, "B") if device_hits else None,
        "duration_us": None,
    }


def _profiler_memory_table(profiler: Any) -> Any:
    return profiler_table(profiler, "events") or profiler_table(profiler, "key_averages") or ()


def _memory_hits(raw: Any, names: tuple[str, ...]) -> list[float]:
    return [value for value in (_memory_value(item, names) for item in raw) if value is not None]


def percentiles(values: list[float]) -> dict[str, Any]:
    return _stats(values, "us")


def _stats(values: list[float], unit: str) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "sum": float(sum(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
        "unit": unit,
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _memory_value(item: Any, names: tuple[str, ...]) -> float | None:
    for name in names:
        if not hasattr(item, name):
            continue
        value = getattr(item, name)
        if value is None:
            continue
        number = float(value)
        if abs(number) > 0:
            return number
    return None
