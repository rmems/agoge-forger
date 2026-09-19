"""Aggregate profiler events into compact family blocks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .summarize_events import profiler_table
from .summarize_names import transfer_direction

WeightedDurations = list[tuple[float, int]]


def group_cuda_kernels(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, WeightedDurations] = defaultdict(list)
    families: dict[str, str] = {}
    for event in events:
        if event["device"] != "cuda":
            continue
        buckets[str(event["name"])].append(_weighted_observation(event))
        families[str(event["name"])] = str(event["family"])
    ranked = sorted(buckets.items(), key=lambda item: -_weighted_sum(item[1]))
    return [
        {
            "name": name,
            "family": families[name],
            "count": _weighted_count(durations),
            "duration_us": _weighted_stats(durations, "us"),
        }
        for name, durations in ranked[:32]
    ]


def duration_block(events: list[dict[str, Any]], device: str, status: str) -> dict[str, Any]:
    durations = _weighted_observations(event for event in events if event["device"] == device)
    if status != "ok":
        return {"status": status, "value": None, "unit": "us", "count": 0}
    if not durations:
        return {"status": "unavailable", "value": None, "unit": "us", "count": 0}
    stats = _weighted_stats(durations, "us")
    stats["status"] = "ok"
    return stats


def family_block(events: list[dict[str, Any]], family: str, status: str) -> dict[str, Any]:
    durations = _weighted_observations(event for event in events if event["family"] == family)
    if not durations:
        return {
            "status": "unavailable" if status == "ok" else status,
            "count": 0,
            "duration_us": None,
        }
    return {
        "status": "ok",
        "count": _weighted_count(durations),
        "duration_us": _weighted_stats(durations, "us"),
    }


def transfer_block(events: list[dict[str, Any]], cuda_status: str) -> dict[str, Any]:
    return _transfer_result(_transfer_durations(events), cuda_status)


def _transfer_result(
    directions: Mapping[str, WeightedDurations], cuda_status: str
) -> dict[str, Any]:
    if not any(directions.values()):
        return _unavailable_transfer(cuda_status)
    return {
        "status": "ok",
        "h2d_us": _optional_weighted_stats(directions["h2d"]),
        "d2h_us": _optional_weighted_stats(directions["d2h"]),
    }


def _unavailable_transfer(cuda_status: str) -> dict[str, Any]:
    status = "unavailable" if cuda_status == "ok" else cuda_status
    return {"status": status, "h2d_us": None, "d2h_us": None}


def _optional_weighted_stats(durations: WeightedDurations) -> dict[str, Any] | None:
    return _weighted_stats(durations, "us") if durations else None


def observable_path(events: list[dict[str, Any]], family: str) -> dict[str, Any]:
    names = sorted({str(event["name"]) for event in events if event["family"] == family})
    if not names:
        return {"status": "unavailable", "path": None, "names": []}
    return {"status": "ok", "path": names[0], "names": names[:8]}


def top_cpu(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, WeightedDurations] = defaultdict(list)
    for event in events:
        if event["device"] != "cpu":
            continue
        buckets[str(event["name"])].append(_weighted_observation(event))
    ranked = sorted(buckets.items(), key=lambda item: -_weighted_sum(item[1]))
    return [
        {
            "name": name,
            "count": _weighted_count(durations),
            "duration_us": _weighted_stats(durations, "us"),
        }
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
    return _weighted_stats([(value, 1) for value in values], "us")


def _stats(values: list[float], unit: str) -> dict[str, Any]:
    return _weighted_stats([(value, 1) for value in values], unit)


def _weighted_observations(events: Iterable[Mapping[str, Any]]) -> WeightedDurations:
    return [_weighted_observation(event) for event in events]


def _weighted_observation(event: Mapping[str, Any]) -> tuple[float, int]:
    return float(event["duration_us"]), max(1, int(event.get("count", 1)))


def _transfer_durations(events: Iterable[Mapping[str, Any]]) -> dict[str, WeightedDurations]:
    directions: dict[str, WeightedDurations] = {"h2d": [], "d2h": []}
    for event in events:
        if event["family"] != "transfer" or event["device"] != "cuda":
            continue
        direction = transfer_direction(str(event["name"]))
        if direction is not None:
            directions[direction].append(_weighted_observation(event))
    return directions


def _weighted_stats(values: WeightedDurations, unit: str) -> dict[str, Any]:
    count = _weighted_count(values)
    total = _weighted_sum(values)
    return {
        "sum": total,
        "mean": total / count,
        "p50": _weighted_quantile(values, 0.50),
        "p95": _weighted_quantile(values, 0.95),
        "unit": unit,
        "count": count,
    }


def _weighted_count(values: WeightedDurations) -> int:
    return sum(count for _, count in values)


def _weighted_sum(values: WeightedDurations) -> float:
    return float(sum(value * count for value, count in values))


def _weighted_quantile(values: WeightedDurations, q: float) -> float:
    if not values:
        return 0.0
    target = max(1, math.ceil(q * _weighted_count(values)))
    cumulative = 0
    for value, count in sorted(values):
        cumulative += count
        if cumulative >= target:
            return float(value)
    return float(max(value for value, _ in values))


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
