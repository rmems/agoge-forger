"""Compact profile-window summary from torch.profiler. Raw traces stay out of git."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_TRANSFER = ("memcpy", "hto d", "dtoh", "htod", "d2h", "h2d", "copy_kernel", "cudamemcpy")
_SYNC = ("sync", "synchronize", "wait")
_ATTENTION = ("flash", "sdpa", "fmha", "attention", "mem_eff")
_GEMM = ("gemm", "cutlass", "nvjet", "wmma", "mma", "addmm", "bmm", "mm_")
_ALLOC = ("alloc", "malloc", "caching", "reserved", "cudaMalloc")


def summarize_profiler(
    profiler: Any,
    *,
    cuda_kernel_status: str,
) -> dict[str, Any]:
    events = _event_rows(profiler)
    if not events:
        events = _average_rows(profiler)
    kernels = _group_cuda_kernels(events)
    return {
        "kernels": kernels,
        "kernel_duration": _duration_block(events, device="cuda", status=cuda_kernel_status),
        "sync": _family_block(events, "sync", cuda_kernel_status),
        "transfers": _transfer_block(events, cuda_kernel_status),
        "cuda_graphs": {
            "status": "unavailable",
            "reason": "torch.profiler does not expose capture/replay as a first-class flag",
        },
        "attention": _observable_path(events, "attention"),
        "gemm": _observable_path(events, "gemm"),
        "memory": _family_block(events, "allocator", "ok" if events else "unavailable"),
        "cpu_ops": _top_cpu(events),
        "distribution": "per-event" if _has_per_event(profiler) else "key_averages_mean_only",
    }


def classify_name(name: str) -> str:
    lowered = name.lower()
    if _contains(lowered, _TRANSFER):
        return "transfer"
    if _contains(lowered, _SYNC):
        return "sync"
    if _contains(lowered, _ATTENTION):
        return "attention"
    if _contains(lowered, _GEMM) or lowered in {"mm", "matmul"}:
        return "gemm"
    if _contains(lowered, _ALLOC):
        return "allocator"
    return "other"


def _event_rows(profiler: Any) -> list[dict[str, Any]]:
    getter = getattr(profiler, "events", None)
    if getter is None:
        return []
    try:
        raw = getter()
    except TypeError:
        raw = getter
    except (AttributeError, RuntimeError):
        return []
    if not raw:
        return []
    return [_row_from_event(item) for item in raw]


def _average_rows(profiler: Any) -> list[dict[str, Any]]:
    averages = getattr(profiler, "key_averages", None)
    if averages is None:
        return []
    try:
        table = averages()
    except (TypeError, AttributeError, RuntimeError):
        return []
    rows: list[dict[str, Any]] = []
    for item in table:
        name = str(getattr(item, "key", getattr(item, "name", "unknown")))
        count = int(getattr(item, "count", 1) or 1)
        cuda_us = float(getattr(item, "device_time_total", 0.0) or 0.0)
        if cuda_us == 0.0:
            cuda_us = float(getattr(item, "cuda_time_total", 0.0) or 0.0)
        cpu_us = float(getattr(item, "cpu_time_total", 0.0) or 0.0)
        device = "cuda" if cuda_us > 0 else "cpu"
        duration = cuda_us if device == "cuda" else cpu_us
        rows.append(
            {
                "name": name,
                "family": classify_name(name),
                "device": device,
                "duration_us": duration / max(count, 1),
                "count": count,
            }
        )
    return rows


def _row_from_event(item: Any) -> dict[str, Any]:
    name = str(getattr(item, "name", "unknown"))
    duration = float(getattr(item, "duration", getattr(item, "cpu_time_total", 0.0)) or 0.0)
    device_type = str(
        getattr(getattr(item, "device_type", None), "name", getattr(item, "device_type", "cpu"))
    )
    device = "cuda" if "cuda" in device_type.lower() or "gpu" in device_type.lower() else "cpu"
    return {
        "name": name,
        "family": classify_name(name),
        "device": device,
        "duration_us": duration,
        "count": 1,
    }


def _group_cuda_kernels(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
            "duration_us": _percentiles(durations),
        }
        for name, durations in ranked[:32]
    ]


def _duration_block(events: list[dict[str, Any]], device: str, status: str) -> dict[str, Any]:
    durations = [float(event["duration_us"]) for event in events if event["device"] == device]
    if status != "ok":
        return {"status": status, "value": None, "unit": "us", "count": 0}
    if not durations:
        return {"status": "unavailable", "value": None, "unit": "us", "count": 0}
    stats = _percentiles(durations)
    stats["status"] = "ok"
    stats["count"] = len(durations)
    return stats


def _family_block(events: list[dict[str, Any]], family: str, status: str) -> dict[str, Any]:
    durations = [float(event["duration_us"]) for event in events if event["family"] == family]
    if not durations:
        return {
            "status": "unavailable" if status == "ok" else status,
            "count": 0,
            "duration_us": None,
        }
    return {"status": "ok", "count": len(durations), "duration_us": _percentiles(durations)}


def _transfer_block(events: list[dict[str, Any]], cuda_status: str) -> dict[str, Any]:
    h2d: list[float] = []
    d2h: list[float] = []
    for event in events:
        if event["family"] != "transfer":
            continue
        name = str(event["name"]).lower()
        if "dtoh" in name or "d2h" in name or "device_to_host" in name:
            d2h.append(float(event["duration_us"]))
        else:
            h2d.append(float(event["duration_us"]))
    if not h2d and not d2h:
        return {
            "status": "unavailable" if cuda_status == "ok" else cuda_status,
            "h2d_us": None,
            "d2h_us": None,
        }
    return {
        "status": "ok",
        "h2d_us": _percentiles(h2d) if h2d else None,
        "d2h_us": _percentiles(d2h) if d2h else None,
    }


def _observable_path(events: list[dict[str, Any]], family: str) -> dict[str, Any]:
    names = sorted({str(event["name"]) for event in events if event["family"] == family})
    if not names:
        return {"status": "unavailable", "path": None, "names": []}
    return {"status": "ok", "path": names[0], "names": names[:8]}


def _top_cpu(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event["device"] != "cpu":
            continue
        buckets[str(event["name"])].append(float(event["duration_us"]))
    ranked = sorted(buckets.items(), key=lambda item: -sum(item[1]))
    return [
        {"name": name, "count": len(durations), "duration_us": _percentiles(durations)}
        for name, durations in ranked[:16]
    ]


def _percentiles(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "sum": float(sum(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
        "unit": "us",
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _contains(name: str, needles: tuple[str, ...]) -> bool:
    return any(needle in name for needle in needles)


def _has_per_event(profiler: Any) -> bool:
    getter = getattr(profiler, "events", None)
    if getter is None:
        return False
    try:
        raw = getter()
    except TypeError:
        raw = getter
    except (AttributeError, RuntimeError):
        return False
    return bool(raw)


def overhead_from_step_times(profiled: list[float], unprofiled: list[float]) -> dict[str, Any]:
    if not profiled:
        return {
            "status": "unavailable",
            "reason": "profile window captured no steps",
            "profiled_step_mean_s": None,
            "unprofiled_step_mean_s": None,
            "ratio": None,
        }
    profiled_mean = sum(profiled) / len(profiled)
    if not unprofiled:
        return {
            "status": "unavailable",
            "reason": "no unprofiled optimizer steps in this run to compare",
            "profiled_step_mean_s": profiled_mean,
            "unprofiled_step_mean_s": None,
            "ratio": None,
        }
    baseline = sum(unprofiled) / len(unprofiled)
    ratio = None if baseline <= 0 else profiled_mean / baseline
    return {
        "status": "ok",
        "profiled_step_mean_s": profiled_mean,
        "unprofiled_step_mean_s": baseline,
        "ratio": ratio,
        "unit": "1",
        "note": "ratio is profiled_step_mean / unprofiled_step_mean on this run",
    }
