"""Extract torch.profiler events without guessing CUDA timings."""

from __future__ import annotations

from typing import Any

from .summarize_names import classify_name


def event_rows(profiler: Any) -> list[dict[str, Any]]:
    raw = profiler_table(profiler, "events")
    if not raw:
        return []
    return [row_from_event(item) for item in raw]


def average_rows(profiler: Any) -> list[dict[str, Any]]:
    table = profiler_table(profiler, "key_averages")
    if not table:
        return []
    return [row_from_average(item) for item in table]


def has_per_event(profiler: Any) -> bool:
    return bool(profiler_table(profiler, "events"))


def profiler_table(profiler: Any, name: str) -> Any:
    attr = getattr(profiler, name, None)
    if attr is None:
        return None
    if callable(attr):
        return _call_table(attr)
    return attr


def row_from_event(item: Any) -> dict[str, Any]:
    name = str(getattr(item, "name", "unknown"))
    device = _device_kind(item)
    return {
        "name": name,
        "family": classify_name(name),
        "device": device,
        "duration_us": _event_duration_us(item, device),
        "count": 1,
    }


def row_from_average(item: Any) -> dict[str, Any]:
    name = str(getattr(item, "key", getattr(item, "name", "unknown")))
    count = int(getattr(item, "count", 1) or 1)
    cuda_us = _positive_float(item, "device_time_total") or _positive_float(item, "cuda_time_total")
    cpu_us = float(getattr(item, "cpu_time_total", 0.0) or 0.0)
    device = "cuda" if cuda_us is not None else "cpu"
    duration = cuda_us if cuda_us is not None else cpu_us
    return {
        "name": name,
        "family": classify_name(name),
        "device": device,
        "duration_us": duration / max(count, 1),
        "count": count,
    }


def _call_table(attr: Any) -> Any:
    try:
        return attr()
    except (TypeError, AttributeError, RuntimeError):
        return None


def _device_kind(item: Any) -> str:
    device_type = str(
        getattr(getattr(item, "device_type", None), "name", getattr(item, "device_type", "cpu"))
    )
    lowered = device_type.lower()
    if "cuda" in lowered or "gpu" in lowered:
        return "cuda"
    return "cpu"


def _event_duration_us(item: Any, device: str) -> float:
    if device == "cuda":
        device_time = _first_attr(item, ("device_time_total", "device_time", "cuda_time_total"))
        if device_time is not None:
            return float(device_time or 0.0)
    return float(getattr(item, "duration", getattr(item, "cpu_time_total", 0.0)) or 0.0)


def _first_attr(item: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _positive_float(item: Any, name: str) -> float | None:
    value = float(getattr(item, name, 0.0) or 0.0)
    if value > 0:
        return value
    return None
