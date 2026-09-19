"""Classify profiler event names into families. Names are evidence, not CUDA truth."""

from __future__ import annotations

_FAMILIES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("memcpy", "hto d", "dtoh", "htod", "d2h", "h2d", "copy_kernel", "cudamemcpy"), "transfer"),
    (("sync", "synchronize", "wait"), "sync"),
    (("flash", "sdpa", "fmha", "attention", "mem_eff"), "attention"),
    (("gemm", "cutlass", "nvjet", "wmma", "mma", "addmm", "bmm", "mm_"), "gemm"),
    (("alloc", "malloc", "caching", "reserved", "cudamalloc"), "allocator"),
)


def classify_name(name: str) -> str:
    lowered = name.lower()
    for needles, family in _FAMILIES:
        if family == "gemm" and lowered in {"mm", "matmul"}:
            return family
        if _contains(lowered, needles):
            return family
    return "other"


_D2H = ("dtoh", "d2h", "device_to_host")
_H2D = ("htod", "h2d", "host_to_device", "hto d")


def transfer_direction(name: str) -> str | None:
    lowered = name.lower()
    if _is_device_to_host(lowered):
        return "d2h"
    if any(token in lowered for token in _H2D):
        return "h2d"
    return None


def _is_device_to_host(lowered: str) -> bool:
    return any(token in lowered for token in _D2H)


def _contains(name: str, needles: tuple[str, ...]) -> bool:
    return any(needle in name for needle in needles)
