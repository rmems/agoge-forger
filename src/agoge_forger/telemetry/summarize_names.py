"""Classify profiler event names into families. Names are evidence, not CUDA truth."""

from __future__ import annotations

_TRANSFER = ("memcpy", "hto d", "dtoh", "htod", "d2h", "h2d", "copy_kernel", "cudamemcpy")
_SYNC = ("sync", "synchronize", "wait")
_ATTENTION = ("flash", "sdpa", "fmha", "attention", "mem_eff")
_GEMM = ("gemm", "cutlass", "nvjet", "wmma", "mma", "addmm", "bmm", "mm_")
_ALLOC = ("alloc", "malloc", "caching", "reserved", "cudamalloc")


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


_D2H = ("dtoh", "d2h", "device_to_host")


def transfer_direction(name: str) -> str:
    lowered = name.lower()
    if _is_device_to_host(lowered):
        return "d2h"
    return "h2d"


def _is_device_to_host(lowered: str) -> bool:
    return any(token in lowered for token in _D2H)


def _contains(name: str, needles: tuple[str, ...]) -> bool:
    return any(needle in name for needle in needles)
