"""Standalone torch.profiler overhead microbenchmark (CPU CI and optional CUDA)."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.profiler import profile

from .backends import torch_profiler_activities
from .schema import measurement


def measure_torch_profiler_overhead(*, repeats: int = 12, size: int = 128) -> dict[str, Any]:
    """Compare a tiny matmul loop with and without torch.profiler.

    This is not a training-quality claim. It states profiler overhead on the
    current host so the recipe does not silently treat tracing as free.
    """

    baseline = _loop_seconds(repeats, size, profiler=None)
    with profile(
        activities=torch_profiler_activities(),
        record_shapes=False,
        acc_events=True,
    ) as prof:
        profiled = _loop_seconds(repeats, size, profiler=prof)
    ratio = None if baseline <= 0 else profiled / baseline
    cuda = torch.cuda.is_available()
    return {
        "workload": "cpu_matmul" if not cuda else "device_matmul",
        "repeats": repeats,
        "matrix": size,
        "unprofiled_s": measurement(baseline, "s", "ok"),
        "profiled_s": measurement(profiled, "s", "ok"),
        "ratio": measurement(ratio, "1", "ok" if ratio is not None else "unavailable"),
        "cuda_available": cuda,
        "note": "torch.profiler is opt-in; ordinary Agoge runs do not pay this cost",
    }


def _loop_seconds(repeats: int, size: int, profiler: Any) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    for _ in range(repeats):
        tensor = torch.randn(size, size, device=device)
        (tensor @ tensor).sum().item()
        if profiler is not None:
            profiler.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started
