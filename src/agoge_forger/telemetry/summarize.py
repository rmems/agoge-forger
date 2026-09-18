"""Compact profile-window summary from torch.profiler. Raw traces stay out of git."""

from __future__ import annotations

from typing import Any

from .summarize_events import average_rows, event_rows, has_per_event
from .summarize_families import (
    duration_block,
    family_block,
    group_cuda_kernels,
    memory_block,
    observable_path,
    top_cpu,
    transfer_block,
)
from .summarize_names import classify_name
from .summarize_overhead import overhead_from_step_times

__all__ = ["classify_name", "overhead_from_step_times", "summarize_profiler"]


def summarize_profiler(
    profiler: Any,
    *,
    cuda_kernel_status: str,
) -> dict[str, Any]:
    events = event_rows(profiler) or average_rows(profiler)
    kernels = group_cuda_kernels(events)
    return {
        "kernels": kernels,
        "kernel_duration": duration_block(events, "cuda", cuda_kernel_status),
        "sync": family_block(events, "sync", cuda_kernel_status),
        "transfers": transfer_block(events, cuda_kernel_status),
        "cuda_graphs": {
            "status": "unavailable",
            "reason": "torch.profiler does not expose capture/replay as a first-class flag",
        },
        "attention": observable_path(events, "attention"),
        "gemm": observable_path(events, "gemm"),
        "memory": memory_block(events, profiler),
        "cpu_ops": top_cpu(events),
        "distribution": "per-event" if has_per_event(profiler) else "key_averages_mean_only",
    }
