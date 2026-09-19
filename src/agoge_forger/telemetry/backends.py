"""Probe profiler backends. Unavailable paths are recorded; they are never guessed."""

from __future__ import annotations

import shutil
from typing import Any

import torch
from torch.profiler import ProfilerActivity

from .schema import KNOWN_BACKENDS, TORCH_BACKEND

_BKL_OWNED = "owned by blackwell-kernel-lab; Agoge does not invoke this profiler"


def probe_profiler_backends() -> dict[str, dict[str, Any]]:
    return {
        TORCH_BACKEND: _probe_torch(),
        "nsight_systems": _external("nsight_systems", "nsys"),
        "nsight_compute": _external("nsight_compute", "ncu"),
        "cupti": {
            "status": "unsupported",
            "reason": _BKL_OWNED,
            "tool": None,
        },
    }


def requested_backend_status(backend: str, probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if backend not in KNOWN_BACKENDS:
        return {
            "status": "unsupported",
            "reason": f"unknown profiler backend {backend!r}; not guessed",
        }
    return probes[backend]


def torch_profiler_activities() -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


def _probe_torch() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda else 0
    if device_count > 1:
        return {
            "status": "unsupported",
            "reason": (
                "torch profiler profile windows require a single CUDA device; "
                f"detected {device_count}"
            ),
            "version": torch.__version__,
            "cuda_available": cuda,
            "cuda_kernels": "unavailable",
            "activities": ["cpu", "cuda"],
        }
    return {
        "status": "ok",
        "version": torch.__version__,
        "cuda_available": cuda,
        "cuda_kernels": "ok" if cuda else "unavailable",
        "activities": ["cpu"] + (["cuda"] if cuda else []),
    }


def _external(_backend: str, binary: str) -> dict[str, Any]:
    path = shutil.which(binary)
    if path is None:
        return {
            "status": "unsupported",
            "reason": f"{_BKL_OWNED}; {binary} not on PATH",
            "tool": None,
        }
    return {
        "status": "unsupported",
        "reason": f"{_BKL_OWNED}; {binary} is present at {path} but not launched from Agoge",
        "tool": path,
    }
