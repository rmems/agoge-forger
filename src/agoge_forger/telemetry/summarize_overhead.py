"""Compare profiled vs unprofiled optimizer-step wall times on one run."""

from __future__ import annotations

from typing import Any


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
