"""Join BKL GPU samples onto one profile-window summary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .join_identity import gpu_match, same_host


def join_profile_window(
    summary: Mapping[str, Any],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [sample for sample in samples if sample_in_window(summary, sample)]


def sample_in_window(summary: Mapping[str, Any], sample: dict[str, Any]) -> bool:
    if sample.get("agoge_run_id") != summary.get("agoge_run_id"):
        return False
    if not same_host(dict(summary), sample):
        return False
    if not gpu_match(summary.get("gpu"), sample.get("gpu")):
        return False
    if not _inside_interval(summary, sample):
        return False
    return _window_ref_ok(summary, sample)


def _inside_interval(summary: Mapping[str, Any], sample: dict[str, Any]) -> bool:
    sample_ns = sample.get("monotonic_ns")
    if not isinstance(sample_ns, int):
        return False
    start_ns = summary.get("monotonic_ns_start")
    end_ns = summary.get("monotonic_ns_end")
    if not isinstance(start_ns, int) or not isinstance(end_ns, int):
        return False
    if sample_ns < start_ns:
        return False
    return sample_ns <= end_ns


def _window_ref_ok(summary: Mapping[str, Any], sample: dict[str, Any]) -> bool:
    ref = sample.get("profile_window_ref")
    return ref is None or ref == summary.get("profile_window_id")
