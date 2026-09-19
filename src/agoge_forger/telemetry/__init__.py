"""Agoge-owned training markers and opt-in bounded CUDA profile windows.

GPU/CUDA collectors live in ``blackwell-kernel-lab``. This package emits the
join keys and, when requested, a compact torch.profiler summary. Ordinary
training does not start a profiler.
"""

from .session import TelemetrySession, attach_telemetry, open_training_session
from .window import parse_profile_window, profile_window_id

__all__ = [
    "TelemetrySession",
    "attach_telemetry",
    "open_training_session",
    "parse_profile_window",
    "profile_window_id",
]
