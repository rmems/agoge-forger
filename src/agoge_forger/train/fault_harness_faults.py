"""Injected fault kinds, schedule validation, and raise helpers for the harness."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FaultPoint = Literal[
    "before_save",
    "during_write",
    "during_rename",
    "after_save",
    "during_export",
]
FaultKind = Literal[
    "oom",
    "sigint",
    "sigterm",
    "short_write",
    "rename_failure",
    "export_failure",
]
_ALLOWED_FAULTS: frozenset[tuple[FaultPoint, FaultKind]] = frozenset(
    {
        ("before_save", "oom"),
        ("before_save", "sigint"),
        ("before_save", "sigterm"),
        ("during_write", "short_write"),
        ("during_rename", "rename_failure"),
        ("after_save", "oom"),
        ("after_save", "sigint"),
        ("after_save", "sigterm"),
        ("during_export", "export_failure"),
        ("during_export", "oom"),
        ("during_export", "sigint"),
        ("during_export", "sigterm"),
    }
)


class InjectedOOMError(RuntimeError):
    """CPU stand-in for a CUDA out-of-memory abort."""


class InjectedInterrupt(Exception):
    """SIGINT-equivalent interruption."""


class InjectedTermination(Exception):
    """SIGTERM-equivalent interruption."""


class InjectedExportError(RuntimeError):
    """Final adapter export aborted before run-root publication."""


def _raise_oom() -> None:
    raise InjectedOOMError("CUDA out of memory")


def _raise_sigint() -> None:
    raise InjectedInterrupt("SIGINT")


def _raise_sigterm() -> None:
    raise InjectedTermination("SIGTERM")


def _raise_export_failure() -> None:
    raise InjectedExportError("adapter export failed")


def _raise_rename_failure() -> None:
    raise OSError("injected rename failure")


def _ignore_short_write() -> None:
    return


_FAULT_RAISERS: dict[FaultKind, Callable[[], None]] = {
    "oom": _raise_oom,
    "sigint": _raise_sigint,
    "sigterm": _raise_sigterm,
    "export_failure": _raise_export_failure,
    "rename_failure": _raise_rename_failure,
    "short_write": _ignore_short_write,
}


@dataclass(frozen=True)
class FaultSpec:
    step: int
    point: FaultPoint
    kind: FaultKind

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError(f"fault step must be >= 1: {self.step}")
        if (self.point, self.kind) not in _ALLOWED_FAULTS:
            raise ValueError(f"unsupported fault combination: {self.point}/{self.kind}")


def validate_fault_schedule(fault: FaultSpec, max_steps: int, save_steps: int) -> None:
    if fault.point == "during_export":
        _require_export_fault_step(fault.step, max_steps)
        return
    _require_save_step_fault(fault.step, max_steps, save_steps)


def _require_export_fault_step(step: int, max_steps: int) -> None:
    if step != max_steps:
        raise ValueError("during_export fault step must equal max_steps")


def _require_save_step_fault(step: int, max_steps: int, save_steps: int) -> None:
    if step > max_steps:
        raise ValueError(_save_step_fault_message(step, max_steps))
    if step % save_steps != 0:
        raise ValueError(_save_step_fault_message(step, max_steps))


def _save_step_fault_message(step: int, max_steps: int) -> str:
    return f"checkpoint fault step {step} is not a save step in 1..{max_steps}"


def is_fault(fault: FaultSpec | None, step: int, point: FaultPoint) -> bool:
    return fault is not None and fault.step == step and fault.point == point


def raise_if_fault(fault: FaultSpec | None, step: int, point: FaultPoint) -> None:
    if fault is None or not is_fault(fault, step, point):
        return
    raiser = _FAULT_RAISERS.get(fault.kind)
    if raiser is None:
        raise ValueError(f"unsupported fault kind: {fault.kind}")
    raiser()


def rename_for_fault(fault: FaultSpec | None, step: int) -> Callable[[Path, Path], None] | None:
    if not is_fault(fault, step, "during_rename"):
        return None

    def boom(_source: Path, _destination: Path) -> None:
        raise OSError("injected rename failure")

    return boom
