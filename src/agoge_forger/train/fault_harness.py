"""Bounded CPU fake-trainer for single-GPU checkpoint fault injection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .._atomic_file import publish_bytes_replace
from .._run_status_torch_archive import torch_mapping
from .checkpoint_publish import (
    IncompleteCheckpointError,
    PublishOptions,
    publish_checkpoint_tree,
    rollback_run_dir,
)
from .checkpoints import (
    PathLike,
    checkpoint_step,
    list_quarantined_checkpoints,
    list_valid_checkpoints,
    quarantine_root,
)
from .fault_harness_export import export_final_adapter
from .fault_harness_faults import (
    FaultKind,
    FaultPoint,
    FaultSpec,
    InjectedExportError,
    InjectedInterrupt,
    InjectedOOMError,
    InjectedTermination,
    is_fault,
    raise_if_fault,
    rename_for_fault,
    validate_fault_schedule,
)
from .fault_harness_payloads import (
    CheckpointSnapshot,
    advance_optimizer,
    advance_rng,
    advance_scheduler,
    optimizer_state,
    scheduler_state,
    seed_rng,
    write_checkpoint_tree,
)
from .resume_state import (
    ResumeInspection,
    inspect_resume_state,
    select_resume_checkpoint,
)

__all__ = [
    "CheckpointSnapshot",
    "FaultKind",
    "FaultPoint",
    "FaultSpec",
    "HarnessConfig",
    "HarnessResult",
    "InjectedExportError",
    "InjectedInterrupt",
    "InjectedOOMError",
    "InjectedTermination",
    "last_failure_evidence",
    "run_fault_harness",
]

_HARNESS_EXCEPTIONS = (
    InjectedOOMError,
    InjectedInterrupt,
    InjectedTermination,
    InjectedExportError,
    IncompleteCheckpointError,
    OSError,
    ValueError,
)


@dataclass(frozen=True)
class HarnessConfig:
    run_dir: Path
    max_steps: int = 4
    save_steps: int = 2
    batch_size: int = 1
    seed: int = 7
    fault: FaultSpec | None = None
    resume: bool = True

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.save_steps < 1:
            raise ValueError("save_steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.fault is None:
            return
        validate_fault_schedule(self.fault, self.max_steps, self.save_steps)


@dataclass(frozen=True)
class HarnessResult:
    completed: bool
    published_steps: tuple[int, ...]
    export_published: bool
    error_type: str | None
    resume: ResumeInspection
    quarantined: tuple[Path, ...]


@dataclass
class _LoopState:
    start_step: int
    sampler_position: int
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]


def run_fault_harness(config: HarnessConfig) -> HarnessResult:
    """Run a bounded fake training loop with optional injected faults."""
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_rng(config.seed)
    error_type: str | None = None
    export_published = False
    completed = False
    try:
        start_step, sampler_position, optimizer, scheduler = _restore_or_initialize(config, run_dir)
        _run_steps(
            config,
            run_dir,
            _LoopState(start_step, sampler_position, optimizer, scheduler),
        )
        export_final_adapter(run_dir, max_steps=config.max_steps, fault=config.fault)
        export_published = True
        completed = True
    except _HARNESS_EXCEPTIONS as exc:
        error_type = type(exc).__name__
        rollback_run_dir(run_dir)
        _record_failure_evidence(run_dir, error_type=error_type)
    published = tuple(checkpoint_step(path) for path in list_valid_checkpoints(run_dir))
    return HarnessResult(
        completed=completed,
        published_steps=published,
        export_published=export_published,
        error_type=error_type,
        resume=inspect_resume_state(select_resume_checkpoint(run_dir).checkpoint),
        quarantined=tuple(list_quarantined_checkpoints(run_dir)),
    )


def last_failure_evidence(run_dir: PathLike) -> dict[str, Any] | None:
    path = quarantine_root(run_dir) / "last_failure.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _restore_or_initialize(
    config: HarnessConfig, run_dir: Path
) -> tuple[int, int, dict[str, Any], dict[str, Any]]:
    if not config.resume:
        return _fresh_loop_state()
    inspection = select_resume_checkpoint(run_dir, restore=True)
    if inspection.checkpoint is None:
        return _fresh_loop_state()
    return _loop_state_from_inspection(inspection)


def _fresh_loop_state() -> tuple[int, int, dict[str, Any], dict[str, Any]]:
    return 0, 0, optimizer_state(step=1), scheduler_state(step=0)


def _loop_state_from_inspection(
    inspection: ResumeInspection,
) -> tuple[int, int, dict[str, Any], dict[str, Any]]:
    _require_equivalent_resume(inspection)
    optimizer, scheduler = _require_restored_mappings(inspection)
    return inspection.global_step or 0, inspection.sampler_position or 0, optimizer, scheduler


def _require_equivalent_resume(inspection: ResumeInspection) -> None:
    if inspection.equivalent:
        return
    reason = inspection.reason or "required trainer state is missing"
    raise ValueError(f"cannot resume from incomplete checkpoint: {reason}")


def _require_restored_mappings(
    inspection: ResumeInspection,
) -> tuple[dict[str, Any], dict[str, Any]]:
    optimizer = _load_mapping(inspection, "optimizer.pt", require_data_record=True)
    scheduler = _load_mapping(inspection, "scheduler.pt", require_data_record=False)
    if optimizer is None or scheduler is None:
        raise ValueError("cannot resume: restored optimizer or scheduler payload is missing")
    return optimizer, scheduler


def _run_steps(config: HarnessConfig, run_dir: Path, loop: _LoopState) -> None:
    position = loop.sampler_position
    optimizer = loop.optimizer
    scheduler = loop.scheduler
    for step in range(loop.start_step + 1, config.max_steps + 1):
        position += config.batch_size
        advance_rng()
        optimizer = advance_optimizer(optimizer, step)
        scheduler = advance_scheduler(scheduler, step)
        if step % config.save_steps != 0:
            continue
        snapshot = CheckpointSnapshot(
            step=step,
            sampler_position=position,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=config.batch_size,
            short_write=is_fault(config.fault, step, "during_write"),
        )
        _save_step(config, run_dir, snapshot)


def _save_step(config: HarnessConfig, run_dir: Path, snapshot: CheckpointSnapshot) -> None:
    raise_if_fault(config.fault, snapshot.step, "before_save")
    publish_checkpoint_tree(
        run_dir,
        snapshot.step,
        partial(write_checkpoint_tree, snapshot=snapshot),
        PublishOptions(rename=rename_for_fault(config.fault, snapshot.step)),
    )
    raise_if_fault(config.fault, snapshot.step, "after_save")


def _load_mapping(
    inspection: ResumeInspection,
    filename: str,
    *,
    require_data_record: bool,
) -> dict[str, Any] | None:
    if inspection.checkpoint is None:
        return None
    payload = torch_mapping(
        inspection.checkpoint / filename,
        require_data_record=require_data_record,
    )
    return payload if isinstance(payload, dict) else None


def _record_failure_evidence(run_dir: Path, *, error_type: str) -> None:
    inspection = inspect_resume_state(select_resume_checkpoint(run_dir).checkpoint)
    payload = {
        "schema_version": 1,
        "error_type": error_type,
        "last_complete_checkpoint": None
        if inspection.checkpoint is None
        else str(inspection.checkpoint),
        "last_complete_step": inspection.global_step,
        "resume_equivalent": inspection.equivalent,
    }
    publish_bytes_replace(
        quarantine_root(run_dir) / "last_failure.json",
        (json.dumps(payload, indent=2) + "\n").encode(),
    )
