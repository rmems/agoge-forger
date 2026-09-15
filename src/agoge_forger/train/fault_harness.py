"""Bounded CPU fake-trainer for single-GPU checkpoint fault injection."""

from __future__ import annotations

import io
import json
import os
import random
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .._atomic_directory import rename_noreplace, require_rename_noreplace_support
from .._atomic_file import publish_bytes_replace, write_fsynced_bytes
from .._run_status_torch_archive import torch_mapping
from .checkpoint_publish import (
    IncompleteCheckpointError,
    PublishOptions,
    publish_checkpoint_tree,
    rollback_run_dir,
)
from .checkpoints import (
    ADAPTER_WEIGHT_FILES,
    EXPORT_STAGING_PREFIX,
    PathLike,
    checkpoint_step,
    is_adapter_artifact,
    list_quarantined_checkpoints,
    list_valid_checkpoints,
    quarantine_root,
    quarantine_tree,
)
from .resume_state import (
    RESUME_SIDECAR_FILENAME,
    ResumeInspection,
    inspect_resume_state,
    select_resume_checkpoint,
)

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
_LORA_SHAPES: dict[str, tuple[int, ...]] = {
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (1, 8),
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (8, 1),
}
_ADAPTER_FILENAMES = ("adapter_config.json", ADAPTER_WEIGHT_FILES[0])
_OPTIMIZER_SHAPES: list[list[tuple[int, ...]]] = [[(1, 8), (8, 1)], []]


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


_HARNESS_EXCEPTIONS = (
    InjectedOOMError,
    InjectedInterrupt,
    InjectedTermination,
    InjectedExportError,
    IncompleteCheckpointError,
    OSError,
)


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
        _validate_fault_schedule(self.fault, self.max_steps, self.save_steps)


@dataclass(frozen=True)
class HarnessResult:
    completed: bool
    published_steps: tuple[int, ...]
    export_published: bool
    error_type: str | None
    resume: ResumeInspection
    quarantined: tuple[Path, ...]


@dataclass(frozen=True)
class CheckpointSnapshot:
    step: int
    sampler_position: int
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    batch_size: int
    short_write: bool = False


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
    _seed_rng(config.seed)
    start_step, sampler_position, optimizer, scheduler = _restore_or_initialize(config, run_dir)
    error_type: str | None = None
    export_published = False
    completed = False
    try:
        _run_steps(
            config,
            run_dir,
            _LoopState(start_step, sampler_position, optimizer, scheduler),
        )
        _export_final_adapter(config, run_dir)
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
        return 0, 0, _optimizer_state(step=1), _scheduler_state(step=0)
    inspection = select_resume_checkpoint(run_dir, restore=True)
    start_step = inspection.global_step or 0
    sampler_position = inspection.sampler_position or 0
    return (
        start_step,
        sampler_position,
        _load_mapping(inspection, "optimizer.pt") or _optimizer_state(step=max(start_step, 1)),
        _load_mapping(inspection, "scheduler.pt") or _scheduler_state(step=start_step),
    )


def _run_steps(config: HarnessConfig, run_dir: Path, loop: _LoopState) -> None:
    position = loop.sampler_position
    optimizer = loop.optimizer
    scheduler = loop.scheduler
    for step in range(loop.start_step + 1, config.max_steps + 1):
        position += config.batch_size
        _advance_rng()
        optimizer = _advance_optimizer(optimizer, step)
        scheduler = _advance_scheduler(scheduler, step)
        if step % config.save_steps != 0:
            continue
        snapshot = CheckpointSnapshot(
            step=step,
            sampler_position=position,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=config.batch_size,
            short_write=_is_fault(config.fault, step, "during_write"),
        )
        _save_step(config, run_dir, snapshot)


def _save_step(config: HarnessConfig, run_dir: Path, snapshot: CheckpointSnapshot) -> None:
    _raise_if_fault(config.fault, snapshot.step, "before_save")
    publish_checkpoint_tree(
        run_dir,
        snapshot.step,
        partial(_write_checkpoint_tree, snapshot=snapshot),
        PublishOptions(rename=_rename_for_fault(config.fault, snapshot.step)),
    )
    _raise_if_fault(config.fault, snapshot.step, "after_save")


def _export_final_adapter(config: HarnessConfig, run_dir: Path) -> None:
    if is_adapter_artifact(run_dir):
        return
    require_rename_noreplace_support(run_dir)
    staging_root = Path(tempfile.mkdtemp(prefix=EXPORT_STAGING_PREFIX, dir=run_dir))
    staged = staging_root / "adapter"
    published: list[Path] = []
    try:
        published.extend(_publish_run_root_adapter(config, run_dir, staged))
    except BaseException:
        _quarantine_failed_export(run_dir, staged, published)
        raise
    finally:
        _reclaim_export_staging(run_dir, staging_root)


def _publish_run_root_adapter(config: HarnessConfig, run_dir: Path, staged: Path) -> list[Path]:
    staged.mkdir()
    _write_adapter_files(staged)
    _raise_if_fault(config.fault, config.max_steps, "during_export")
    published: list[Path] = []
    for filename in _ADAPTER_FILENAMES:
        destination = run_dir / filename
        _publish_adapter_file(staged / filename, destination)
        published.append(destination)
    _rmdir_if_empty(staged)
    return published


def _publish_adapter_file(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing adapter file: {destination}")
    rename_noreplace(source, destination)


def _quarantine_failed_export(run_dir: Path, staged: Path, published: list[Path]) -> None:
    for leaked in published:
        if os.path.lexists(leaked):
            quarantine_tree(leaked, reason="export_failure", run_dir=run_dir)
    if staged.exists():
        quarantine_tree(staged, reason="export_failure", run_dir=run_dir)


def _reclaim_export_staging(run_dir: Path, staging_root: Path) -> None:
    if not staging_root.exists() and not staging_root.is_symlink():
        return
    try:
        leftover = staging_root.is_symlink() or any(staging_root.iterdir())
    except OSError:
        leftover = True
    if leftover:
        quarantine_tree(staging_root, reason="leftover_staging", run_dir=run_dir)
        return
    try:
        staging_root.rmdir()
    except OSError:
        quarantine_tree(staging_root, reason="leftover_staging", run_dir=run_dir)


def _rmdir_if_empty(path: Path) -> None:
    try:
        leftover = any(path.iterdir())
    except OSError:
        leftover = True
    if leftover:
        return
    try:
        path.rmdir()
    except OSError:
        return


def _write_checkpoint_tree(staged: Path, snapshot: CheckpointSnapshot) -> None:
    trainer_state = {"global_step": snapshot.step, "train_batch_size": snapshot.batch_size}
    write_fsynced_bytes(staged / "trainer_state.json", (json.dumps(trainer_state) + "\n").encode())
    sidecar = {
        "schema_version": 1,
        "global_step": snapshot.step,
        "sampler_position": snapshot.sampler_position,
        "optimizer_shapes": [[list(shape) for shape in group] for group in _OPTIMIZER_SHAPES],
    }
    write_fsynced_bytes(
        staged / RESUME_SIDECAR_FILENAME,
        (json.dumps(sidecar) + "\n").encode(),
    )
    _write_adapter_files(staged)
    _write_torch(staged / "optimizer.pt", dict(snapshot.optimizer))
    _write_torch(staged / "scheduler.pt", dict(snapshot.scheduler))
    _write_torch(
        staged / "rng_state.pth",
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.zeros(16, dtype=torch.uint8),
        },
    )
    if snapshot.short_write:
        (staged / ADAPTER_WEIGHT_FILES[0]).write_bytes(b"\0\0\0\0")


def _write_adapter_files(directory: Path) -> None:
    config = {
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "harness/fake-base",
    }
    write_fsynced_bytes(directory / "adapter_config.json", (json.dumps(config) + "\n").encode())
    write_fsynced_bytes(directory / ADAPTER_WEIGHT_FILES[0], _safetensors_bytes(_LORA_SHAPES))


def _safetensors_bytes(shapes: Mapping[str, tuple[int, ...]]) -> bytes:
    offset = 0
    payload: dict[str, Any] = {}
    for name, shape in shapes.items():
        size = 4
        for dimension in shape:
            size *= dimension
        payload[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header = json.dumps(payload, separators=(",", ":")).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return len(header).to_bytes(8, "little") + header + b"\0" * offset


def _write_torch(path: Path, payload: object) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    write_fsynced_bytes(path, buffer.getvalue())


def _optimizer_state(step: int) -> dict[str, Any]:
    def adamw_group(params: list[int], weight_decay: float) -> dict[str, Any]:
        return {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": weight_decay,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": True,
            "params": params,
        }

    shapes = ((1, 8), (8, 1))
    return {
        "state": {
            parameter_id: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.full(shape, float(step)),
                "exp_avg_sq": torch.zeros(shape),
            }
            for parameter_id, shape in enumerate(shapes)
        },
        "param_groups": [
            adamw_group(list(range(len(shapes))), 0.01),
            adamw_group([], 0.0),
        ],
    }


def _scheduler_state(step: int) -> dict[str, Any]:
    return {
        "last_epoch": step,
        "_step_count": step + 1,
        "base_lrs": [0.001, 0.001],
        "_last_lr": [0.001, 0.001],
        "lr_lambdas": [{}, {}],
    }


def _advance_optimizer(state: Mapping[str, Any], step: int) -> dict[str, Any]:
    inner = state.get("state")
    if not isinstance(inner, dict):
        return _optimizer_state(step)
    advanced: dict[str, Any] = {
        "state": {},
        "param_groups": state.get("param_groups"),
    }
    for parameter_id, bucket in inner.items():
        if not isinstance(bucket, dict):
            advanced["state"][parameter_id] = bucket
            continue
        updated = dict(bucket)
        updated["step"] = torch.tensor(float(step))
        advanced["state"][parameter_id] = updated
    return advanced


def _advance_scheduler(state: Mapping[str, Any], step: int) -> dict[str, Any]:
    advanced = dict(state)
    advanced["last_epoch"] = step
    advanced["_step_count"] = step + 1
    return advanced


def _load_mapping(inspection: ResumeInspection, filename: str) -> dict[str, Any] | None:
    if inspection.checkpoint is None:
        return None
    payload = torch_mapping(inspection.checkpoint / filename, require_data_record=True)
    return payload if isinstance(payload, dict) else None


def _validate_fault_schedule(fault: FaultSpec, max_steps: int, save_steps: int) -> None:
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


def _seed_rng(seed: int) -> None:
    random.seed(seed)  # nosec B311 - trainer RNG, not a secret
    np.random.seed(seed)
    torch.manual_seed(seed)


def _advance_rng() -> None:
    random.random()  # nosec B311 - advances trainer RNG
    np.random.random()
    torch.rand(1)


def _is_fault(fault: FaultSpec | None, step: int, point: FaultPoint) -> bool:
    return fault is not None and fault.step == step and fault.point == point


def _raise_if_fault(fault: FaultSpec | None, step: int, point: FaultPoint) -> None:
    if fault is None or not _is_fault(fault, step, point):
        return
    raiser = _FAULT_RAISERS.get(fault.kind)
    if raiser is None:
        raise ValueError(f"unsupported fault kind: {fault.kind}")
    raiser()


def _rename_for_fault(fault: FaultSpec | None, step: int) -> Callable[[Path, Path], None] | None:
    if not _is_fault(fault, step, "during_rename"):
        return None

    def boom(_source: Path, _destination: Path) -> None:
        raise OSError("injected rename failure")

    return boom


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
