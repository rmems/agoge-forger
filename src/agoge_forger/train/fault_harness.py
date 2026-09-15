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
    publish_checkpoint_tree,
    rollback_run_dir,
)
from .checkpoints import (
    EXPORT_STAGING_PREFIX,
    PathLike,
    checkpoint_step,
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
_OPTIMIZER_SHAPES: list[list[tuple[int, ...]]] = [[(1, 8), (8, 1)], []]


class InjectedOOMError(RuntimeError):
    """CPU stand-in for a CUDA out-of-memory abort."""


class InjectedInterrupt(KeyboardInterrupt):
    """SIGINT-equivalent interruption."""


class InjectedTermination(Exception):
    """SIGTERM-equivalent interruption."""


class InjectedExportError(RuntimeError):
    """Final adapter export aborted before run-root publication."""


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


@dataclass(frozen=True)
class HarnessResult:
    completed: bool
    published_steps: tuple[int, ...]
    export_published: bool
    error_type: str | None
    resume: ResumeInspection
    quarantined: tuple[Path, ...]


def run_fault_harness(config: HarnessConfig) -> HarnessResult:
    """Run a bounded fake training loop with optional injected faults."""
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_rng(config.seed)
    start_step, sampler_position, optimizer = _restore_or_initialize(config, run_dir)
    error_type: str | None = None
    export_published = False
    completed = False
    try:
        _run_steps(config, run_dir, start_step, sampler_position, optimizer)
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


def _restore_or_initialize(config: HarnessConfig, run_dir: Path) -> tuple[int, int, dict[str, Any]]:
    if not config.resume:
        return 0, 0, _optimizer_state(step=1)
    inspection = select_resume_checkpoint(run_dir, restore=True)
    start_step = inspection.global_step or 0
    sampler_position = inspection.sampler_position or 0
    if inspection.checkpoint is not None and "optimizer" in inspection.restored:
        payload = torch_mapping(inspection.checkpoint / "optimizer.pt", require_data_record=True)
        if isinstance(payload, dict):
            return start_step, sampler_position, payload
    return start_step, sampler_position, _optimizer_state(step=max(start_step, 1))


def _run_steps(
    config: HarnessConfig,
    run_dir: Path,
    start_step: int,
    sampler_position: int,
    optimizer: dict[str, Any],
) -> None:
    position = sampler_position
    state = optimizer
    for step in range(start_step + 1, config.max_steps + 1):
        position += config.batch_size
        _advance_rng()
        state = _optimizer_state(step=step)
        if step % config.save_steps != 0:
            continue
        _raise_if_fault(config.fault, step, "before_save")
        write_tree = partial(
            _write_checkpoint_tree,
            step=step,
            sampler_position=position,
            optimizer=state,
            batch_size=config.batch_size,
            short_write=_is_fault(config.fault, step, "during_write"),
        )
        publish_checkpoint_tree(
            run_dir,
            step,
            write_tree,
            rename=_rename_for_fault(config.fault, step),
        )
        _raise_if_fault(config.fault, step, "after_save")


def _export_final_adapter(config: HarnessConfig, run_dir: Path) -> None:
    _raise_if_fault(config.fault, config.max_steps, "during_export")
    require_rename_noreplace_support(run_dir)
    staging_root = Path(tempfile.mkdtemp(prefix=EXPORT_STAGING_PREFIX, dir=run_dir))
    staged = staging_root / "adapter"
    try:
        staged.mkdir()
        _write_adapter_files(staged)
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            destination = run_dir / filename
            if os.path.lexists(destination):
                raise FileExistsError(f"refusing to overwrite existing adapter file: {destination}")
            rename_noreplace(staged / filename, destination)
    except BaseException:
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            leaked = run_dir / filename
            if os.path.lexists(leaked):
                quarantine_tree(leaked, reason="export_failure", run_dir=run_dir)
        if staged.exists():
            quarantine_tree(staged, reason="export_failure", run_dir=run_dir)
        raise
    finally:
        _reclaim_export_staging(run_dir, staging_root)


def _reclaim_export_staging(run_dir: Path, staging_root: Path) -> None:
    if not staging_root.exists():
        return
    leftover = any(staging_root.iterdir())
    if leftover:
        quarantine_tree(staging_root, reason="leftover_staging", run_dir=run_dir)
        return
    staging_root.rmdir()


def _write_checkpoint_tree(
    staged: Path,
    *,
    step: int,
    sampler_position: int,
    optimizer: Mapping[str, Any],
    batch_size: int,
    short_write: bool,
) -> None:
    trainer_state = {"global_step": step, "train_batch_size": batch_size}
    write_fsynced_bytes(staged / "trainer_state.json", (json.dumps(trainer_state) + "\n").encode())
    sidecar = {
        "schema_version": 1,
        "global_step": step,
        "sampler_position": sampler_position,
        "optimizer_shapes": [[list(shape) for shape in group] for group in _OPTIMIZER_SHAPES],
    }
    write_fsynced_bytes(
        staged / RESUME_SIDECAR_FILENAME,
        (json.dumps(sidecar) + "\n").encode(),
    )
    _write_adapter_files(staged)
    _write_torch(staged / "optimizer.pt", dict(optimizer))
    _write_torch(
        staged / "scheduler.pt",
        {
            "last_epoch": step,
            "_step_count": step + 1,
            "base_lrs": [0.001, 0.001],
            "_last_lr": [0.001, 0.001],
            "lr_lambdas": [{}, {}],
        },
    )
    _write_torch(
        staged / "rng_state.pth",
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.zeros(16, dtype=torch.uint8),
        },
    )
    if short_write:
        (staged / "adapter_model.safetensors").write_bytes(b"\0\0\0\0")


def _write_adapter_files(directory: Path) -> None:
    config = {
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "harness/fake-base",
    }
    write_fsynced_bytes(directory / "adapter_config.json", (json.dumps(config) + "\n").encode())
    write_fsynced_bytes(directory / "adapter_model.safetensors", _safetensors_bytes(_LORA_SHAPES))


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
                "exp_avg": torch.zeros(shape),
                "exp_avg_sq": torch.zeros(shape),
            }
            for parameter_id, shape in enumerate(shapes)
        },
        "param_groups": [
            adamw_group(list(range(len(shapes))), 0.01),
            adamw_group([], 0.0),
        ],
    }


def _seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _advance_rng() -> None:
    random.random()
    np.random.random()
    torch.rand(1)


def _is_fault(fault: FaultSpec | None, step: int, point: FaultPoint) -> bool:
    return fault is not None and fault.step == step and fault.point == point


def _raise_if_fault(fault: FaultSpec | None, step: int, point: FaultPoint) -> None:
    if not _is_fault(fault, step, point):
        return
    assert fault is not None
    if fault.kind == "oom":
        raise InjectedOOMError("CUDA out of memory")
    if fault.kind == "sigint":
        raise InjectedInterrupt("SIGINT")
    if fault.kind == "sigterm":
        raise InjectedTermination("SIGTERM")
    if fault.kind == "export_failure":
        raise InjectedExportError("adapter export failed")
    if fault.kind == "rename_failure":
        raise OSError("injected rename failure")
    if fault.kind == "short_write":
        return
    raise ValueError(f"unsupported fault kind: {fault.kind}")


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
