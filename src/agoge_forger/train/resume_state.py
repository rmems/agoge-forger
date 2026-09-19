"""Inspect and restore single-process checkpoint resume state without false equivalence."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from .._run_status_rng import _cuda_rng_state_usable, _rng_payload_usable, _rng_state_usable
from .._run_status_safetensors import safetensors_usable
from .._run_status_torch_archive import torch_mapping
from .._run_status_trainer_metadata import _trainer_state_step
from .._run_status_trainer_state import _optimizer_payload_usable, _scheduler_payload_usable
from .._run_status_validation import adapter_optimizer_shapes
from .checkpoints import (
    ADAPTER_WEIGHT_FILES,
    LEGACY_ADAPTER_WEIGHT_FILES,
    PathLike,
    list_valid_checkpoints,
    quarantine_incomplete_checkpoints,
    quarantine_tree,
)

RESUME_SIDECAR_FILENAME = "agoge_resume.json"
REQUIRED_RESUME_FIELDS = (
    "global_step",
    "optimizer",
    "scheduler",
    "sampler_position",
    "rng",
    "adapter_weights",
)
_T = TypeVar("_T")


@dataclass(frozen=True)
class ResumeInspection:
    checkpoint: Path | None
    global_step: int | None
    sampler_position: int | None
    restored: tuple[str, ...]
    missing: tuple[str, ...]
    equivalent: bool
    reason: str | None


def inspect_resume_state(
    checkpoint: PathLike | None,
    *,
    restore: bool = False,
    allow_unsafe: bool = False,
) -> ResumeInspection:
    """Describe whether a checkpoint can restore required trainer state.

    ``equivalent`` is true only when every required field is present and usable.
    Present pieces may still be restored when ``restore=True``; that does not
    flip ``equivalent`` if anything required is missing.
    """
    if checkpoint is None:
        return _empty_inspection()
    checkpoint_dir = Path(checkpoint)
    inventory = _resume_inventory(checkpoint_dir, restore=restore, allow_unsafe=allow_unsafe)
    missing = tuple(inventory.missing)
    equivalent = not missing
    return ResumeInspection(
        checkpoint=checkpoint_dir,
        global_step=inventory.global_step,
        sampler_position=inventory.sampler_position,
        restored=tuple(inventory.restored),
        missing=missing,
        equivalent=equivalent,
        reason=None if equivalent else "missing " + ", ".join(missing),
    )


def select_resume_checkpoint(
    run_dir: PathLike, *, allow_unsafe: bool = False, restore: bool = False
) -> ResumeInspection:
    """Quarantine incompletes, then inspect the latest equivalent checkpoint."""
    quarantine_incomplete_checkpoints(run_dir, allow_unsafe=allow_unsafe)
    candidates = list(reversed(list_valid_checkpoints(run_dir, allow_unsafe=allow_unsafe)))
    selected, skipped = _first_equivalent_candidate(candidates, allow_unsafe=allow_unsafe)
    if selected is None:
        latest = candidates[0] if candidates else None
        return inspect_resume_state(latest, restore=restore, allow_unsafe=allow_unsafe)
    _quarantine_skipped_non_equivalent(run_dir, skipped)
    if not restore:
        return selected
    return inspect_resume_state(selected.checkpoint, restore=True, allow_unsafe=allow_unsafe)


def _first_equivalent_candidate(
    candidates: list[Path],
    *,
    allow_unsafe: bool,
) -> tuple[ResumeInspection | None, list[tuple[Path, ResumeInspection]]]:
    skipped: list[tuple[Path, ResumeInspection]] = []
    for checkpoint in candidates:
        inspection = inspect_resume_state(checkpoint, restore=False, allow_unsafe=allow_unsafe)
        if inspection.equivalent:
            return inspection, skipped
        skipped.append((Path(checkpoint), inspection))
    return None, skipped


def _quarantine_skipped_non_equivalent(
    run_dir: PathLike, skipped: list[tuple[Path, ResumeInspection]]
) -> None:
    for checkpoint, inspection in skipped:
        quarantine_tree(
            checkpoint,
            reason="non_equivalent_resume",
            run_dir=run_dir,
            details={"missing": list(inspection.missing)},
        )


@dataclass
class _ResumeInventory:
    restored: list[str]
    missing: list[str]
    global_step: int | None
    sampler_position: int | None


def _empty_inspection() -> ResumeInspection:
    return ResumeInspection(
        checkpoint=None,
        global_step=None,
        sampler_position=None,
        restored=(),
        missing=REQUIRED_RESUME_FIELDS,
        equivalent=False,
        reason="no complete checkpoint",
    )


def _resume_inventory(checkpoint: Path, *, restore: bool, allow_unsafe: bool) -> _ResumeInventory:
    restored: list[str] = []
    missing: list[str] = []
    global_step = _trainer_state_step(checkpoint)
    _record_field("global_step", global_step is not None, restored, missing)
    sidecar = _load_sidecar(checkpoint)
    sampler_position = _sampler_position(sidecar, global_step)
    _record_field("sampler_position", sampler_position is not None, restored, missing)
    _record_field(
        "optimizer",
        _optimizer_usable(
            checkpoint,
            _optimizer_groups(checkpoint, sidecar, allow_unsafe=allow_unsafe),
            global_step,
        ),
        restored,
        missing,
    )
    _record_field("scheduler", _scheduler_usable(checkpoint, global_step), restored, missing)
    rng_ok = _rng_state_usable(checkpoint)
    _record_field("rng", rng_ok, restored, missing)
    _record_field(
        "adapter_weights",
        _adapter_weights_usable(checkpoint, allow_unsafe=allow_unsafe),
        restored,
        missing,
    )
    if restore and rng_ok:
        _restore_rng(checkpoint)
    return _ResumeInventory(restored, missing, global_step, sampler_position)


def _record_field(name: str, present: bool, restored: list[str], missing: list[str]) -> None:
    target = restored if present else missing
    target.append(name)


def _load_sidecar(checkpoint: Path) -> dict[str, Any] | None:
    path = checkpoint / RESUME_SIDECAR_FILENAME
    if path.is_symlink() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _sampler_position(sidecar: dict[str, Any] | None, global_step: int | None) -> int | None:
    if sidecar is None:
        return None
    sidecar_step = _nonneg_int(sidecar.get("global_step"))
    if not _sidecar_step_matches_checkpoint(sidecar_step, global_step):
        return None
    return _nonneg_int(sidecar.get("sampler_position"))


def _sidecar_step_matches_checkpoint(sidecar_step: int | None, global_step: int | None) -> bool:
    if global_step is None:
        return True
    if sidecar_step is None:
        return True
    return sidecar_step == global_step


def _adapter_weights_usable(checkpoint: Path, *, allow_unsafe: bool) -> bool:
    if all(safetensors_usable(checkpoint / name) for name in ADAPTER_WEIGHT_FILES):
        return True
    if not allow_unsafe:
        return False
    return any((checkpoint / name).is_file() for name in LEGACY_ADAPTER_WEIGHT_FILES)


def _nonneg_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < 0:
        return None
    return value


def _positive_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < 1:
        return None
    return value


def _optimizer_groups(
    checkpoint: Path,
    sidecar: dict[str, Any] | None,
    *,
    allow_unsafe: bool,
) -> list[list[tuple[int, ...]]] | None:
    if sidecar is not None:
        return _shape_groups(sidecar.get("optimizer_shapes"))
    return adapter_optimizer_shapes(checkpoint, allow_unsafe=allow_unsafe)


def _optimizer_usable(
    checkpoint: Path,
    expected_groups: list[list[tuple[int, ...]]] | None,
    step: int | None,
) -> bool:
    optimizer = torch_mapping(checkpoint / "optimizer.pt", require_data_record=True)
    return bool(
        expected_groups is not None
        and step is not None
        and _optimizer_payload_usable(optimizer, step, expected_groups)
    )


def _scheduler_usable(checkpoint: Path, global_step: int | None) -> bool:
    optimizer = torch_mapping(checkpoint / "optimizer.pt", require_data_record=True)
    group_count = _param_group_count(optimizer)
    return bool(
        group_count is not None
        and global_step is not None
        and _scheduler_payload_usable(
            torch_mapping(checkpoint / "scheduler.pt"),
            global_step,
            group_count,
        )
    )


def _param_group_count(optimizer: dict[str, Any] | None) -> int | None:
    groups = None if optimizer is None else optimizer.get("param_groups")
    if isinstance(groups, list) and groups:
        return len(groups)
    return None


def _shape_groups(value: Any) -> list[list[tuple[int, ...]]] | None:
    return _parse_list(value, _shape_group)


def _shape_group(group: Any) -> list[tuple[int, ...]] | None:
    return _parse_list(group, _shape_tuple)


def _parse_list(value: Any, parse_item: Callable[[Any], _T | None]) -> list[_T] | None:
    if not isinstance(value, list):
        return None
    parsed_items: list[_T] = []
    for item in value:
        parsed = parse_item(item)
        if parsed is None:
            return None
        parsed_items.append(parsed)
    return parsed_items


def _shape_tuple(shape: Any) -> tuple[int, ...] | None:
    if not isinstance(shape, list) or not shape:
        return None
    if not all(_positive_int(dim) is not None for dim in shape):
        return None
    return tuple(shape)


def _restore_rng(checkpoint: Path) -> None:
    payload = torch_mapping(
        checkpoint / "rng_state.pth",
        allow_numpy=True,
        require_data_record=True,
    )
    if payload is None:
        return
    if not _rng_payload_usable(payload):
        return
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["cpu"])
    _restore_cuda_rng(payload.get("cuda"))


def _restore_cuda_rng(cuda_state: object) -> None:
    if not torch.cuda.is_available():
        return
    if not isinstance(cuda_state, torch.Tensor):
        return
    if not _cuda_rng_state_usable(cuda_state):
        return
    torch.cuda.set_rng_state(cuda_state)
