"""Inspect and restore single-process checkpoint resume state without false equivalence."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .._run_status_rng import _rng_payload_usable, _rng_state_usable
from .._run_status_torch_archive import torch_mapping
from .._run_status_trainer_metadata import _json_object, _trainer_state_step
from .._run_status_trainer_state import _optimizer_payload_usable, _scheduler_payload_usable
from .._run_status_validation import adapter_optimizer_shapes
from .checkpoints import PathLike, find_latest_valid_checkpoint, quarantine_incomplete_checkpoints

RESUME_SIDECAR_FILENAME = "agoge_resume.json"
REQUIRED_RESUME_FIELDS = (
    "global_step",
    "optimizer",
    "scheduler",
    "sampler_position",
    "rng",
)


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
        return ResumeInspection(
            checkpoint=None,
            global_step=None,
            sampler_position=None,
            restored=(),
            missing=REQUIRED_RESUME_FIELDS,
            equivalent=False,
            reason="no complete checkpoint",
        )

    checkpoint_dir = Path(checkpoint)
    restored: list[str] = []
    missing: list[str] = []
    global_step = _trainer_state_step(checkpoint_dir)
    if global_step is None:
        missing.append("global_step")
    else:
        restored.append("global_step")

    sidecar = _load_sidecar(checkpoint_dir)
    sampler_position = _sampler_position(checkpoint_dir, sidecar, global_step)
    if sampler_position is None:
        missing.append("sampler_position")
    else:
        restored.append("sampler_position")

    expected_groups = _optimizer_groups(checkpoint_dir, sidecar, allow_unsafe=allow_unsafe)
    optimizer = torch_mapping(checkpoint_dir / "optimizer.pt", require_data_record=True)
    groups = None if optimizer is None else optimizer.get("param_groups")
    group_count = len(groups) if isinstance(groups, list) and groups else None
    if (
        expected_groups is not None
        and global_step is not None
        and _optimizer_payload_usable(optimizer, global_step, expected_groups)
    ):
        restored.append("optimizer")
    else:
        missing.append("optimizer")

    if (
        group_count is not None
        and global_step is not None
        and _scheduler_payload_usable(
            torch_mapping(checkpoint_dir / "scheduler.pt"),
            global_step,
            group_count,
        )
    ):
        restored.append("scheduler")
    else:
        missing.append("scheduler")

    if _rng_state_usable(checkpoint_dir):
        restored.append("rng")
        if restore:
            _restore_rng(checkpoint_dir)
    else:
        missing.append("rng")

    missing_fields = tuple(missing)
    equivalent = not missing_fields
    reason = None if equivalent else "missing " + ", ".join(missing_fields)
    return ResumeInspection(
        checkpoint=checkpoint_dir,
        global_step=global_step,
        sampler_position=sampler_position,
        restored=tuple(restored),
        missing=missing_fields,
        equivalent=equivalent,
        reason=reason,
    )


def select_resume_checkpoint(
    run_dir: PathLike, *, allow_unsafe: bool = False, restore: bool = False
) -> ResumeInspection:
    """Quarantine incompletes, then inspect the latest complete checkpoint."""
    quarantine_incomplete_checkpoints(run_dir, allow_unsafe=allow_unsafe)
    latest = find_latest_valid_checkpoint(run_dir, allow_unsafe=allow_unsafe)
    return inspect_resume_state(latest, restore=restore, allow_unsafe=allow_unsafe)


def _load_sidecar(checkpoint: Path) -> dict[str, Any] | None:
    path = checkpoint / RESUME_SIDECAR_FILENAME
    if path.is_symlink() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _sampler_position(
    checkpoint: Path,
    sidecar: dict[str, Any] | None,
    global_step: int | None,
) -> int | None:
    if sidecar is not None:
        recorded = sidecar.get("sampler_position")
        if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded >= 0:
            return recorded
        return None
    payload = _json_object(checkpoint / "trainer_state.json")
    batch_size = None if payload is None else payload.get("train_batch_size")
    if (
        global_step is None
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        return None
    return global_step * batch_size


def _optimizer_groups(
    checkpoint: Path,
    sidecar: dict[str, Any] | None,
    *,
    allow_unsafe: bool,
) -> list[list[tuple[int, ...]]] | None:
    if sidecar is not None:
        return _shape_groups(sidecar.get("optimizer_shapes"))
    return adapter_optimizer_shapes(checkpoint, allow_unsafe=allow_unsafe)


def _shape_groups(value: Any) -> list[list[tuple[int, ...]]] | None:
    if not isinstance(value, list):
        return None
    groups: list[list[tuple[int, ...]]] = []
    for group in value:
        if not isinstance(group, list):
            return None
        shapes: list[tuple[int, ...]] = []
        for shape in group:
            if not isinstance(shape, list) or not shape:
                return None
            if not all(
                isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in shape
            ):
                return None
            shapes.append(tuple(shape))
        groups.append(shapes)
    return groups


def _restore_rng(checkpoint: Path) -> None:
    payload = torch_mapping(
        checkpoint / "rng_state.pth",
        allow_numpy=True,
        require_data_record=True,
    )
    if payload is None or not _rng_payload_usable(payload):
        return
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["cpu"])
