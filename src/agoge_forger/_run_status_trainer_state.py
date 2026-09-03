"""Non-executing Trainer-state integrity checks used by run-status."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._run_status_torch_archive import torch_mapping

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def _json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _trainer_state_matches_checkpoint(checkpoint: Path) -> bool:
    payload = _json_object(checkpoint / "trainer_state.json")
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if payload is None or match is None:
        return False
    global_step = payload.get("global_step")
    train_batch_size = payload.get("train_batch_size")
    return bool(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and global_step == int(match.group(1))
        and isinstance(train_batch_size, int)
        and not isinstance(train_batch_size, bool)
        and train_batch_size > 0
    )


def _rng_state_usable(checkpoint: Path) -> bool:
    single = checkpoint / "rng_state.pth"
    if single.is_file():
        return _rng_payload_usable(torch_mapping(single, allow_numpy=True))
    # Transformers does not persist the original world size in safe JSON.
    # Rank filenames cannot prove that trailing states are present, so report
    # distributed checkpoints as not ready instead of returning a false positive.
    return False


def _valid_int(value: Any, *, minimum: int | None = None) -> bool:
    return bool(
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _optimizer_state_entries_usable(state: Any) -> bool:
    return bool(
        isinstance(state, dict)
        and all(_valid_int(key) and isinstance(value, dict) for key, value in state.items())
    )


def _group_parameter_ids(group: Any) -> list[Any] | None:
    if not isinstance(group, dict):
        return None
    params = group.get("params")
    return params if isinstance(params, list) else None


def _parameter_lists_usable(parameter_lists: list[list[Any] | None]) -> bool:
    return bool(
        any(parameter_lists)
        and all(
            _valid_int(param)
            for group_params in parameter_lists
            if group_params is not None
            for param in group_params
        )
    )


def _optimizer_groups_usable(groups: Any) -> bool:
    if not isinstance(groups, list) or not groups:
        return False
    parameter_lists = [_group_parameter_ids(group) for group in groups]
    if any(params is None for params in parameter_lists):
        return False
    return _parameter_lists_usable(parameter_lists)


def _optimizer_payload_usable(payload: dict[str, Any] | None) -> bool:
    return bool(
        payload is not None
        and _optimizer_state_entries_usable(payload.get("state"))
        and _optimizer_groups_usable(payload.get("param_groups"))
    )


def _scheduler_payload_usable(payload: dict[str, Any] | None) -> bool:
    return bool(
        payload is not None
        and _valid_int(payload.get("last_epoch"), minimum=-1)
        and _valid_int(payload.get("_step_count"), minimum=0)
    )


def _rng_payload_usable(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    try:
        random.Random().setstate(payload["python"])  # nosec B311 - validates saved RNG state
        np.random.RandomState().set_state(payload["numpy"])
        torch.Generator(device="cpu").set_state(payload["cpu"])
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    return True


def trainer_state_usable(checkpoint: str | Path | None) -> bool:
    if checkpoint is None:
        return False
    checkpoint_dir = Path(checkpoint)
    return bool(
        _trainer_state_matches_checkpoint(checkpoint_dir)
        and _optimizer_payload_usable(torch_mapping(checkpoint_dir / "optimizer.pt"))
        and _scheduler_payload_usable(torch_mapping(checkpoint_dir / "scheduler.pt"))
        and _rng_state_usable(checkpoint_dir)
    )
