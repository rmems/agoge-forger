"""Non-executing Trainer-state integrity checks used by run-status."""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
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


def _trainer_metadata_usable(payload: dict[str, Any], step: int) -> bool:
    global_step = payload.get("global_step")
    train_batch_size = payload.get("train_batch_size")
    return bool(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and global_step == step
        and isinstance(train_batch_size, int)
        and not isinstance(train_batch_size, bool)
        and train_batch_size > 0
    )


def _trainer_state_step(checkpoint: Path) -> int | None:
    payload = _json_object(checkpoint / "trainer_state.json")
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if payload is None or match is None:
        return None
    step = int(match.group(1))
    return step if _trainer_metadata_usable(payload, step) else None


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


def _cpu_contiguous_tensor(value: Any) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.is_contiguous()
    )


def _optimizer_moments_usable(exp_avg: Any, exp_avg_sq: Any) -> bool:
    if not (_cpu_contiguous_tensor(exp_avg) and _cpu_contiguous_tensor(exp_avg_sq)):
        return False
    return all(
        (
            exp_avg.is_floating_point(),
            exp_avg_sq.is_floating_point(),
            exp_avg.dtype == exp_avg_sq.dtype,
            exp_avg.shape == exp_avg_sq.shape,
            exp_avg.numel() > 0,
        )
    )


def _optimizer_step_usable(value: Any, checkpoint_step: int) -> bool:
    if not (_cpu_contiguous_tensor(value) and value.ndim == 0 and value.is_floating_point()):
        return False
    step = float(value.item())
    return math.isfinite(step) and step.is_integer() and 0 < step <= checkpoint_step


def _optimizer_state_entry_usable(value: Any, checkpoint_step: int) -> bool:
    return bool(
        isinstance(value, dict)
        and _optimizer_step_usable(value.get("step"), checkpoint_step)
        and _optimizer_moments_usable(value.get("exp_avg"), value.get("exp_avg_sq"))
    )


def _optimizer_state_entries_usable(state: Any, checkpoint_step: int) -> bool:
    return bool(
        isinstance(state, dict)
        and all(
            _valid_int(key) and _optimizer_state_entry_usable(value, checkpoint_step)
            for key, value in state.items()
        )
    )


def _group_parameter_ids(group: Any) -> list[Any] | None:
    if not isinstance(group, dict):
        return None
    params = group.get("params")
    return params if isinstance(params, list) else None


def _flatten_group_parameter_ids(groups: list[Any]) -> list[Any] | None:
    parameter_ids = []
    for group in groups:
        group_ids = _group_parameter_ids(group)
        if group_ids is None:
            return None
        parameter_ids.extend(group_ids)
    return parameter_ids


def _optimizer_parameter_ids(groups: Any) -> list[Any] | None:
    if not isinstance(groups, list) or not groups:
        return None
    parameter_ids = _flatten_group_parameter_ids(groups)
    if not parameter_ids or not all(_valid_int(param) for param in parameter_ids):
        return None
    return parameter_ids


def _optimizer_shapes_match_adapter(
    state: dict[int, dict[str, Any]],
    adapter_shapes: dict[str, tuple[int, ...]],
) -> bool:
    optimizer_shapes = Counter(tuple(entry["exp_avg"].shape) for entry in state.values())
    return optimizer_shapes == Counter(adapter_shapes.values())


def _optimizer_payload_usable(
    payload: dict[str, Any] | None,
    checkpoint_step: int,
    adapter_shapes: dict[str, tuple[int, ...]],
) -> bool:
    if payload is None or not _optimizer_state_entries_usable(
        payload.get("state"), checkpoint_step
    ):
        return False
    parameter_ids = _optimizer_parameter_ids(payload.get("param_groups"))
    if parameter_ids is None or len(parameter_ids) != len(set(parameter_ids)):
        return False
    state = payload["state"]
    return set(state) == set(parameter_ids) and _optimizer_shapes_match_adapter(
        state, adapter_shapes
    )


def _scheduler_payload_usable(payload: dict[str, Any] | None, step: int) -> bool:
    last_epoch = None if payload is None else payload.get("last_epoch")
    step_count = None if payload is None else payload.get("_step_count")
    return bool(
        _valid_int(last_epoch)
        and last_epoch == step
        and _valid_int(step_count)
        and step_count == step + 1
    )


def _cuda_tensor_usable(value: Any) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.dtype == torch.uint8
        and value.is_contiguous()
        and value.numel() in {8, 16}
    )


def _cuda_rng_state_usable(value: Any) -> bool:
    if not _cuda_tensor_usable(value):
        return False
    if value.numel() == 8:
        return True
    offset = int.from_bytes(bytes(value[8:16].tolist()), byteorder=sys.byteorder, signed=True)
    return offset >= 0 and offset % 4 == 0


def _rng_payload_usable(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    try:
        random.Random().setstate(payload["python"])  # nosec B311 - validates saved RNG state
        np.random.RandomState().set_state(payload["numpy"])
        torch.Generator(device="cpu").set_state(payload["cpu"])
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    return _cuda_rng_state_usable(payload.get("cuda"))


def trainer_state_usable(
    checkpoint: str | Path | None,
    adapter_shapes: dict[str, tuple[int, ...]],
) -> bool:
    if checkpoint is None:
        return False
    checkpoint_dir = Path(checkpoint)
    step = _trainer_state_step(checkpoint_dir)
    return bool(
        step is not None
        and _optimizer_payload_usable(
            torch_mapping(checkpoint_dir / "optimizer.pt"), step, adapter_shapes
        )
        and _scheduler_payload_usable(torch_mapping(checkpoint_dir / "scheduler.pt"), step)
        and _rng_state_usable(checkpoint_dir)
    )
