"""Restricted Trainer-state integrity checks used by run-status."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from ._run_status_optimizer_order import optimizer_shapes_match
from ._run_status_rng import _rng_state_usable
from ._run_status_torch_archive import torch_mapping
from ._run_status_trainer_metadata import _trainer_state_step, _valid_int

_ADAMW_BOOL_FIELDS = ("amsgrad", "maximize", "capturable", "differentiable")
_ADAMW_NULLABLE_BOOL_FIELDS = ("foreach", "fused")


def _finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(number) and (minimum is None or number >= minimum)


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


def _optimizer_step_tensor_usable(value: Any) -> bool:
    if not _cpu_contiguous_tensor(value):
        return False
    return value.ndim == 0 and value.is_floating_point()


def _optimizer_step_usable(value: Any, checkpoint_step: int) -> bool:
    if not _optimizer_step_tensor_usable(value):
        return False
    step = float(value.item())
    return all((math.isfinite(step), step.is_integer(), step > 0, step <= checkpoint_step))


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


def _adamw_betas_usable(value: Any) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_finite_number(beta, minimum=0) and beta < 1 for beta in value)
    )


def _adamw_group_scalars_usable(group: dict[str, Any]) -> bool:
    return bool(
        _finite_number(group.get("lr"), minimum=0)
        and _adamw_betas_usable(group.get("betas"))
        and _finite_number(group.get("eps"), minimum=0)
        and _finite_number(group.get("weight_decay"), minimum=0)
    )


def _adamw_group_flags_usable(group: dict[str, Any]) -> bool:
    required = all(isinstance(group.get(field), bool) for field in _ADAMW_BOOL_FIELDS)
    nullable = all(
        group.get(field) is None or isinstance(group.get(field), bool)
        for field in _ADAMW_NULLABLE_BOOL_FIELDS
    )
    decoupled = "decoupled_weight_decay" not in group or group["decoupled_weight_decay"] is True
    return required and nullable and decoupled


def _group_parameter_ids(group: Any) -> list[Any] | None:
    if not isinstance(group, dict):
        return None
    params = group.get("params")
    return (
        params
        if isinstance(params, list)
        and _adamw_group_scalars_usable(group)
        and _adamw_group_flags_usable(group)
        else None
    )


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


def _amsgrad_state_entry_usable(entry: Any) -> bool:
    return bool(
        isinstance(entry, dict)
        and _optimizer_moments_usable(entry.get("exp_avg_sq"), entry.get("max_exp_avg_sq"))
    )


def _amsgrad_group_states_usable(group: Any, state: dict[int, dict[str, Any]]) -> bool:
    parameter_ids = _group_parameter_ids(group)
    if parameter_ids is None:
        return False
    if not group["amsgrad"]:
        return True
    return all(_amsgrad_state_entry_usable(state.get(param)) for param in parameter_ids)


def _amsgrad_states_usable(groups: list[Any], state: dict[int, dict[str, Any]]) -> bool:
    return all(_amsgrad_group_states_usable(group, state) for group in groups)


def _optimizer_payload_usable(
    payload: dict[str, Any] | None,
    checkpoint_step: int,
    adapter_shapes: list[list[tuple[int, ...]]],
) -> bool:
    if payload is None or not _optimizer_state_entries_usable(
        payload.get("state"), checkpoint_step
    ):
        return False
    groups = payload.get("param_groups")
    if not isinstance(groups, list):
        return False
    parameter_ids = _optimizer_parameter_ids(groups)
    if parameter_ids is None or len(parameter_ids) != len(set(parameter_ids)):
        return False
    state = payload["state"]
    return bool(
        set(state) == set(parameter_ids)
        and optimizer_shapes_match(state, groups, adapter_shapes)
        and _amsgrad_states_usable(groups, state)
    )


def _scheduler_rates_usable(value: Any, group_count: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == group_count
        and all(_finite_number(rate) for rate in value)
    )


def _scheduler_lambdas_usable(value: Any, group_count: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == group_count
        and all(entry is None or isinstance(entry, dict) for entry in value)
    )


def _scheduler_payload_usable(
    payload: dict[str, Any] | None,
    step: int,
    group_count: int,
) -> bool:
    last_epoch = None if payload is None else payload.get("last_epoch")
    step_count = None if payload is None else payload.get("_step_count")
    base_lrs = None if payload is None else payload.get("base_lrs")
    last_lrs = None if payload is None else payload.get("_last_lr")
    lr_lambdas = None if payload is None else payload.get("lr_lambdas")
    return all(
        (
            _valid_int(last_epoch),
            last_epoch == step,
            _valid_int(step_count),
            step_count == step + 1,
            _scheduler_rates_usable(base_lrs, group_count),
            _scheduler_rates_usable(last_lrs, group_count),
            _scheduler_lambdas_usable(lr_lambdas, group_count),
        )
    )


def trainer_state_usable(
    checkpoint: str | Path | None,
    adapter_shapes: list[list[tuple[int, ...]]],
) -> bool:
    if checkpoint is None:
        return False
    checkpoint_dir = Path(checkpoint)
    step = _trainer_state_step(checkpoint_dir)
    optimizer = torch_mapping(checkpoint_dir / "optimizer.pt", require_data_record=True)
    groups = None if optimizer is None else optimizer.get("param_groups")
    group_count = len(groups) if isinstance(groups, list) and groups else None
    if step is None or group_count is None:
        return False
    return all(
        (
            _optimizer_payload_usable(optimizer, step, adapter_shapes),
            _scheduler_payload_usable(
                torch_mapping(checkpoint_dir / "scheduler.pt"), step, group_count
            ),
            _rng_state_usable(checkpoint_dir),
        )
    )
