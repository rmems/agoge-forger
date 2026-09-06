"""Reconstruct and compare Trainer optimizer parameter ordering."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from peft import get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, Trainer


def _serialized_names_by_parameter(model: Any) -> dict[str, str] | None:
    state_names = list(model.state_dict())
    marker_state = {name: torch.tensor(index) for index, name in enumerate(state_names)}
    serialized = get_peft_model_state_dict(
        model,
        state_dict=marker_state,
        save_embedding_layers=False,
    )
    source_by_marker = dict(enumerate(state_names))
    result = {}
    for serialized_name, marker in serialized.items():
        if not isinstance(marker, torch.Tensor) or marker.ndim != 0:
            return None
        source_name = source_by_marker.get(int(marker.item()))
        if source_name is None or source_name in result:
            return None
        result[source_name] = serialized_name
    return result


def _trainer_parameter_names(model: Any) -> list[list[str]]:
    trainer = object.__new__(Trainer)
    decay_names = set(trainer.get_decay_parameter_names(model))
    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    return [
        [name for name, _ in named if name in decay_names],
        [name for name, _ in named if name not in decay_names],
    ]


def ordered_trainable_shapes(
    shapes: dict[str, tuple[int, ...]], config: Any, base_config: Any
) -> list[list[tuple[int, ...]]] | None:
    """Map validated serialized tensors to installed Trainer/PEFT group order."""
    try:
        with torch.device("meta"):
            base = AutoModelForCausalLM.from_config(base_config, trust_remote_code=False)
            model = get_peft_model(base, replace(config, inference_mode=False))
        serialized_names = _serialized_names_by_parameter(model)
        if serialized_names is None:
            return None
        parameters = dict(model.named_parameters())
        ordered_groups = []
        used_serialized_names = set()
        for parameter_names in _trainer_parameter_names(model):
            ordered = []
            for name in parameter_names:
                serialized_name = serialized_names.get(name)
                if serialized_name is None:
                    return None
                parameter = parameters[name]
                serialized_shape = shapes.get(serialized_name)
                if serialized_name in used_serialized_names or serialized_shape != tuple(
                    parameter.shape
                ):
                    return None
                used_serialized_names.add(serialized_name)
                ordered.append(serialized_shape)
            ordered_groups.append(ordered)
        return ordered_groups if used_serialized_names else None
    except (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def optimizer_shapes_match(
    state: dict[int, dict[str, Any]],
    groups: list[Any],
    expected_groups: list[list[tuple[int, ...]]],
) -> bool:
    """Match every optimizer ID to its expected Trainer-group tensor shape."""
    if len(groups) != len(expected_groups):
        return False
    for group, expected_shapes in zip(groups, expected_groups, strict=True):
        parameter_ids = group["params"]
        if len(parameter_ids) != len(expected_shapes):
            return False
        if not all(
            tuple(state[parameter_id]["exp_avg"].shape) == expected
            for parameter_id, expected in zip(parameter_ids, expected_shapes, strict=True)
        ):
            return False
    return True
