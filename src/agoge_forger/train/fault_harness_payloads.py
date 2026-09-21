"""Checkpoint payload writers and fake optimizer/scheduler/RNG state for the harness."""

from __future__ import annotations

import io
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .._atomic_file import write_fsynced_bytes
from .checkpoints import ADAPTER_CONFIG_FILENAME, ADAPTER_WEIGHT_FILES
from .resume_state import RESUME_SIDECAR_FILENAME

LORA_SHAPES: dict[str, tuple[int, ...]] = {
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (1, 8),
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (8, 1),
}
ADAPTER_FILENAMES = (ADAPTER_CONFIG_FILENAME, ADAPTER_WEIGHT_FILES[0])
OPTIMIZER_SHAPES: list[list[tuple[int, ...]]] = [[(1, 8), (8, 1)], []]


@dataclass(frozen=True)
class CheckpointSnapshot:
    step: int
    sampler_position: int
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    batch_size: int
    short_write: bool = False


def write_checkpoint_tree(staged: Path, snapshot: CheckpointSnapshot) -> None:
    trainer_state = {"global_step": snapshot.step, "train_batch_size": snapshot.batch_size}
    write_fsynced_bytes(staged / "trainer_state.json", (json.dumps(trainer_state) + "\n").encode())
    sidecar = {
        "schema_version": 1,
        "global_step": snapshot.step,
        "sampler_position": snapshot.sampler_position,
        "optimizer_shapes": [[list(shape) for shape in group] for group in OPTIMIZER_SHAPES],
    }
    write_fsynced_bytes(
        staged / RESUME_SIDECAR_FILENAME,
        (json.dumps(sidecar) + "\n").encode(),
    )
    write_adapter_files(staged)
    write_torch(staged / "optimizer.pt", dict(snapshot.optimizer))
    write_torch(staged / "scheduler.pt", dict(snapshot.scheduler))
    write_torch(
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


def write_adapter_files(directory: Path) -> None:
    config = {
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "harness/fake-base",
    }
    write_fsynced_bytes(directory / ADAPTER_CONFIG_FILENAME, (json.dumps(config) + "\n").encode())
    write_fsynced_bytes(directory / ADAPTER_WEIGHT_FILES[0], safetensors_bytes(LORA_SHAPES))


def safetensors_bytes(shapes: Mapping[str, tuple[int, ...]]) -> bytes:
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


def write_torch(path: Path, payload: object) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    write_fsynced_bytes(path, buffer.getvalue())


def optimizer_state(step: int) -> dict[str, Any]:
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


def scheduler_state(step: int) -> dict[str, Any]:
    return {
        "last_epoch": step,
        "_step_count": step + 1,
        "base_lrs": [0.001, 0.001],
        "_last_lr": [0.001, 0.001],
        "lr_lambdas": [{}, {}],
    }


def advance_optimizer(state: Mapping[str, Any], step: int) -> dict[str, Any]:
    inner = state.get("state")
    if not isinstance(inner, dict):
        return optimizer_state(step)
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


def advance_scheduler(state: Mapping[str, Any], step: int) -> dict[str, Any]:
    advanced = dict(state)
    advanced["last_epoch"] = step
    advanced["_step_count"] = step + 1
    return advanced


def seed_rng(seed: int) -> None:
    random.seed(seed)  # nosec B311 - trainer RNG, not a secret
    np.random.seed(seed)
    torch.manual_seed(seed)


def advance_rng() -> None:
    random.random()  # nosec B311 - advances trainer RNG
    _advance_numpy_module_rng()
    torch.rand(1)


def _advance_numpy_module_rng() -> None:
    """Advance the module RandomState that resume restores via ``set_state``."""
    next_seed = int(np.random.default_rng(_numpy_stream_seed()).integers(1, 2**31 - 1))
    np.random.seed(next_seed)


def _numpy_stream_seed() -> int:
    payload = np.random.get_state()
    if not isinstance(payload, tuple) or len(payload) < 3:
        return 1
    keys, position = payload[1], payload[2]
    if not hasattr(keys, "__getitem__"):
        return int(position)
    return int(keys[0]) ^ int(position)
