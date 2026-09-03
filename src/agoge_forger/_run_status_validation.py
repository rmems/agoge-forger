"""Lightweight file validation for operator-facing run readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import CONFIG_MAPPING

from ._run_status_artifact_index import artifact_index_usable
from ._run_status_lora import load_lora_config, lora_config_usable, lora_shapes_usable
from ._run_status_safetensors import has_complete_merged_weights, safetensors_shapes
from ._run_status_torch_archive import torch_mapping
from .config import normalize_revision

PathLike = str | Path


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _merged_config_usable(candidate: Path) -> bool:
    config = _load_json_object(candidate / "config.json")
    model_type = None if config is None else config.get("model_type")
    return isinstance(model_type, str) and model_type in CONFIG_MAPPING


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _revision_usable(payload: dict[str, Any]) -> bool:
    if "revision" not in payload or payload["revision"] is None:
        return True
    try:
        return normalize_revision(payload["revision"]) is not None
    except TypeError:
        return False


def is_merged_model_dir(path: PathLike) -> bool:
    """True for a complete, indexed merged-model export."""
    candidate = Path(path)
    return bool(
        candidate.is_dir()
        and _merged_config_usable(candidate)
        and has_complete_merged_weights(candidate)
        and artifact_index_usable(candidate)
    )


def adapter_config_usable(adapter_path: PathLike | None) -> bool:
    """True for an Agoge LoRA config usable by the default export flow."""
    if adapter_path is None:
        return False
    payload = _load_json_object(Path(adapter_path) / "adapter_config.json")
    if payload is None:
        return False
    base = payload.get("base_model_name_or_path")
    return bool(
        _nonempty_string(base)
        and payload.get("peft_type") == "LORA"
        and _revision_usable(payload)
        and lora_config_usable(payload)
    )


def _adapter_lora_config(adapter_dir: Path) -> Any:
    payload = _load_json_object(adapter_dir / "adapter_config.json")
    return None if payload is None else load_lora_config(payload)


def _legacy_lora_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    payload = torch_mapping(path)
    return {
        key: tuple(value.shape)
        for key, value in (payload or {}).items()
        if isinstance(value, torch.Tensor)
    }


def adapter_weights_usable(
    adapter_path: PathLike | None,
    *,
    allow_unsafe: bool = False,
) -> bool:
    """Validate safetensors, or minimally probe explicitly opted-in legacy bytes."""
    if adapter_path is None:
        return False
    adapter_dir = Path(adapter_path)
    config = _adapter_lora_config(adapter_dir)
    if config is None:
        return False
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.is_file():
        return lora_shapes_usable(safetensors_shapes(safetensors_path), config)
    if allow_unsafe:
        legacy = adapter_dir / "adapter_model.bin"
        if legacy.is_file():
            return lora_shapes_usable(_legacy_lora_shapes(legacy), config)
    return False
