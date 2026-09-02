"""Lightweight file validation for operator-facing run readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import SafetensorError, safe_open

PathLike = str | Path


def _json_object(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return isinstance(payload, dict)


def _safetensors_usable(path: Path) -> bool:
    """Validate a container without materializing any tensor data."""
    try:
        with safe_open(path, framework="pt", device="cpu") as weights:
            tuple(weights.keys())
    except SafetensorError:
        return False
    return True


def _is_root_model_shard_name(name: str) -> bool:
    return bool(
        name
        and name == Path(name).name
        and name != "adapter_model.safetensors"
        and name.endswith(".safetensors")
    )


def _shard_filenames(weight_map: dict[str, Any]) -> set[str] | None:
    names = list(weight_map.values())
    if not all(isinstance(name, str) for name in names):
        return None
    shards = set(names)
    return shards or None


def _has_complete_sharded_weights(candidate: Path) -> bool:
    index_path = candidate / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards = _shard_filenames(weight_map)
    return bool(
        shards
        and all(
            _is_root_model_shard_name(name)
            and (candidate / name).is_file()
            and _safetensors_usable(candidate / name)
            for name in shards
        )
    )


def _has_complete_merged_weights(candidate: Path) -> bool:
    unsharded = candidate / "model.safetensors"
    if unsharded.is_file():
        return _safetensors_usable(unsharded)
    return _has_complete_sharded_weights(candidate)


def is_merged_model_dir(path: PathLike) -> bool:
    """True for a complete merged model and tokenizer save_pretrained tree."""
    candidate = Path(path)
    return bool(
        candidate.is_dir()
        and _json_object(candidate / "config.json")
        and _has_complete_merged_weights(candidate)
        and _json_object(candidate / "tokenizer_config.json")
    )


def adapter_config_usable(adapter_path: PathLike | None) -> bool:
    """True for an Agoge LoRA config usable by the default export flow."""
    if adapter_path is None:
        return False
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.is_file():
        return False
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    base = payload.get("base_model_name_or_path")
    return isinstance(base, str) and bool(base) and payload.get("peft_type") == "LORA"


def trainer_state_usable(checkpoint: PathLike | None) -> bool:
    if checkpoint is None:
        return False
    return _json_object(Path(checkpoint) / "trainer_state.json")


def adapter_weights_usable(
    adapter_path: PathLike | None,
    *,
    allow_unsafe: bool = False,
) -> bool:
    """Validate safetensors, or minimally probe explicitly opted-in legacy bytes."""
    if adapter_path is None:
        return False
    adapter_dir = Path(adapter_path)
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.is_file():
        return _safetensors_usable(safetensors_path)
    if allow_unsafe:
        legacy = adapter_dir / "adapter_model.bin"
        if legacy.is_file():
            with legacy.open("rb") as handle:
                return bool(handle.read(1))
    return False
