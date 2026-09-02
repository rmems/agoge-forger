"""Lightweight file validation for operator-facing run readiness."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from safetensors import SafetensorError, safe_open

PathLike = str | Path


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _json_object(path: Path) -> bool:
    return _load_json_object(path) is not None


def _safetensors_keys(path: Path) -> set[str] | None:
    """Return tensor keys from a valid container without materializing data."""
    try:
        with safe_open(path, framework="pt", device="cpu") as weights:
            return set(weights.keys())
    except SafetensorError:
        return None


def _safetensors_usable(path: Path) -> bool:
    return bool(_safetensors_keys(path))


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


def _load_shard_weight_map(candidate: Path) -> dict[str, Any] | None:
    index_path = candidate / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    return weight_map


def _load_shard_keys(candidate: Path, shards: set[str]) -> dict[str, set[str]] | None:
    shard_keys: dict[str, set[str]] = {}
    for name in shards:
        if not _is_root_model_shard_name(name):
            return None
        shard = candidate / name
        if not shard.is_file():
            return None
        keys = _safetensors_keys(shard)
        if not keys:
            return None
        shard_keys[name] = keys
    return shard_keys


def _weight_map_matches_shards(
    weight_map: dict[str, Any],
    shard_keys: dict[str, set[str]],
) -> bool:
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(shard_name, str) or tensor_name not in shard_keys[shard_name]:
            return False
    return True


def _has_complete_sharded_weights(candidate: Path) -> bool:
    weight_map = _load_shard_weight_map(candidate)
    if weight_map is None:
        return False
    shards = _shard_filenames(weight_map)
    if shards is None:
        return False
    shard_keys = _load_shard_keys(candidate, shards)
    return shard_keys is not None and _weight_map_matches_shards(weight_map, shard_keys)


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
        and _artifact_index_usable(candidate)
    )


def _artifact_index_usable(candidate: Path) -> bool:
    """Require the completion marker written after tokenizer export."""
    index = _load_json_object(candidate / "artifact_index.json")
    return bool(index and isinstance(index.get("artifacts"), list) and index["artifacts"])


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
    checkpoint_dir = Path(checkpoint)
    return bool(
        _json_object(checkpoint_dir / "trainer_state.json")
        and _torch_state_usable(checkpoint_dir / "optimizer.pt")
        and _torch_state_usable(checkpoint_dir / "scheduler.pt")
    )


def _torch_state_usable(path: Path) -> bool:
    """Validate current PyTorch's zip container without unpickling it."""
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle, zipfile.ZipFile(handle) as archive:
            return bool(archive.namelist())
    except zipfile.BadZipFile:
        return False


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
