"""Non-materializing safetensors checks used by run-status."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from safetensors import SafetensorError, safe_open

_NUMBERED_SHARD_RE = re.compile(r"model-(\d+)-of-(\d+)\.safetensors")


def safetensors_keys(path: Path) -> set[str] | None:
    if path.is_symlink():
        return None
    try:
        with safe_open(path, framework="pt", device="cpu") as weights:
            return set(weights.keys())
    except SafetensorError:
        return None


def safetensors_usable(path: Path) -> bool:
    return bool(safetensors_keys(path))


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


def _numbered_shards_complete(shards: set[str]) -> bool:
    example = next((name for name in shards if _NUMBERED_SHARD_RE.fullmatch(name)), None)
    if example is None:
        return True
    match = _NUMBERED_SHARD_RE.fullmatch(example)
    if match is None:
        return False
    ordinal_width = len(match.group(1))
    total_text = match.group(2)
    try:
        total = int(total_text)
    except ValueError:
        return False
    if total != len(shards):
        return False
    return all(
        f"model-{ordinal:0{ordinal_width}d}-of-{total_text}.safetensors" in shards
        for ordinal in range(1, total + 1)
    )


def _load_shard_weight_map(candidate: Path) -> dict[str, Any] | None:
    index_path = candidate / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    return weight_map if isinstance(weight_map, dict) and weight_map else None


def _load_shard_keys(candidate: Path, shards: set[str]) -> dict[str, set[str]] | None:
    shard_keys: dict[str, set[str]] = {}
    for name in shards:
        if not _is_root_model_shard_name(name):
            return None
        shard = candidate / name
        if not shard.is_file():
            return None
        keys = safetensors_keys(shard)
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
    if shards is None or not _numbered_shards_complete(shards):
        return False
    shard_keys = _load_shard_keys(candidate, shards)
    return shard_keys is not None and _weight_map_matches_shards(weight_map, shard_keys)


def has_complete_merged_weights(candidate: Path) -> bool:
    unsharded = candidate / "model.safetensors"
    if unsharded.is_file():
        return safetensors_usable(unsharded)
    return _has_complete_sharded_weights(candidate)
