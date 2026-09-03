"""Non-materializing safetensors checks used by run-status."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from safetensors import SafetensorError, safe_open

_NUMBERED_SHARD_RE = re.compile(r"model-(\d+)-of-(\d+)\.safetensors")
_MAX_SHARD_INDEX_BYTES = 4 * 1024 * 1024


def safetensors_keys(path: Path) -> set[str] | None:
    shapes = safetensors_shapes(path)
    return None if shapes is None else set(shapes)


def safetensors_shapes(path: Path) -> dict[str, tuple[int, ...]] | None:
    if path.is_symlink():
        return None
    try:
        with safe_open(path, framework="pt", device="cpu") as weights:
            shapes = {}
            for key in weights.keys():  # noqa: SIM118 - safe_open is not iterable
                tensor = weights.get_slice(key)
                dtype = tensor.get_dtype()
                if dtype != "BF16" and not dtype.startswith("F"):
                    return None
                shapes[key] = tuple(tensor.get_shape())
            return shapes
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
    if index_path.is_symlink() or not index_path.is_file():
        return None
    if index_path.stat().st_size > _MAX_SHARD_INDEX_BYTES:
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (RecursionError, ValueError):
        return None
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    return weight_map if isinstance(weight_map, dict) and weight_map else None


def _load_shard_shapes(
    candidate: Path,
    shards: set[str],
) -> dict[str, dict[str, tuple[int, ...]]] | None:
    shard_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    for name in shards:
        if not _is_root_model_shard_name(name):
            return None
        shard = candidate / name
        if not shard.is_file():
            return None
        shapes = safetensors_shapes(shard)
        if not shapes:
            return None
        shard_shapes[name] = shapes
    return shard_shapes


def _weight_map_matches_shards(
    weight_map: dict[str, Any],
    shard_shapes: dict[str, dict[str, tuple[int, ...]]],
) -> bool:
    indexed_keys: dict[str, set[str]] = {name: set() for name in shard_shapes}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(shard_name, str) or shard_name not in indexed_keys:
            return False
        indexed_keys[shard_name].add(tensor_name)
    return indexed_keys == {name: set(shapes) for name, shapes in shard_shapes.items()}


def _physical_model_shards(candidate: Path) -> set[str]:
    return {entry.name for entry in candidate.iterdir() if _is_root_model_shard_name(entry.name)}


def _shard_set_usable(candidate: Path, shards: set[str]) -> bool:
    return shards == _physical_model_shards(candidate) and _numbered_shards_complete(shards)


def _unsharded_layout_usable(candidate: Path) -> bool:
    shard_index = candidate / "model.safetensors.index.json"
    if shard_index.exists() or shard_index.is_symlink():
        return False
    return _physical_model_shards(candidate) == {"model.safetensors"}


def _complete_sharded_shapes(candidate: Path) -> dict[str, tuple[int, ...]] | None:
    weight_map = _load_shard_weight_map(candidate)
    if weight_map is None:
        return None
    shards = _shard_filenames(weight_map)
    if shards is None:
        return None
    if not _shard_set_usable(candidate, shards):
        return None
    shard_shapes = _load_shard_shapes(candidate, shards)
    if shard_shapes is None or not _weight_map_matches_shards(weight_map, shard_shapes):
        return None
    return {
        tensor_name: shape
        for shapes in shard_shapes.values()
        for tensor_name, shape in shapes.items()
    }


def merged_safetensors_shapes(candidate: Path) -> dict[str, tuple[int, ...]] | None:
    """Return one complete root weight inventory without materializing tensors."""
    unsharded = candidate / "model.safetensors"
    if unsharded.is_file():
        if not _unsharded_layout_usable(candidate):
            return None
        return safetensors_shapes(unsharded)
    return _complete_sharded_shapes(candidate)


def has_complete_merged_weights(
    candidate: Path,
    expected_shapes: dict[str, tuple[int, ...]],
) -> bool:
    return merged_safetensors_shapes(candidate) == expected_shapes
