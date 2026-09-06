"""Local-only Hugging Face cache snapshot weight validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from ._run_status_safetensors import (
    _is_root_model_shard_name,
    _numbered_shards_complete,
    _shard_filenames,
    _weight_map_matches_shards,
    safetensors_shapes,
)

_MAX_SHARD_INDEX_BYTES = 4 * 1024 * 1024
_WEIGHTS_NAME = "model.safetensors"
_COMMIT_REVISION = re.compile(r"[0-9a-f]{40}")


def immutable_hub_revision(revision: str | None) -> bool:
    """Return whether revision identifies one immutable Hub commit."""
    return isinstance(revision, str) and _COMMIT_REVISION.fullmatch(revision) is not None


def cached_snapshot(repo_id: str, revision: str | None, cache_dir: str | Path) -> Path | None:
    """Resolve exactly one already-cached model snapshot without Hub access."""
    if not immutable_hub_revision(revision):
        return None
    try:
        resolved = snapshot_download(  # nosec B615
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except (OSError, ValueError):
        return None
    if not isinstance(resolved, str):
        return None
    snapshot = Path(resolved)
    return snapshot if snapshot.is_dir() and not snapshot.is_symlink() else None


def _cached_file(snapshot: Path, name: str) -> Path | None:
    candidate = snapshot / name
    if candidate.is_symlink():
        try:
            target = candidate.resolve(strict=True)
            target.relative_to((snapshot.parent.parent / "blobs").resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            return None
        return target if target.is_file() and not target.is_symlink() else None
    return candidate if candidate.is_file() else None


def _physical_shards(snapshot: Path) -> set[str]:
    return {
        entry.name
        for entry in snapshot.iterdir()
        if _is_root_model_shard_name(entry.name) and (entry.is_file() or entry.is_symlink())
    }


def _unsharded_shapes(snapshot: Path) -> dict[str, tuple[int, ...]] | None:
    physical = _physical_shards(snapshot)
    index = snapshot / "model.safetensors.index.json"
    if physical != {_WEIGHTS_NAME} or index.exists() or index.is_symlink():
        return None
    weights = _cached_file(snapshot, _WEIGHTS_NAME)
    return None if weights is None else safetensors_shapes(weights)


def _weight_map(snapshot: Path) -> dict[str, Any] | None:
    index = _cached_file(snapshot, "model.safetensors.index.json")
    if index is None or index.stat().st_size > _MAX_SHARD_INDEX_BYTES:
        return None
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, RecursionError, ValueError):
        return None
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    return weight_map if isinstance(weight_map, dict) and weight_map else None


def _sharded_shapes(snapshot: Path) -> dict[str, tuple[int, ...]] | None:
    weight_map = _weight_map(snapshot)
    if weight_map is None:
        return None
    shards = _shard_filenames(weight_map)
    if (
        shards is None
        or shards != _physical_shards(snapshot)
        or not _numbered_shards_complete(shards)
    ):
        return None
    shard_shapes = {}
    for name in shards:
        if not _is_root_model_shard_name(name):
            return None
        shard = _cached_file(snapshot, name)
        shapes = None if shard is None else safetensors_shapes(shard)
        if not shapes:
            return None
        shard_shapes[name] = shapes
    if not _weight_map_matches_shards(weight_map, shard_shapes):
        return None
    return {
        tensor_name: shape
        for shapes in shard_shapes.values()
        for tensor_name, shape in shapes.items()
    }


def cached_weights_usable(
    snapshot: Path,
    expected_shapes: dict[str, tuple[int, ...]],
) -> bool:
    """Require an exact safe weight inventory from one cache snapshot."""
    shapes = (
        _unsharded_shapes(snapshot)
        if _WEIGHTS_NAME in _physical_shards(snapshot)
        else _sharded_shapes(snapshot)
    )
    return shapes == expected_shapes
