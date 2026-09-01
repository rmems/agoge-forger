"""Race-safe safetensor header inspection."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path, PurePosixPath

from safetensors import SafetensorError, safe_open

from ._artifact_schema import ArtifactIndexEntry, IndexedArtifacts


def read_verified_tensor_schema(
    indexed: IndexedArtifacts,
    weights: set[PurePosixPath],
) -> dict[str, tuple[int, ...]]:
    schema: dict[str, tuple[int, ...]] = {}
    with tempfile.TemporaryDirectory(prefix="agoge-tensor-schema-") as snapshot_dir:
        for index, portable in enumerate(sorted(weights)):
            entry, path = indexed[portable]
            snapshot = Path(snapshot_dir) / f"{index:05d}.safetensors"
            copy_verified_tensor_snapshot(entry, path, snapshot, portable)
            try:
                collect_tensor_schema(snapshot, portable, schema)
            finally:
                snapshot.unlink(missing_ok=True)
    return schema


def copy_verified_tensor_snapshot(
    entry: ArtifactIndexEntry,
    source: Path,
    snapshot: Path,
    portable: PurePosixPath,
) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as source_handle, snapshot.open("xb") as snapshot_handle:
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                snapshot_handle.write(chunk)
    except OSError as exc:
        raise ValueError(f"invalid safetensors artifact: {portable}") from exc
    if size != entry.size_bytes or digest.hexdigest() != entry.sha256:
        raise ValueError(f"indexed artifact changed before tensor schema validation: {portable}")


def collect_tensor_schema(
    snapshot: Path,
    portable: PurePosixPath,
    schema: dict[str, tuple[int, ...]],
) -> None:
    try:
        with safe_open(snapshot, framework="pt") as handle:
            tensor_keys = handle.keys()
            for key in tensor_keys:
                if key in schema:
                    raise ValueError(f"tensor key occurs in multiple model shards: {key}")
                schema[key] = tuple(handle.get_slice(key).get_shape())
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"invalid safetensors artifact: {portable}") from exc
