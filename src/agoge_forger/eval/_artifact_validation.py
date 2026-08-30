"""Fail-closed validation for immutable evaluation artifact bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Literal, TypeGuard

from safetensors import SafetensorError, safe_open

from ..split_contract import sha256_bytes
from ._artifact_schema import (
    _ADAPTER_CONFIG_PATH,
    _ADAPTER_WEIGHTS_PATH,
    _MERGED_CONFIG_PATH,
    _MERGED_WEIGHTS_INDEX_PATH,
    _MERGED_WEIGHTS_PATH,
    ArtifactIndex,
    ArtifactIndexEntry,
    ArtifactKind,
    ArtifactProducerProvenance,
    IndexedArtifacts,
    _ArtifactValidationContext,
    _parse_artifact_index,
    _portable_artifact_path,
    _require_canonical_model_shard_set,
    _unique_json_object,
)

ResolvedArtifact = tuple[PurePosixPath, ArtifactIndexEntry, Path]
BundleEntryKind = Literal["directory", "file"]


def _require_artifact_index(
    path: Path,
    expected: str,
    context: _ArtifactValidationContext,
) -> None:
    index = _load_artifact_index(path, expected)
    artifacts = _resolve_index_entries(path, index)
    targets = tuple(target for _, _, target in artifacts)
    _require_unique_artifact_targets(targets)
    for _, entry, target in artifacts:
        _require_matching_artifact(entry, target)
    indexed = {portable: (entry, target) for portable, entry, target in artifacts}
    _require_complete_artifact_bundle(path, targets)
    _require_artifact_layout(context, indexed, index.producer_provenance)


def _resolve_index_entries(index_path: Path, index: ArtifactIndex) -> tuple[ResolvedArtifact, ...]:
    return tuple(_resolve_index_entry(index_path, entry) for entry in index.artifacts)


def _resolve_index_entry(index_path: Path, entry: ArtifactIndexEntry) -> ResolvedArtifact:
    return (
        _portable_artifact_path(entry.file),
        entry,
        _resolve_indexed_target(index_path, entry.file),
    )


def _require_complete_artifact_bundle(index_path: Path, targets: tuple[Path, ...]) -> None:
    bundle_files = _enumerate_artifact_bundle(index_path.parent, index_path)
    omitted = bundle_files.difference(targets)
    if omitted:
        relative = sorted(str(path.relative_to(index_path.parent)) for path in omitted)
        raise ValueError(f"artifact bundle contains files omitted from artifact index: {relative}")


def _enumerate_artifact_bundle(root: Path, index_path: Path) -> set[Path]:
    files: set[Path] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in _read_bundle_directory(directory):
            _collect_bundle_entry(entry, index_path, files, pending)
    return files


def _read_bundle_directory(directory: Path) -> list[os.DirEntry[str]]:
    try:
        return sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError as exc:
        raise ValueError(f"artifact bundle is not readable: {directory}") from exc


def _collect_bundle_entry(
    entry: os.DirEntry[str],
    index_path: Path,
    files: set[Path],
    pending: list[Path],
) -> None:
    kind, candidate = _bundle_entry_kind(entry)
    if kind == "directory":
        pending.append(candidate)
        return
    _record_bundle_file(candidate, index_path, files)


def _bundle_entry_kind(entry: os.DirEntry[str]) -> tuple[BundleEntryKind, Path]:
    candidate = Path(entry.path)
    metadata = _bundle_entry_metadata(entry, candidate)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"artifact bundle cannot contain symlinks: {candidate}")
    if stat.S_ISDIR(metadata.st_mode):
        return "directory", candidate
    if stat.S_ISREG(metadata.st_mode):
        return "file", candidate
    raise ValueError(f"artifact bundle entry is not a regular file: {candidate}")


def _bundle_entry_metadata(entry: os.DirEntry[str], candidate: Path) -> os.stat_result:
    try:
        return entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"artifact bundle entry is not readable: {candidate}") from exc


def _record_bundle_file(candidate: Path, index_path: Path, files: set[Path]) -> None:
    resolved = candidate.resolve(strict=True)
    if resolved != index_path.resolve(strict=True):
        files.add(resolved)


def _require_artifact_layout(
    context: _ArtifactValidationContext,
    indexed: IndexedArtifacts,
    producer_provenance: ArtifactProducerProvenance | None,
) -> None:
    _require_safe_weight_paths(indexed)
    _require_valid_safetensors(indexed)
    if context.kind == "peft_adapter":
        _require_peft_adapter_layout(indexed, context)
        return
    _require_merged_model_layout(indexed, context, producer_provenance)


def _require_safe_weight_paths(indexed: IndexedArtifacts) -> None:
    unsafe = sorted(str(path) for path in indexed if _is_unsafe_weight_path(path))
    if unsafe:
        raise ValueError(f"artifact bundle contains unsafe serialized weights: {unsafe}")


def _require_peft_adapter_layout(
    indexed: IndexedArtifacts,
    context: _ArtifactValidationContext,
) -> None:
    _require_indexed_paths(
        indexed,
        {_ADAPTER_CONFIG_PATH, _ADAPTER_WEIGHTS_PATH},
        kind="peft_adapter",
    )
    _require_no_merged_config(indexed)
    _require_peft_root_weights(indexed)
    _require_no_merged_weights(indexed)
    _require_no_unexpected_peft_safetensors(indexed)
    entry, config_path = indexed[_ADAPTER_CONFIG_PATH]
    config = _load_verified_json(entry, config_path, label="PEFT adapter config")
    _require_nonempty_string(config, "peft_type", label="PEFT adapter config")
    _require_adapter_provenance(config, context)


def _require_no_merged_config(indexed: IndexedArtifacts) -> None:
    if _MERGED_CONFIG_PATH in indexed:
        raise ValueError("peft_adapter artifact cannot contain merged-model config.json")


def _require_peft_root_weights(indexed: IndexedArtifacts) -> None:
    root_safetensors = {path for path in indexed if _is_root_safetensor(path)}
    if root_safetensors != {_ADAPTER_WEIGHTS_PATH}:
        raise ValueError(
            "peft_adapter artifact must contain only adapter_model.safetensors weights"
        )


def _is_root_safetensor(path: PurePosixPath) -> bool:
    return len(path.parts) == 1 and path.suffix == ".safetensors"


def _require_no_merged_weights(indexed: IndexedArtifacts) -> None:
    merged_weights = {path for path in indexed if _is_merged_weight_path(path)}
    if merged_weights:
        raise ValueError("peft_adapter artifact cannot contain merged-model weights")


def _is_merged_weight_path(path: PurePosixPath) -> bool:
    return path == _MERGED_WEIGHTS_INDEX_PATH or (
        path.name.startswith("model") and path.suffix == ".safetensors"
    )


def _require_no_unexpected_peft_safetensors(indexed: IndexedArtifacts) -> None:
    unexpected = sorted(str(path) for path in indexed if _is_unexpected_peft_safetensors(path))
    if unexpected:
        raise ValueError(f"peft_adapter artifact contains unexpected safetensors: {unexpected}")


def _require_adapter_provenance(
    config: dict[str, object],
    context: _ArtifactValidationContext,
) -> None:
    expected = {
        "base_model_name_or_path": context.model_repository,
        "revision": context.model_revision,
    }
    for field, expected_value in expected.items():
        actual = _require_adapter_provenance_value(config, field)
        if actual != expected_value:
            raise ValueError(
                f"PEFT adapter config {field} does not match the contracted SFT arm: "
                f"expected {expected_value!r}, found {actual!r}"
            )


def _require_adapter_provenance_value(config: dict[str, object], field: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"PEFT adapter config is missing required {field}")
    return value


def _require_merged_model_layout(
    indexed: IndexedArtifacts,
    context: _ArtifactValidationContext,
    producer_provenance: ArtifactProducerProvenance | None,
) -> None:
    _require_indexed_paths(indexed, {_MERGED_CONFIG_PATH}, kind="merged_model")
    _require_no_adapter_files(indexed)
    _require_merged_config(indexed)
    _require_merged_producer_provenance(producer_provenance, context)
    has_single = _MERGED_WEIGHTS_PATH in indexed
    has_sharded = _MERGED_WEIGHTS_INDEX_PATH in indexed
    _require_single_or_sharded_weights(has_single, has_sharded)
    model_weights = {path for path in indexed if path.suffix == ".safetensors"}
    if has_single:
        _require_unsharded_model_weights(model_weights)
        return
    _require_sharded_model_layout(indexed, model_weights)


def _require_merged_producer_provenance(
    provenance: ArtifactProducerProvenance | None,
    context: _ArtifactValidationContext,
) -> None:
    if provenance is None:
        raise ValueError("merged_model artifact index requires producer_provenance")
    if provenance.base_model_name_or_path != context.model_repository:
        raise ValueError(
            "merged_model producer provenance base_model_name_or_path does not match "
            "the contracted SFT arm"
        )
    if provenance.revision != context.model_revision:
        raise ValueError(
            "merged_model producer provenance revision does not match the contracted SFT arm"
        )


def _require_no_adapter_files(indexed: IndexedArtifacts) -> None:
    if _ADAPTER_CONFIG_PATH in indexed:
        raise ValueError("merged_model artifact cannot contain PEFT adapter files")
    if any(path.name.startswith("adapter_model") for path in indexed):
        raise ValueError("merged_model artifact cannot contain PEFT adapter files")


def _require_merged_config(indexed: IndexedArtifacts) -> None:
    config_entry, config_path = indexed[_MERGED_CONFIG_PATH]
    config = _load_verified_json(config_entry, config_path, label="merged-model config")
    _require_nonempty_string(config, "model_type", label="merged-model config")


def _require_single_or_sharded_weights(has_single: bool, has_sharded: bool) -> None:
    if has_single == has_sharded:
        raise ValueError(
            "merged_model artifact must contain exactly one of model.safetensors or "
            "model.safetensors.index.json"
        )


def _require_unsharded_model_weights(model_weights: set[PurePosixPath]) -> None:
    if model_weights != {_MERGED_WEIGHTS_PATH}:
        raise ValueError("unsharded merged_model artifact contains unexpected model shards")


def _require_sharded_model_layout(
    indexed: IndexedArtifacts,
    model_weights: set[PurePosixPath],
) -> None:
    entry, shard_index_path = indexed[_MERGED_WEIGHTS_INDEX_PATH]
    shard_index = _load_verified_json(entry, shard_index_path, label="merged-model shard index")
    tensor_shards = _model_tensor_shards(shard_index)
    referenced_shards = set(tensor_shards.values())
    _require_indexed_shards(indexed, referenced_shards)
    _require_only_referenced_shards(model_weights, referenced_shards)
    _require_exact_shard_tensor_map(indexed, tensor_shards)


def _require_indexed_shards(
    indexed: IndexedArtifacts,
    referenced_shards: set[PurePosixPath],
) -> None:
    missing = sorted(str(path) for path in referenced_shards if path not in indexed)
    if missing:
        raise ValueError(f"merged-model shard index references unindexed shards: {missing}")


def _require_only_referenced_shards(
    model_weights: set[PurePosixPath],
    referenced_shards: set[PurePosixPath],
) -> None:
    unexpected = sorted(str(path) for path in model_weights.difference(referenced_shards))
    if referenced_shards != model_weights:
        raise ValueError(
            f"merged_model artifact contains shards absent from weight_map: {unexpected}"
        )


def _model_tensor_shards(shard_index: dict[str, object]) -> dict[str, PurePosixPath]:
    weight_map = _require_weight_map(shard_index)
    _require_shard_metadata(shard_index)
    tensor_shards = {
        tensor_name: _model_tensor_shard(tensor_name, value)
        for tensor_name, value in weight_map.items()
    }
    _require_canonical_model_shard_set(set(tensor_shards.values()))
    return tensor_shards


def _require_weight_map(shard_index: dict[str, object]) -> dict[str, object]:
    weight_map = shard_index.get("weight_map")
    if not _is_json_object(weight_map):
        raise ValueError("merged-model shard index requires a non-empty weight_map")
    if not weight_map:
        raise ValueError("merged-model shard index requires a non-empty weight_map")
    return weight_map


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _require_shard_metadata(shard_index: dict[str, object]) -> None:
    if not isinstance(shard_index.get("metadata"), dict):
        raise TypeError("merged-model shard index requires a metadata object")


def _model_tensor_shard(tensor_name: str, value: object) -> PurePosixPath:
    if not tensor_name:
        raise ValueError("merged-model weight_map tensor names must be non-empty")
    if not isinstance(value, str):
        raise TypeError("merged-model weight_map shard paths must be strings")
    path = _portable_artifact_path(value)
    if value != path.as_posix():
        raise ValueError("merged-model weight_map shard paths must be canonical")
    if path.suffix != ".safetensors":
        raise ValueError("merged-model weight_map must reference safetensors shards only")
    return path


def _require_exact_shard_tensor_map(
    indexed: IndexedArtifacts,
    expected: dict[str, PurePosixPath],
) -> None:
    actual = _read_actual_tensor_map(indexed, expected)
    _require_expected_tensor_keys(expected, actual)


def _read_actual_tensor_map(
    indexed: IndexedArtifacts,
    expected: dict[str, PurePosixPath],
) -> dict[str, PurePosixPath]:
    actual: dict[str, PurePosixPath] = {}
    for shard in set(expected.values()):
        keys = _read_safetensor_keys(indexed[shard][1], shard)
        duplicate = sorted(key for key in keys if key in actual)
        if duplicate:
            raise ValueError(f"tensor keys occur in multiple merged-model shards: {duplicate}")
        actual.update(dict.fromkeys(keys, shard))
    return actual


def _require_expected_tensor_keys(
    expected: dict[str, PurePosixPath],
    actual: dict[str, PurePosixPath],
) -> None:
    missing = sorted(set(expected).difference(actual))
    if missing:
        raise ValueError(f"merged-model weight_map names tensors absent from shards: {missing}")
    extra = sorted(set(actual).difference(expected))
    if extra:
        raise ValueError(f"merged-model shards contain tensors absent from weight_map: {extra}")
    misplaced = sorted(key for key in expected if expected[key] != actual[key])
    if misplaced:
        raise ValueError(f"merged-model weight_map assigns tensors to wrong shards: {misplaced}")


def _is_unsafe_weight_path(path: PurePosixPath) -> bool:
    name = path.name.lower()
    suffixes = {suffix.lower() for suffix in path.suffixes}
    pickle_family = name.startswith(("adapter_model", "pytorch_model")) and ".bin" in suffixes
    return pickle_family or path.suffix.lower() == ".ckpt"


def _is_unexpected_peft_safetensors(path: PurePosixPath) -> bool:
    if path.suffix != ".safetensors" or path == _ADAPTER_WEIGHTS_PATH:
        return False
    return not (
        len(path.parts) == 2
        and re.fullmatch(r"checkpoint-\d+", path.parts[0]) is not None
        and path.name == _ADAPTER_WEIGHTS_PATH.name
    )


def _require_valid_safetensors(indexed: IndexedArtifacts) -> None:
    for portable, (_, path) in indexed.items():
        if portable.suffix == ".safetensors":
            _read_safetensor_keys(path, portable)


def _read_safetensor_keys(path: Path, portable: PurePosixPath) -> set[str]:
    try:
        with safe_open(path, framework="pt") as handle:
            keys = set(handle.keys())
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"invalid safetensors artifact: {portable}") from exc
    if not keys:
        raise ValueError(f"safetensors file contains no tensors: {portable}")
    return keys


def _require_nonempty_string(config: dict[str, object], field: str, *, label: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} requires non-empty {field}")
    return value


def _require_indexed_paths(
    indexed: IndexedArtifacts,
    required: set[PurePosixPath],
    *,
    kind: ArtifactKind,
) -> None:
    missing = sorted(str(path) for path in required.difference(indexed))
    if missing:
        raise ValueError(f"{kind} artifact is missing required indexed files: {missing}")


def _load_verified_json(
    entry: ArtifactIndexEntry,
    path: Path,
    *,
    label: str,
) -> dict[str, object]:
    payload = _read_regular_file(path, label=f"invalid {label}")
    if len(payload) != entry.size_bytes or sha256_bytes(payload) != entry.sha256:
        raise ValueError(f"{label} changed after artifact validation: {path}")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"invalid {label}: expected a JSON object")
    return value


def _load_artifact_index(path: Path, expected: str) -> ArtifactIndex:
    payload = _read_artifact_index(path)
    actual = sha256_bytes(payload)
    if actual != expected:
        raise ValueError(
            f"SFT artifact-index SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return _parse_artifact_index(path, payload)


def _read_artifact_index(path: Path) -> bytes:
    return _read_regular_file(path, label="invalid SFT artifact index")


def _read_regular_file(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label}: not a regular file: {path}")
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValueError(f"{label}: {path}") from exc
    if _file_identity(before) != _file_identity(after):
        raise ValueError(f"{label}: file changed while reading: {path}")
    return payload


def _resolve_indexed_target(index_path: Path, value: str) -> Path:
    target = _resolve_artifact_entry(index_path.parent.resolve(), value)
    if target == index_path.resolve():
        raise ValueError("artifact index cannot list itself")
    return target


def _require_unique_artifact_targets(targets: tuple[Path, ...]) -> None:
    if len(targets) != len(set(targets)):
        raise ValueError("artifact index resolves duplicate targets")


def _require_matching_artifact(entry: ArtifactIndexEntry, path: Path) -> None:
    actual_size, actual_digest = _stream_artifact(path)
    if actual_size != entry.size_bytes:
        raise ValueError(
            f"indexed artifact size mismatch for {entry.file}: "
            f"expected {entry.size_bytes}, found {actual_size}"
        )
    if actual_digest != entry.sha256:
        raise ValueError(
            f"indexed artifact SHA-256 mismatch for {entry.file}: "
            f"expected {entry.sha256}, found {actual_digest}"
        )


def _stream_artifact(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"indexed artifact is not a regular file: {path}")
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValueError(f"indexed artifact is not a readable file: {path}") from exc
    if _file_identity(before) != _file_identity(after):
        raise ValueError(f"indexed artifact changed during validation: {path}")
    return before.st_size, digest.hexdigest()


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _resolve_artifact_entry(root: Path, value: str) -> Path:
    portable = _portable_artifact_path(value)
    resolved = _resolve_existing_artifact(root, portable, value)
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact index path escapes its directory: {value}")
    return resolved


def _resolve_existing_artifact(root: Path, portable: PurePosixPath, value: str) -> Path:
    try:
        return root.joinpath(*portable.parts).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"indexed artifact does not exist: {value}") from exc
