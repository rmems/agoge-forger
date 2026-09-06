"""Fail-closed validation for immutable evaluation artifact bundles."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeGuard

from safetensors import SafetensorError, safe_open

from ..split_contract import sha256_bytes
from ._adapter_schema import require_adapter_tensor_schema
from ._artifact_schema import (
    _ADAPTER_CONFIG_PATH,
    _ADAPTER_WEIGHTS_PATH,
    _MERGED_CONFIG_PATH,
    _MERGED_WEIGHTS_INDEX_PATH,
    _MERGED_WEIGHTS_PATH,
    ArtifactIndexEntry,
    ArtifactKind,
    ArtifactProducerProvenance,
    ArtifactValidationContext,
    IndexedArtifacts,
    _portable_artifact_path,
    _require_canonical_model_shard_set,
    _unique_json_object,
)
from ._artifact_snapshot import verified_artifact_snapshot
from ._descriptor_bundle import open_bundle, read_relative_file
from ._merged_model_schema import require_merged_tensor_schema

_SAFETENSORS_SUFFIX = ".safetensors"


@dataclass(frozen=True)
class VerifiedAdapterSource:
    root: Path
    provenance: ArtifactProducerProvenance
    indexed: IndexedArtifacts
    adapter_config: dict[str, object]


@contextmanager
def verified_adapter_source(
    root: Path,
    model_repository: str,
    model_revision: str | None,
) -> Iterator[VerifiedAdapterSource]:
    """Yield a descriptor-pinned, fully verified adapter snapshot for merging."""

    index_path = root / "artifact_index.json"
    expected_digest = _descriptor_index_digest(root)
    with verified_artifact_snapshot(root, index_path, expected_digest) as (index, snapshot):
        provenance = index.producer_provenance
        if provenance is None:
            raise ValueError("peft_adapter artifact index requires producer_provenance")
        effective_revision = model_revision or provenance.revision
        context = ArtifactValidationContext(
            kind="peft_adapter",
            model_repository=model_repository,
            model_revision=effective_revision,
            split_manifest_sha256=provenance.training_split_manifest_sha256,
            train_split_sha256=provenance.training_split_sha256,
        )
        _require_safe_weight_paths(snapshot)
        _require_valid_safetensors(snapshot)
        adapter_config = _require_peft_adapter_structure(snapshot, context)
        _require_producer_provenance(provenance, context)
        yield VerifiedAdapterSource(
            root=snapshot[_ADAPTER_CONFIG_PATH][1].parent,
            provenance=provenance,
            indexed=snapshot,
            adapter_config=adapter_config,
        )


def require_adapter_source_tensor_schema(
    source: VerifiedAdapterSource,
    model_repository: str,
    model_revision: str,
) -> None:
    context = ArtifactValidationContext(
        kind="peft_adapter",
        model_repository=model_repository,
        model_revision=model_revision,
        split_manifest_sha256=source.provenance.training_split_manifest_sha256,
        train_split_sha256=source.provenance.training_split_sha256,
    )
    require_adapter_tensor_schema(source.indexed, source.adapter_config, context)


def _descriptor_index_digest(root: Path) -> str:
    root_descriptor = open_bundle(root)
    try:
        _, digest = read_relative_file(root_descriptor, PurePosixPath("artifact_index.json"))
    finally:
        os.close(root_descriptor)
    return digest


def require_artifact_index(
    path: Path,
    expected: str,
    context: ArtifactValidationContext,
) -> None:
    with verified_artifact_snapshot(path.parent, path, expected) as (index, snapshot):
        _require_artifact_layout(context, snapshot, index.producer_provenance)


def _require_artifact_layout(
    context: ArtifactValidationContext,
    indexed: IndexedArtifacts,
    producer_provenance: ArtifactProducerProvenance | None,
) -> None:
    _require_safe_weight_paths(indexed)
    _require_valid_safetensors(indexed)
    if context.kind == "peft_adapter":
        _require_peft_adapter_layout(indexed, context)
        _require_producer_provenance(producer_provenance, context)
        return
    _require_merged_model_layout(indexed, context, producer_provenance)


def _require_safe_weight_paths(indexed: IndexedArtifacts) -> None:
    unsafe = sorted(str(path) for path in indexed if _is_unsafe_weight_path(path))
    if unsafe:
        raise ValueError(f"artifact bundle contains unsafe serialized weights: {unsafe}")


def _require_peft_adapter_layout(
    indexed: IndexedArtifacts,
    context: ArtifactValidationContext,
) -> None:
    config = _require_peft_adapter_structure(indexed, context)
    require_adapter_tensor_schema(indexed, config, context)


def _require_peft_adapter_structure(
    indexed: IndexedArtifacts,
    context: ArtifactValidationContext,
) -> dict[str, object]:
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
    return config


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
    return len(path.parts) == 1 and path.suffix == _SAFETENSORS_SUFFIX


def _require_no_merged_weights(indexed: IndexedArtifacts) -> None:
    merged_weights = {path for path in indexed if _is_merged_weight_path(path)}
    if merged_weights:
        raise ValueError("peft_adapter artifact cannot contain merged-model weights")


def _is_merged_weight_path(path: PurePosixPath) -> bool:
    return path == _MERGED_WEIGHTS_INDEX_PATH or (
        path.name.startswith("model") and path.suffix == _SAFETENSORS_SUFFIX
    )


def _require_no_unexpected_peft_safetensors(indexed: IndexedArtifacts) -> None:
    unexpected = sorted(str(path) for path in indexed if _is_unexpected_peft_safetensors(path))
    if unexpected:
        raise ValueError(f"peft_adapter artifact contains unexpected safetensors: {unexpected}")


def _require_adapter_provenance(
    config: dict[str, object],
    context: ArtifactValidationContext,
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
    context: ArtifactValidationContext,
    producer_provenance: ArtifactProducerProvenance | None,
) -> None:
    _require_indexed_paths(indexed, {_MERGED_CONFIG_PATH}, kind="merged_model")
    _require_no_adapter_files(indexed)
    config = _require_merged_config(indexed)
    _require_producer_provenance(producer_provenance, context)
    has_single = _MERGED_WEIGHTS_PATH in indexed
    has_sharded = _MERGED_WEIGHTS_INDEX_PATH in indexed
    _require_single_or_sharded_weights(has_single, has_sharded)
    model_weights = {path for path in indexed if path.suffix == _SAFETENSORS_SUFFIX}
    if has_single:
        _require_unsharded_model_weights(model_weights)
    else:
        _require_sharded_model_layout(indexed, model_weights)
    require_merged_tensor_schema(indexed, model_weights, config)


def _require_producer_provenance(
    provenance: ArtifactProducerProvenance | None,
    context: ArtifactValidationContext,
) -> None:
    if provenance is None:
        raise ValueError(f"{context.kind} artifact index requires producer_provenance")
    expected = {
        "base_model_name_or_path": context.model_repository,
        "revision": context.model_revision,
        "training_split_manifest_sha256": context.split_manifest_sha256,
        "training_split_sha256": context.train_split_sha256,
    }
    for field, expected_value in expected.items():
        _require_provenance_value(provenance, field, expected_value)


def _require_provenance_value(
    provenance: ArtifactProducerProvenance, field: str, expected: str
) -> None:
    if getattr(provenance, field) != expected:
        raise ValueError(
            f"artifact producer provenance {field} does not match the contracted "
            "SFT training identity"
        )


def _require_no_adapter_files(indexed: IndexedArtifacts) -> None:
    if _ADAPTER_CONFIG_PATH in indexed:
        raise ValueError("merged_model artifact cannot contain PEFT adapter files")
    if any(path.name.startswith("adapter_model") for path in indexed):
        raise ValueError("merged_model artifact cannot contain PEFT adapter files")


def _require_merged_config(indexed: IndexedArtifacts) -> dict[str, object]:
    config_entry, config_path = indexed[_MERGED_CONFIG_PATH]
    config = _load_verified_json(config_entry, config_path, label="merged-model config")
    _require_nonempty_string(config, "model_type", label="merged-model config")
    return config


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
        raise ValueError("merged-model shard index requires a metadata object")  # noqa: TRY004


def _model_tensor_shard(tensor_name: str, value: object) -> PurePosixPath:
    if not tensor_name:
        raise ValueError("merged-model weight_map tensor names must be non-empty")
    return _require_model_shard_path(value)


def _require_model_shard_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError("merged-model weight_map shard paths must be strings")  # noqa: TRY004
    path = _portable_artifact_path(value)
    if value != path.as_posix():
        raise ValueError("merged-model weight_map shard paths must be canonical")
    if path.suffix != _SAFETENSORS_SUFFIX:
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
    if path.suffix != _SAFETENSORS_SUFFIX or path == _ADAPTER_WEIGHTS_PATH:
        return False
    return not (
        len(path.parts) == 2
        and re.fullmatch(r"checkpoint-\d+", path.parts[0]) is not None
        and path.name == _ADAPTER_WEIGHTS_PATH.name
    )


def _require_valid_safetensors(indexed: IndexedArtifacts) -> None:
    for portable, (_, path) in indexed.items():
        if portable.suffix == _SAFETENSORS_SUFFIX:
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
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected a JSON object")  # noqa: TRY004
    return value


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


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
