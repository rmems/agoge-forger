"""Offline tensor-schema validation for merged model artifacts."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from safetensors import SafetensorError, safe_open

from ._artifact_schema import ArtifactIndexEntry, IndexedArtifacts


def require_merged_tensor_schema(
    indexed: IndexedArtifacts,
    model_weights: set[PurePosixPath],
    config: dict[str, object],
) -> None:
    actual = _read_merged_tensor_schema(indexed, model_weights)
    expected = _expected_merged_tensor_schema(config)
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    wrong_shapes = sorted(
        key for key in expected.keys() & actual.keys() if expected[key] != actual[key]
    )
    if any((missing, unexpected, wrong_shapes)):
        raise ValueError(
            "merged-model tensor schema does not match config.json: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"wrong_shapes={wrong_shapes[:5]}"
        )


def _read_merged_tensor_schema(
    indexed: IndexedArtifacts,
    model_weights: set[PurePosixPath],
) -> dict[str, tuple[int, ...]]:
    schema: dict[str, tuple[int, ...]] = {}
    with tempfile.TemporaryDirectory(prefix="agoge-model-schema-") as snapshot_dir:
        for index, portable in enumerate(sorted(model_weights)):
            entry, path = indexed[portable]
            snapshot = Path(snapshot_dir) / f"{index:05d}.safetensors"
            _copy_verified_tensor_snapshot(entry, path, snapshot, portable)
            try:
                collect_tensor_schema(snapshot, portable, schema)
            finally:
                snapshot.unlink(missing_ok=True)
    return schema


def _copy_verified_tensor_snapshot(
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


def _expected_merged_tensor_schema(
    config_payload: dict[str, object],
) -> dict[str, tuple[int, ...]]:
    model = _empty_causal_lm(config_payload)
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    drop_omitted_model_keys(expected, model)
    return expected


def _empty_causal_lm(config_payload: dict[str, object]) -> Any:
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        values = dict(config_payload)
        model_type = values.pop("model_type")
        if not isinstance(model_type, str):
            raise TypeError("model_type must be a string")
        config = AutoConfig.for_model(model_type, **values)
        with init_empty_weights(include_buffers=True):
            return AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    except (ImportError, OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            "merged-model config cannot resolve a local, remote-code-disabled causal LM schema"
        ) from exc


def drop_omitted_model_keys(expected: dict[str, tuple[int, ...]], model: object) -> None:
    tied = getattr(model, "all_tied_weights_keys", None)
    if isinstance(tied, dict):
        for omitted_alias in tied:
            expected.pop(omitted_alias, None)
    ignored = getattr(model, "_keys_to_ignore_on_save", None) or ()
    for ignored_name in ignored:
        expected.pop(ignored_name, None)
