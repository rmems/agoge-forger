"""Artifact layout rules for peft_adapter and merged_model indexes."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..artifacts.safetensors_io import UNSAFE_WEIGHT_PATTERNS
from ..eval import ArtifactIndex, ArtifactProducerProvenance
from ..split_schema import SplitManifest
from .report import BundleFailure
from .verify_io import load_object

_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"
_MERGED_CONFIG = "config.json"
_MERGED_WEIGHTS = "model.safetensors"
_MERGED_WEIGHTS_INDEX = "model.safetensors.index.json"
_ADAPTER_FILES = {_ADAPTER_CONFIG, _ADAPTER_WEIGHTS}


@dataclass(frozen=True)
class ArtifactCheck:
    root: Path
    index_path: str
    kind: str
    split_digest: str | None
    split_manifest: SplitManifest | None
    locked_path: str


def layout_failures(check: ArtifactCheck, index: ArtifactIndex) -> list[BundleFailure]:
    names = {PurePosixPath(entry.file).as_posix() for entry in index.artifacts}
    if check.kind == "peft_adapter":
        return _adapter_layout_failures(check, names, index.producer_provenance)
    return _merged_layout_failures(check, names)


def _adapter_layout_failures(
    check: ArtifactCheck,
    names: set[str],
    provenance: ArtifactProducerProvenance | None,
) -> list[BundleFailure]:
    missing = sorted(_ADAPTER_FILES - names)
    unsafe = sorted(
        name
        for name in names - _ADAPTER_FILES
        if any(
            fnmatch.fnmatch(PurePosixPath(name).name, pattern) for pattern in UNSAFE_WEIGHT_PATTERNS
        )
    )
    failures: list[BundleFailure] = []
    if missing:
        failures.append(
            BundleFailure(
                code="artifact_index",
                path=check.index_path,
                message=f"peft_adapter artifact is missing required files: {missing}",
            )
        )
    if unsafe:
        failures.append(
            BundleFailure(
                code="artifact_index",
                path=check.index_path,
                message=f"peft_adapter artifact contains unsafe files: {unsafe}",
            )
        )
    if _ADAPTER_CONFIG in names:
        failures.extend(_adapter_config_failures(check, provenance))
    return failures


def _adapter_config_failures(
    check: ArtifactCheck,
    provenance: ArtifactProducerProvenance | None,
) -> list[BundleFailure]:
    relative = f"{PurePosixPath(check.index_path).parent / _ADAPTER_CONFIG}"
    loaded = load_object(check.root, relative, "adapter config")
    if isinstance(loaded, BundleFailure):
        return [loaded]
    failures = [
        BundleFailure(
            code="artifact_index",
            path=relative,
            message=f"adapter_config.json requires non-empty {field}",
        )
        for field in ("peft_type", "base_model_name_or_path")
        if not isinstance(loaded.get(field), str) or not loaded.get(field)
    ]
    if provenance is not None:
        failures.extend(_adapter_identity_failures(relative, loaded, provenance))
    return failures


def _adapter_identity_failures(
    relative: str,
    loaded: dict[str, Any],
    provenance: ArtifactProducerProvenance,
) -> list[BundleFailure]:
    failures: list[BundleFailure] = []
    if loaded.get("base_model_name_or_path") != provenance.base_model_name_or_path:
        failures.append(
            BundleFailure(
                code="artifact_index",
                path=relative,
                message="adapter_config.json base model does not match artifact provenance",
            )
        )
    if loaded.get("revision") != provenance.revision:
        failures.append(
            BundleFailure(
                code="artifact_index",
                path=relative,
                message="adapter_config.json revision does not match artifact provenance",
            )
        )
    return failures


def _merged_layout_failures(check: ArtifactCheck, names: set[str]) -> list[BundleFailure]:
    if _MERGED_CONFIG not in names:
        return [
            BundleFailure(
                code="artifact_index",
                path=check.index_path,
                message="merged_model artifact is missing config.json",
            )
        ]
    has_single = _MERGED_WEIGHTS in names
    has_sharded = _MERGED_WEIGHTS_INDEX in names
    if has_single == has_sharded:
        return [
            BundleFailure(
                code="artifact_index",
                path=check.index_path,
                message=(
                    "merged_model artifact must contain exactly one of "
                    "model.safetensors or model.safetensors.index.json"
                ),
            )
        ]
    failures = _merged_config_failures(check)
    if has_sharded:
        failures.extend(_shard_failures(check, names))
    return failures


def _merged_config_failures(check: ArtifactCheck) -> list[BundleFailure]:
    relative = f"{PurePosixPath(check.index_path).parent / _MERGED_CONFIG}"
    loaded = load_object(check.root, relative, "merged model config")
    if isinstance(loaded, BundleFailure):
        return [loaded]
    model_type = loaded.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        return [
            BundleFailure(
                code="artifact_index",
                path=relative,
                message="merged-model config.json requires a non-empty model_type",
            )
        ]
    return []


def _shard_failures(check: ArtifactCheck, names: set[str]) -> list[BundleFailure]:
    relative = f"{PurePosixPath(check.index_path).parent / _MERGED_WEIGHTS_INDEX}"
    loaded = load_object(check.root, relative, "merged shard index")
    if isinstance(loaded, BundleFailure):
        return [loaded]
    weight_map = loaded.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return [
            BundleFailure(
                code="artifact_index",
                path=relative,
                message="merged-model shard index requires a non-empty weight_map",
            )
        ]
    if any(not isinstance(value, str) or not value for value in weight_map.values()):
        return [
            BundleFailure(
                code="artifact_index",
                path=relative,
                message="merged-model weight_map shard paths must be strings",
            )
        ]
    return _missing_shard_failures(check, weight_map, names)


def _missing_shard_failures(
    check: ArtifactCheck,
    weight_map: dict[str, Any],
    names: set[str],
) -> list[BundleFailure]:
    missing = sorted(set(weight_map.values()) - names)
    if missing:
        return [
            BundleFailure(
                code="missing_file",
                path=check.index_path,
                message=f"merged-model shard index references missing shards: {missing}",
            )
        ]
    return []
