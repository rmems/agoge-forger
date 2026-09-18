"""Artifact-index layout, digest, and provenance checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from .._strict_json import decode_json_object
from ..config import ExperimentConfig
from ..eval._artifact_schema import (
    ArtifactIndex,
    ArtifactProducerProvenance,
    portable_artifact_path,
)
from ..eval._descriptor_bundle import EntryIdentity, scan_bundle
from ..split_schema import SplitManifest
from .report import BundleFailure
from .schema import ReproducibilityBundle
from .verify_errors import classify_scan_error, index_parse_failure
from .verify_io import classify_io_error, hash_relative, load_object, read_relative

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


def artifact_index_failures(
    root: Path,
    document: ReproducibilityBundle,
    split_digest: str | None,
    split_manifest: SplitManifest | None,
) -> list[BundleFailure]:
    checks = [
        ArtifactCheck(
            root,
            document.adapter_artifact_index_path,
            "peft_adapter",
            split_digest,
            split_manifest,
            document.locked_config_path,
        )
    ]
    if document.merged_artifact_index_path is not None:
        checks.append(
            ArtifactCheck(
                root,
                document.merged_artifact_index_path,
                "merged_model",
                split_digest,
                split_manifest,
                document.locked_config_path,
            )
        )
    failures: list[BundleFailure] = []
    for check in checks:
        failures.extend(_one_artifact_failures(check))
    return failures


def _one_artifact_failures(check: ArtifactCheck) -> list[BundleFailure]:
    loaded = _load_artifact_index(check.root, check.index_path)
    if isinstance(loaded, BundleFailure):
        return [loaded]
    index, identities = loaded
    failures = _membership_failures(check.index_path, index, identities)
    failures.extend(_digest_failures(check, index))
    failures.extend(_layout_failures(check, index))
    failures.extend(_provenance_failures(check, index))
    return failures


def _load_artifact_index(
    root: Path,
    relative: str,
) -> tuple[ArtifactIndex, dict[PurePosixPath, EntryIdentity]] | BundleFailure:
    try:
        payload, _digest = read_relative(root, PurePosixPath(relative))
        value = decode_json_object(payload, str(root / relative), object_label="artifact index")
        index = ArtifactIndex.model_validate(value)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            return index_parse_failure(relative, exc)
        return classify_io_error(exc, relative)
    artifact_root = (root / relative).parent
    try:
        identities = scan_bundle(artifact_root)
    except ValueError as exc:
        return classify_scan_error(exc, relative)
    return index, identities


def _membership_failures(
    index_path: str,
    index: ArtifactIndex,
    identities: dict[PurePosixPath, EntryIdentity],
) -> list[BundleFailure]:
    try:
        expected = {portable_artifact_path(entry.file) for entry in index.artifacts}
    except ValueError as exc:
        return [BundleFailure(code="path_escaping", path=index_path, message=str(exc))]
    actual = {
        path
        for path, identity in identities.items()
        if identity.kind == "file" and path.as_posix() != "artifact_index.json"
    }
    parent = PurePosixPath(index_path).parent
    failures = [
        BundleFailure(
            code="missing_file",
            path=f"{parent / path}",
            message="artifact index references a missing file",
        )
        for path in sorted(expected - actual, key=str)
    ]
    failures.extend(
        BundleFailure(
            code="extra_file",
            path=f"{parent / path}",
            message="artifact directory contains a file omitted from artifact index",
        )
        for path in sorted(actual - expected, key=str)
    )
    return failures


def _digest_failures(check: ArtifactCheck, index: ArtifactIndex) -> list[BundleFailure]:
    failures: list[BundleFailure] = []
    parent = (check.root / check.index_path).parent
    for entry in index.artifacts:
        relative = parent / entry.file
        try:
            portable = relative.relative_to(check.root).as_posix()
            size, digest = hash_relative(check.root, PurePosixPath(portable))
        except (OSError, ValueError) as exc:
            failures.append(
                classify_io_error(exc, f"{PurePosixPath(check.index_path).parent / entry.file}")
            )
            continue
        if size != entry.size_bytes or digest != entry.sha256:
            failures.append(
                BundleFailure(
                    code="modified_file",
                    path=portable,
                    message=(
                        "artifact index digest does not match the file: "
                        f"expected {entry.sha256}, found {digest}"
                    ),
                )
            )
    return failures


def _layout_failures(check: ArtifactCheck, index: ArtifactIndex) -> list[BundleFailure]:
    names = {PurePosixPath(entry.file).as_posix() for entry in index.artifacts}
    if check.kind == "peft_adapter":
        return _adapter_layout_failures(check.index_path, names, check.root)
    return _merged_layout_failures(check, names)


def _adapter_layout_failures(index_path: str, names: set[str], root: Path) -> list[BundleFailure]:
    missing = sorted(_ADAPTER_FILES - names)
    unexpected = sorted(names - _ADAPTER_FILES)
    failures: list[BundleFailure] = []
    if missing:
        failures.append(
            BundleFailure(
                code="artifact_index",
                path=index_path,
                message=f"peft_adapter artifact is missing required files: {missing}",
            )
        )
    if unexpected:
        failures.append(
            BundleFailure(
                code="artifact_index",
                path=index_path,
                message=f"peft_adapter artifact contains unexpected files: {unexpected}",
            )
        )
    if _ADAPTER_CONFIG in names:
        failures.extend(_adapter_config_failures(root, index_path))
    return failures


def _adapter_config_failures(root: Path, index_path: str) -> list[BundleFailure]:
    relative = f"{PurePosixPath(index_path).parent / _ADAPTER_CONFIG}"
    loaded = load_object(root, relative, "adapter config")
    if isinstance(loaded, BundleFailure):
        return [loaded]
    failures: list[BundleFailure] = []
    for field in ("peft_type", "base_model_name_or_path"):
        value = loaded.get(field)
        if not isinstance(value, str) or not value:
            failures.append(
                BundleFailure(
                    code="artifact_index",
                    path=relative,
                    message=f"adapter_config.json requires non-empty {field}",
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
    if has_single:
        return []
    return _shard_failures(check, names)


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
    shards = {str(value) for value in weight_map.values() if isinstance(value, str)}
    if len(shards) != len(weight_map) or not shards:
        return [
            BundleFailure(
                code="artifact_index",
                path=relative,
                message="merged-model weight_map shard paths must be strings",
            )
        ]
    missing = sorted(shards - names)
    if missing:
        return [
            BundleFailure(
                code="missing_file",
                path=check.index_path,
                message=f"merged-model shard index references missing shards: {missing}",
            )
        ]
    return []


def _provenance_failures(check: ArtifactCheck, index: ArtifactIndex) -> list[BundleFailure]:
    provenance = index.producer_provenance
    if provenance is None:
        return [
            BundleFailure(
                code="artifact_index",
                path=check.index_path,
                message="artifact index requires producer_provenance",
            )
        ]
    failures: list[BundleFailure] = []
    if (
        check.split_digest is not None
        and provenance.training_split_manifest_sha256 != check.split_digest
    ):
        failures.append(
            BundleFailure(
                code="split_identity",
                path=check.index_path,
                message="artifact producer provenance does not match the bundled split manifest",
            )
        )
    if check.split_manifest is not None:
        train_digest = check.split_manifest.splits["train"].sha256
        if provenance.training_split_sha256 != train_digest:
            failures.append(
                BundleFailure(
                    code="split_identity",
                    path=check.index_path,
                    message="artifact producer provenance does not match the frozen train split",
                )
            )
    locked = load_object(check.root, check.locked_path, "locked config")
    if isinstance(locked, dict):
        failures.extend(_provenance_config_failures(check.index_path, provenance, locked))
    return failures


def _provenance_config_failures(
    index_path: str,
    provenance: ArtifactProducerProvenance,
    locked: dict[str, Any],
) -> list[BundleFailure]:
    try:
        config = ExperimentConfig.model_validate(locked)
    except ValidationError:
        return []
    failures: list[BundleFailure] = []
    if config.model_id != provenance.base_model_name_or_path:
        failures.append(
            BundleFailure(
                code="locked_config",
                path=index_path,
                message="artifact base model does not match the locked config",
            )
        )
    if config.revision is None or config.revision != provenance.revision:
        failures.append(
            BundleFailure(
                code="locked_config",
                path=index_path,
                message="artifact revision does not match the locked config",
            )
        )
    return failures
