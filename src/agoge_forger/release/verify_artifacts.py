"""Artifact-index layout, digest, and provenance checks."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from .._strict_json import decode_json_object
from ..config import ExperimentConfig
from ..eval import (
    ArtifactIndex,
    ArtifactProducerProvenance,
    EntryIdentity,
    portable_artifact_path,
    scan_bundle,
)
from ..split_schema import SplitManifest
from .report import BundleFailure
from .schema import ReproducibilityBundle
from .verify_errors import classify_scan_error, index_parse_failure
from .verify_io import classify_io_error, hash_relative, load_object, read_relative
from .verify_layout import ArtifactCheck, layout_failures


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
    failures.extend(layout_failures(check, index))
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
    canonical = _canonical_entry_paths(index_path, index)
    if isinstance(canonical, BundleFailure):
        return [canonical]
    expected = set(canonical)
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


def _canonical_entry_paths(
    index_path: str,
    index: ArtifactIndex,
) -> list[PurePosixPath] | BundleFailure:
    try:
        portable = [portable_artifact_path(entry.file) for entry in index.artifacts]
    except ValueError as exc:
        return BundleFailure(code="path_escaping", path=index_path, message=str(exc))
    if any(path.as_posix() != entry.file for path, entry in zip(portable, index.artifacts)):
        return BundleFailure(
            code="path_escaping",
            path=index_path,
            message="artifact index paths must be canonical POSIX paths",
        )
    if len(set(portable)) != len(portable):
        return BundleFailure(
            code="duplicate_path",
            path=index_path,
            message="artifact index paths normalize to the same target",
        )
    return portable


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
