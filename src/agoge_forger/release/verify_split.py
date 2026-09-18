"""Read-only split-manifest checks that never write staging snapshots."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .._strict_json import decode_json_object
from ..split_schema import SplitManifest, sha256_bytes
from .report import BundleFailure
from .schema import ReproducibilityBundle
from .verify_errors import json_failure, schema_failure, unknown_schema
from .verify_io import classify_io_error, read_relative


def load_split(
    root: Path,
    document: ReproducibilityBundle,
) -> tuple[SplitManifest, str] | BundleFailure:
    relative = document.split_manifest_path
    try:
        payload, _digest = read_relative(root, PurePosixPath(relative))
        value = decode_json_object(payload, str(root / relative), object_label="split manifest")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and not isinstance(exc, ValidationError):
            return json_failure(exc, relative)
        return classify_io_error(exc, relative)
    version = value.get("schema_version")
    if version != "agoge.split-manifest.v1":
        return unknown_schema(relative, f"unknown split-manifest schema version: {version!r}")
    try:
        manifest = SplitManifest.model_validate(value)
    except ValidationError as exc:
        return schema_failure(exc, relative)
    return manifest, sha256_bytes(payload)


def split_artifact_failures(
    document: ReproducibilityBundle,
    manifest: SplitManifest,
) -> list[BundleFailure]:
    listed = {entry.path: entry for entry in document.files}
    failures: list[BundleFailure] = []
    for name, artifact in manifest.splits.items():
        entry = listed.get(artifact.path)
        if entry is None:
            failures.append(
                BundleFailure(
                    code="missing_file",
                    path=artifact.path,
                    message=f"{name} split artifact is not listed in the bundle inventory",
                )
            )
            continue
        if entry.sha256 != artifact.sha256:
            failures.append(
                BundleFailure(
                    code="split_identity",
                    path=artifact.path,
                    message=f"{name} split artifact digest does not match the inventory",
                )
            )
    return failures
