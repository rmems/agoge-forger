"""Read-only split-manifest checks that never write staging snapshots."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .._strict_json import decode_json_object  # noinspection PyProtectedMember
from ..split_schema import SplitArtifact, SplitManifest, sha256_bytes
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
        relative = _split_relative_path(document, artifact.path)
        entry = listed.get(relative)
        if entry is None:
            failures.append(
                BundleFailure(
                    code="missing_file",
                    path=relative,
                    message=f"{name} split artifact is not listed in the bundle inventory",
                )
            )
            continue
        if entry.sha256 != artifact.sha256:
            failures.append(
                BundleFailure(
                    code="split_identity",
                    path=relative,
                    message=f"{name} split artifact digest does not match the inventory",
                )
            )
    return failures


def split_membership_failures(
    root: Path,
    document: ReproducibilityBundle,
    manifest: SplitManifest,
) -> list[BundleFailure]:
    """Recompute each bundled split's records and compare them to declared members."""

    failures: list[BundleFailure] = []
    for name, artifact in manifest.splits.items():
        relative = _split_relative_path(document, artifact.path)
        try:
            payload, _digest = read_relative(root, PurePosixPath(relative))
        except (OSError, ValueError) as exc:
            failures.append(classify_io_error(exc, relative))
            continue
        failures.extend(_member_failures(relative, name, artifact, payload))
    return failures


def _split_relative_path(document: ReproducibilityBundle, artifact_path: str) -> str:
    manifest_dir = PurePosixPath(document.split_manifest_path).parent
    return (manifest_dir / artifact_path).as_posix()


def _member_failures(
    relative: str,
    name: str,
    artifact: SplitArtifact,
    payload: bytes,
) -> list[BundleFailure]:
    lines = [line for line in payload.splitlines(keepends=True) if line.strip()]
    if len(lines) != artifact.record_count:
        return [
            BundleFailure(
                code="split_identity",
                path=relative,
                message=(
                    f"{name} split record count differs from the manifest: "
                    f"expected {artifact.record_count}, found {len(lines)}"
                ),
            )
        ]
    if any(
        sha256_bytes(line) != member.raw_line_sha256
        for line, member in zip(lines, artifact.members, strict=True)
    ):
        return [
            BundleFailure(
                code="split_identity",
                path=relative,
                message=f"{name} split records do not match the manifest member digests",
            )
        ]
    return []
