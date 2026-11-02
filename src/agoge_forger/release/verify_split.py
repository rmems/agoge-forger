"""Read-only split-manifest checks that never write staging snapshots."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .._strict_json import decode_json_object  # noinspection PyProtectedMember
from ..split_materialize import (  # noinspection PyProtectedMember
    LeakageAuditBuilder,
    SourceRecord,
    TrainingRepresentationTracker,
    _build_source_record,
    _content_hash_payload,
    _reject_duplicate_identity,
    _SourceLine,
)
from ..split_schema import SPLIT_NAMES, SplitArtifact, SplitManifest, SplitMember, sha256_bytes
from ..split_validation import (  # noinspection PyProtectedMember
    _identity_fields_match,
    _manifest_records,
    _recorded_source_coordinate_matches,
    _require_expected_ownership,
)
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
    """Recompute each bundled split's records and compare them to declared members.

    Everything runs in memory on bytes already read from the bundle: each line is
    strictly decoded and its member identity re-derived the way split
    materialization does, then the leakage audit and split-policy ownership are
    recomputed from the declared members, as split validation does.
    """

    failures: list[BundleFailure] = []
    tracker = TrainingRepresentationTracker()
    for name, artifact in manifest.splits.items():
        relative = _split_relative_path(document, artifact.path)
        try:
            payload, _digest = read_relative(root, PurePosixPath(relative))
        except (OSError, ValueError) as exc:
            failures.append(classify_io_error(exc, relative))
            continue
        failures.extend(
            _member_failures(relative, name, artifact, payload, _MemberContext(manifest, tracker))
        )
    failures.extend(_manifest_policy_failures(document.split_manifest_path, manifest))
    return failures


@dataclass(frozen=True)
class _MemberContext:
    manifest: SplitManifest
    tracker: TrainingRepresentationTracker


def _split_relative_path(document: ReproducibilityBundle, artifact_path: str) -> str:
    manifest_dir = PurePosixPath(document.split_manifest_path).parent
    return (manifest_dir / artifact_path).as_posix()


def _member_failures(
    relative: str,
    name: str,
    artifact: SplitArtifact,
    payload: bytes,
    context: _MemberContext,
) -> list[BundleFailure]:
    lines = [
        (line_number, line)
        for line_number, line in enumerate(io.BytesIO(payload), 1)
        if line.strip()
    ]
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
    seen_ids: dict[str, str] = {}
    mismatch = False
    for (line_number, line), member in zip(lines, artifact.members, strict=True):
        try:
            actual = _derive_member(relative, line_number, line, context)
            _reject_duplicate_identity(actual.member, seen_ids)
        except (TypeError, ValueError) as exc:
            return [
                BundleFailure(
                    code="split_identity",
                    path=relative,
                    message=f"{name} split record cannot be derived into a split member: {exc}",
                )
            ]
        if not _member_matches(line, actual, member, context.manifest):
            mismatch = True
    if mismatch:
        return [
            BundleFailure(
                code="split_identity",
                path=relative,
                message=f"{name} split records do not match the declared manifest members",
            )
        ]
    return []


def _derive_member(
    relative: str,
    line_number: int,
    line: bytes,
    context: _MemberContext,
) -> SourceRecord:
    """Rebuild one member from a split line exactly as split materialization does."""

    identity = context.manifest.canonical_identity
    coordinate = f"{relative}:{line_number}"
    row = decode_json_object(line, coordinate)
    record = _build_source_record(
        _SourceLine(
            row=row,
            raw_line=line,
            coordinate=coordinate,
            training_payload=_content_hash_payload(row, identity, line_number),
        ),
        identity,
        require_declared_lineage=False,
    )
    if identity.content_hash_policy == "normalized-training-payload-v1":
        context.tracker.observe(row, coordinate)
    return record


def _member_matches(
    line: bytes,
    actual: SourceRecord,
    expected: SplitMember,
    manifest: SplitManifest,
) -> bool:
    # The bundle carries no source file, so raw_line_sha256 cannot be checked
    # offline; the split file holds the materialized bytes.
    return (
        sha256_bytes(line) == expected.materialized_line_sha256
        and _identity_fields_match(actual.member, expected)
        and _recorded_source_coordinate_matches(expected.source_coordinate, manifest)
    )


def _manifest_policy_failures(relative: str, manifest: SplitManifest) -> list[BundleFailure]:
    """Recompute the leakage audit and split ownership from the declared members."""

    failures: list[BundleFailure] = []
    builder = LeakageAuditBuilder()
    for split in SPLIT_NAMES:
        for member in manifest.splits[split].members:
            builder.observe(split, member)
    try:
        audit = builder.result()
    except ValueError as exc:
        failures.append(BundleFailure(code="split_identity", path=relative, message=str(exc)))
    else:
        if audit != manifest.leakage_audit:
            failures.append(
                BundleFailure(
                    code="split_identity",
                    path=relative,
                    message="stored leakage audit differs from the recomputed audit",
                )
            )
    try:
        _require_expected_ownership(manifest, _manifest_records(manifest))
    except ValueError as exc:
        failures.append(BundleFailure(code="split_identity", path=relative, message=str(exc)))
    return failures
