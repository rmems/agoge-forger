"""Validation and frozen-record loaders for canonical split manifests."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ._source_snapshot import copy_source_snapshot
from ._strict_json import decode_json_object
from ._validation_staging import (
    unwritable_staging_error,
    validation_directory,
    validation_staging_dir,
)
from .split_materialize import (
    LeakageAuditBuilder,
    SourceRecord,
    TrainingRepresentationTracker,
    assign_records,
    iter_source_records,
    read_source_records,
)
from .split_schema import (
    SPLIT_NAMES,
    SplitArtifact,
    SplitManifest,
    SplitMaterializationSpec,
    SplitMember,
    SplitName,
)


def load_split_manifest(manifest_path: str | Path) -> SplitManifest:
    path = Path(manifest_path).expanduser().resolve(strict=True)
    return _load_split_manifest_snapshot(path.read_bytes(), path)


def _load_split_manifest_snapshot(content: bytes, path: Path) -> SplitManifest:
    try:
        value = decode_json_object(content, str(path), object_label="split manifest")
    except ValueError as exc:
        raise ValueError(f"invalid split manifest JSON: {path}") from exc
    return SplitManifest.model_validate(value)


def validate_split_manifest(
    manifest_path: str | Path, *, source_path: str | Path | None = None
) -> SplitManifest:
    """Validate schema, source identity, artifacts, members, and leakage gates."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    return validate_split_manifest_snapshot(path, path.read_bytes(), source_path=source_path)


def validate_split_manifest_snapshot(
    manifest_path: str | Path,
    content: bytes,
    *,
    source_path: str | Path | None = None,
) -> SplitManifest:
    """Validate one already-read manifest snapshot and its referenced artifacts."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = _load_split_manifest_snapshot(content, path)
    return _validate_split_manifest(path, manifest, source_path)


def _validate_split_manifest(
    path: Path,
    manifest: SplitManifest,
    source_path: str | Path | None,
) -> SplitManifest:
    if source_path is not None:
        _validate_source(manifest, Path(source_path).expanduser().resolve(strict=True))
    context = _ArtifactValidationContext(
        manifest_path=path,
        manifest=manifest,
        audit_builder=LeakageAuditBuilder(),
        representation_tracker=TrainingRepresentationTracker(),
        source_authenticated=source_path is not None,
    )
    for split, artifact in manifest.splits.items():
        _validate_artifact(context, split, artifact)
    audit = context.audit_builder.result()
    _require_equal(
        audit,
        manifest.leakage_audit,
        "stored leakage audit differs from recomputed audit",
    )
    _require_expected_ownership(manifest, _manifest_records(manifest))
    return manifest


def _manifest_records(manifest: SplitManifest) -> list[SourceRecord]:
    return [
        SourceRecord(row={}, raw_line=b"", member=member)
        for split in SPLIT_NAMES
        for member in manifest.splits[split].members
    ]


def _validate_source(manifest: SplitManifest, source: Path) -> None:
    with validation_directory(".agoge-source-validation-", source.parent) as snapshot_dir:
        snapshot = snapshot_dir / "source.jsonl"
        actual_source_sha = copy_source_snapshot(source, snapshot)
        _require_equal(
            actual_source_sha,
            manifest.source.sha256,
            f"source SHA-256 mismatch: expected {manifest.source.sha256}, "
            f"found {actual_source_sha}",
        )
        records = read_source_records(
            snapshot,
            manifest.canonical_identity,
            source_coordinate_path=manifest.source.path,
        )
    _require_expected_ownership(manifest, records)
    source_members = {record.member.canonical_id: record.member for record in records}
    manifest_members = {
        member.canonical_id: member
        for artifact in manifest.splits.values()
        for member in artifact.members
    }
    _require_equal(
        source_members,
        manifest_members,
        "manifest membership metadata differs from the pinned source",
    )


def _require_expected_ownership(manifest: SplitManifest, records: list[SourceRecord]) -> None:
    spec = SplitMaterializationSpec(
        source_repository=manifest.source.repository,
        source_revision=manifest.source.revision,
        dataset_version=manifest.source.dataset_version,
        source_path=manifest.source.path,
        split_policy=manifest.split_policy,
        canonical_identity=manifest.canonical_identity,
    )
    expected = assign_records(records, spec)
    expected_ids = {
        split: tuple(records[index].member.canonical_id for index in expected[split])
        for split in SPLIT_NAMES
    }
    manifest_ids = {
        split: tuple(member.canonical_id for member in manifest.splits[split].members)
        for split in SPLIT_NAMES
    }
    _require_equal(
        expected_ids,
        manifest_ids,
        "manifest split ownership differs from the pinned split policy",
    )


@dataclass(frozen=True)
class _ArtifactValidationContext:
    manifest_path: Path
    manifest: SplitManifest
    audit_builder: LeakageAuditBuilder
    representation_tracker: TrainingRepresentationTracker
    source_authenticated: bool


def _validate_artifact(
    context: _ArtifactValidationContext,
    split: SplitName,
    artifact: SplitArtifact,
) -> None:
    with verified_split_snapshot(context.manifest_path, split, artifact) as snapshot_path:
        count = 0
        membership_mismatch = False
        records = iter_source_records(
            snapshot_path,
            context.manifest.canonical_identity,
            source_coordinate_path=artifact.path,
            representation_tracker=context.representation_tracker,
        )
        for count, record in enumerate(records, 1):
            if count > artifact.record_count:
                continue
            expected = artifact.members[count - 1]
            actual = record.member
            if not _artifact_member_matches(actual, expected, context):
                membership_mismatch = True
            else:
                context.audit_builder.observe(split, expected)
    _require_equal(count, artifact.record_count, f"{split} record count mismatch")
    if membership_mismatch:
        raise ValueError(f"{split} membership metadata does not match materialized records")


def _artifact_member_matches(
    actual: SplitMember,
    expected: SplitMember,
    context: _ArtifactValidationContext,
) -> bool:
    """Compare artifact-derived fields without copying expected provenance onto actual."""

    if not _identity_fields_match(actual, expected):
        return False
    if not _recorded_source_coordinate_matches(expected.source_coordinate, context.manifest):
        return False
    if context.source_authenticated:
        return True
    return expected.raw_line_sha256 == actual.raw_line_sha256


def _identity_fields_match(actual: SplitMember, expected: SplitMember) -> bool:
    return (
        actual.canonical_id,
        actual.lineage_id,
        actual.group_id,
        actual.content_sha256,
        actual.materialized_line_sha256,
    ) == (
        expected.canonical_id,
        expected.lineage_id,
        expected.group_id,
        expected.content_sha256,
        expected.materialized_line_sha256,
    )


def _recorded_source_coordinate_matches(coordinate: str, manifest: SplitManifest) -> bool:
    prefix = f"{manifest.source.path}:"
    if not coordinate.startswith(prefix):
        return False
    suffix = coordinate.removeprefix(prefix)
    if not suffix.isdigit() or suffix.startswith("0"):
        return False
    line = int(suffix)
    return 1 <= line <= manifest.source.record_count


@contextmanager
def verified_split_snapshot(
    manifest_path: Path,
    split: SplitName,
    artifact: SplitArtifact,
) -> Iterator[Path]:
    """Yield a digest-verified snapshot while pinning the declared artifact path."""

    path = resolve_split_path(manifest_path, artifact)
    try:
        descriptor = _open_split_descriptor(manifest_path.parent.resolve(), artifact.path)
    except NotImplementedError as exc:
        raise ValueError(
            f"{split} artifact cannot be validated safely on this platform: "
            "os.open(dir_fd=...) is unsupported"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"{split} artifact could not be opened without following a symlink"
        ) from exc
    try:
        initial = _artifact_identity(os.fstat(descriptor), split)
        _require_stable_artifact(path, descriptor, initial, split)
        try:
            snapshot_descriptor, snapshot_name = tempfile.mkstemp(
                prefix=f"agoge-{split}-snapshot-",
                suffix=".jsonl",
                dir=validation_staging_dir(manifest_path.parent),
            )
        except PermissionError as exc:
            raise unwritable_staging_error() from exc
        snapshot_path = Path(snapshot_name)
        try:
            digest = _copy_artifact_snapshot(descriptor, snapshot_descriptor)
            _require_stable_artifact(path, descriptor, initial, split)
            _require_equal(
                digest,
                artifact.sha256,
                f"{split} digest mismatch: expected {artifact.sha256}, found {digest}",
            )
            try:
                yield snapshot_path
            finally:
                _require_stable_artifact(path, descriptor, initial, split)
        finally:
            snapshot_path.unlink(missing_ok=True)
    finally:
        os.close(descriptor)


def _open_split_descriptor(root: Path, relative_path: str) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow | close_on_exec
    file_flags = os.O_RDONLY | no_follow | close_on_exec
    directory_descriptor = os.open(root, directory_flags)
    try:
        components = relative_path.split("/")
        for component in components[:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        return os.open(components[-1], file_flags, dir_fd=directory_descriptor)
    finally:
        os.close(directory_descriptor)


@dataclass(frozen=True)
class _ArtifactIdentity:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


def _artifact_identity(value: os.stat_result, split: SplitName) -> _ArtifactIdentity:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{split} artifact must be a regular file")
    return _ArtifactIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
        size=value.st_size,
        modified_ns=value.st_mtime_ns,
        changed_ns=value.st_ctime_ns,
    )


def _require_stable_artifact(
    path: Path,
    descriptor: int,
    initial: _ArtifactIdentity,
    split: SplitName,
) -> None:
    current_descriptor = _artifact_identity(os.fstat(descriptor), split)
    try:
        current_path = _artifact_identity(os.stat(path, follow_symlinks=False), split)
    except OSError as exc:
        raise ValueError(f"{split} artifact changed while it was being validated") from exc
    if current_descriptor != initial or current_path != initial:
        raise ValueError(f"{split} artifact changed while it was being validated")


def _copy_artifact_snapshot(descriptor: int, snapshot_descriptor: int) -> str:
    digest = hashlib.sha256()
    with (
        os.fdopen(os.dup(descriptor), "rb") as source,
        os.fdopen(snapshot_descriptor, "wb") as target,
    ):
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            target.write(chunk)
    return digest.hexdigest()


def resolve_split_path(manifest_path: Path, artifact: SplitArtifact) -> Path:
    root = manifest_path.parent.resolve()
    candidate = root / artifact.path
    resolved = candidate.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"split artifact escapes manifest directory: {artifact.path}")
    return candidate


def _require_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(message)
