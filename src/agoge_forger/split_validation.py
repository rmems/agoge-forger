"""Validation and frozen-record loaders for canonical split manifests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .split_materialize import SourceRecord, assign_records, leakage_audit, read_source_records
from .split_schema import (
    SPLIT_NAMES,
    SplitArtifact,
    SplitManifest,
    SplitMaterializationSpec,
    SplitMember,
    SplitName,
    sha256_file,
)


def load_split_manifest(manifest_path: str | Path) -> SplitManifest:
    path = Path(manifest_path).expanduser().resolve(strict=True)
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid split manifest JSON: {path}") from exc
    return SplitManifest.model_validate(content)


def validate_split_manifest(
    manifest_path: str | Path, *, source_path: str | Path | None = None
) -> SplitManifest:
    """Validate schema, source identity, artifacts, members, and leakage gates."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = load_split_manifest(path)
    if source_path is not None:
        _validate_source(manifest, Path(source_path).expanduser().resolve(strict=True))
    observed = {
        split: _validate_artifact(path, split, artifact, manifest)
        for split, artifact in manifest.splits.items()
    }
    audit = _audit_observed(observed)
    _require_equal(
        audit,
        manifest.leakage_audit,
        "stored leakage audit differs from recomputed audit",
    )
    return manifest


def _validate_source(manifest: SplitManifest, source: Path) -> None:
    actual_source_sha = sha256_file(source)
    _require_equal(
        actual_source_sha,
        manifest.source.sha256,
        f"source SHA-256 mismatch: expected {manifest.source.sha256}, found {actual_source_sha}",
    )
    records = read_source_records(source, manifest.canonical_identity)
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


def _validate_artifact(
    manifest_path: Path,
    split: SplitName,
    artifact: SplitArtifact,
    manifest: SplitManifest,
) -> list[SourceRecord]:
    artifact_path = resolve_split_path(manifest_path, artifact)
    _require_artifact_digest(split, artifact, artifact_path)
    records = read_source_records(artifact_path, manifest.canonical_identity)
    _require_equal(len(records), artifact.record_count, f"{split} record count mismatch")
    actual_members = _materialized_members(records, artifact.members)
    _require_equal(
        actual_members,
        list(artifact.members),
        f"{split} membership metadata does not match materialized records",
    )
    return [
        SourceRecord(row=record.row, raw_line=record.raw_line, member=artifact.members[index])
        for index, record in enumerate(records)
    ]


def _require_artifact_digest(split: SplitName, artifact: SplitArtifact, path: Path) -> None:
    actual_digest = sha256_file(path)
    _require_equal(
        actual_digest,
        artifact.sha256,
        f"{split} digest mismatch: expected {artifact.sha256}, found {actual_digest}",
    )


def _materialized_members(
    records: Sequence[SourceRecord], expected: Sequence[SplitMember]
) -> list[SplitMember]:
    return [
        record.member.model_copy(
            update={
                "source_coordinate": expected[index].source_coordinate,
                "raw_line_sha256": expected[index].raw_line_sha256,
            }
        )
        for index, record in enumerate(records)
    ]


def _audit_observed(observed: dict[SplitName, list[SourceRecord]]):
    flattened: list[SourceRecord] = []
    remapped: dict[SplitName, list[int]] = {name: [] for name in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        for record in observed[split]:
            remapped[split].append(len(flattened))
            flattened.append(record)
    return leakage_audit(remapped, flattened)


def resolve_split_path(manifest_path: Path, artifact: SplitArtifact) -> Path:
    root = manifest_path.parent.resolve()
    candidate = (root / artifact.path).resolve(strict=True)
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"split artifact escapes manifest directory: {artifact.path}")
    return candidate


def _require_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(message)
