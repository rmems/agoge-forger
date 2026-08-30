"""Validation and frozen-record loaders for canonical split manifests."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset  # type: ignore[attr-defined]

from .datasets import normalize_row
from .split_materialize import SourceRecord, leakage_audit, read_source_records
from .split_schema import (
    SPLIT_NAMES,
    SplitArtifact,
    SplitManifest,
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
    if audit != manifest.leakage_audit:
        raise ValueError("stored leakage audit differs from recomputed audit")
    return manifest


def _validate_source(manifest: SplitManifest, source: Path) -> None:
    actual_source_sha = sha256_file(source)
    if actual_source_sha != manifest.source.sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {manifest.source.sha256}, found {actual_source_sha}"
        )
    records = read_source_records(source, manifest.canonical_identity)
    source_members = {record.member.canonical_id: record.member for record in records}
    manifest_members = {
        member.canonical_id: member
        for artifact in manifest.splits.values()
        for member in artifact.members
    }
    if source_members != manifest_members:
        raise ValueError("manifest membership metadata differs from the pinned source")


def _validate_artifact(
    manifest_path: Path,
    split: SplitName,
    artifact: SplitArtifact,
    manifest: SplitManifest,
) -> list[SourceRecord]:
    artifact_path = resolve_split_path(manifest_path, artifact)
    _require_artifact_digest(split, artifact, artifact_path)
    records = read_source_records(artifact_path, manifest.canonical_identity)
    if len(records) != artifact.record_count:
        raise ValueError(f"{split} record count mismatch")
    actual_members = _materialized_members(records, artifact.members)
    if actual_members != list(artifact.members):
        raise ValueError(f"{split} membership metadata does not match materialized records")
    return [
        SourceRecord(row=record.row, raw_line=record.raw_line, member=artifact.members[index])
        for index, record in enumerate(records)
    ]


def _require_artifact_digest(split: SplitName, artifact: SplitArtifact, path: Path) -> None:
    actual_digest = sha256_file(path)
    if actual_digest != artifact.sha256:
        raise ValueError(
            f"{split} digest mismatch: expected {artifact.sha256}, found {actual_digest}"
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


def iter_materialized_records(
    path: Path, manifest: SplitManifest, split: SplitName
) -> Iterator[dict[str, Any]]:
    artifact_path = resolve_split_path(path, manifest.splits[split])
    with artifact_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"materialized {split} row is not an object")
            yield row


def iter_frozen_records(manifest_path: str | Path, split: SplitName) -> Iterator[dict[str, Any]]:
    """Yield raw source records from a verified frozen partition."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = validate_split_manifest(path)
    yield from iter_materialized_records(path, manifest, split)


def load_frozen_dataset(
    manifest_path: str | Path, split: SplitName, tokenizer: Any = None
) -> Dataset:
    """Training loader for a frozen split; it never derives new membership."""

    def generate() -> Iterator[dict[str, Any]]:
        for index, row in enumerate(iter_frozen_records(manifest_path, split), 1):
            yield normalize_row(row, tokenizer=tokenizer, index=index)

    return Dataset.from_generator(generate)
