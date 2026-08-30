"""Deterministic materialization for the canonical split schema."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .datasets import normalize_row
from .split_schema import (
    SPLIT_NAMES,
    CanonicalIdentityPolicy,
    LeakageAudit,
    SourceFile,
    SplitArtifact,
    SplitManifest,
    SplitMaterializationSpec,
    SplitMember,
    SplitName,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


@dataclass(frozen=True)
class SourceRecord:
    row: dict[str, Any]
    raw_line: bytes
    member: SplitMember


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def read_source_records(source_path: Path, identity: CanonicalIdentityPolicy) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    seen_ids: dict[str, str] = {}
    with source_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            coordinate = f"{source_path.name}:{line_number}"
            row = _decode_source_row(raw_line, coordinate)
            normalize_row(row, tokenizer=None, index=line_number)
            record = _build_source_record(row, raw_line, coordinate, identity)
            _reject_duplicate_identity(record.member, seen_ids)
            records.append(record)
    if not records:
        raise ValueError(f"source contains no JSONL records: {source_path}")
    return records


def _decode_source_row(raw_line: bytes, coordinate: str) -> dict[str, Any]:
    try:
        decoded = raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{coordinate}: source line is not UTF-8") from exc
    try:
        row = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{coordinate}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{coordinate}: source row must be a JSON object")  # noqa: TRY004
    return row


def _build_source_record(
    row: dict[str, Any],
    raw_line: bytes,
    coordinate: str,
    identity: CanonicalIdentityPolicy,
) -> SourceRecord:
    canonical_id = _required_string(row, identity.canonical_id_field, coordinate)
    lineage_id = _optional_string(row, identity.lineage_id_field, coordinate) or canonical_id
    group_id = _optional_string(row, identity.group_id_field, coordinate)
    content_row = _without_identity_fields(row, identity)
    materialized_line = canonical_json_bytes(row) + b"\n"
    member = SplitMember(
        canonical_id=canonical_id,
        lineage_id=lineage_id,
        group_id=group_id,
        source_coordinate=coordinate,
        content_sha256=sha256_bytes(canonical_json_bytes(content_row)),
        raw_line_sha256=sha256_bytes(raw_line),
        materialized_line_sha256=sha256_bytes(materialized_line),
    )
    return SourceRecord(row=row, raw_line=raw_line, member=member)


def _required_string(row: Mapping[str, Any], field: str, coordinate: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{coordinate}: required identity field '{field}' must be a string")
    return value.strip()


def _optional_string(row: Mapping[str, Any], field: str, coordinate: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{coordinate}: optional identity field '{field}' must be a string")
    return value.strip()


def _without_identity_fields(
    row: Mapping[str, Any], identity: CanonicalIdentityPolicy
) -> dict[str, Any]:
    content_row = dict(row)
    for field_name in (
        identity.canonical_id_field,
        identity.lineage_id_field,
        identity.group_id_field,
    ):
        content_row.pop(field_name, None)
    return content_row


def _reject_duplicate_identity(member: SplitMember, seen: dict[str, str]) -> None:
    previous = seen.get(member.canonical_id)
    if previous is not None:
        raise ValueError(
            f"duplicate canonical ID '{member.canonical_id}' at "
            f"{member.source_coordinate} and {previous}"
        )
    seen[member.canonical_id] = member.source_coordinate


def _atomic_components(records: Sequence[SourceRecord]) -> list[list[int]]:
    """Join records sharing lineage, declared group, or exact content."""

    union_find = _UnionFind(len(records))
    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for key in _component_keys(record.member):
            if key in seen:
                union_find.union(index, seen[key])
            else:
                seen[key] = index
    components: dict[int, list[int]] = {}
    for index in range(len(records)):
        components.setdefault(union_find.find(index), []).append(index)
    return list(components.values())


def _component_keys(member: SplitMember) -> tuple[tuple[str, str], ...]:
    keys = [
        ("lineage", member.lineage_id),
        ("content", member.content_sha256),
    ]
    if member.group_id is not None:
        keys.append(("group", member.group_id))
    return tuple(keys)


def _component_anchor(records: Sequence[SourceRecord], component: Sequence[int]) -> str:
    groups = sorted(
        group for index in component if (group := records[index].member.group_id) is not None
    )
    if groups:
        return f"group:{groups[0]}"
    lineages = sorted(records[index].member.lineage_id for index in component)
    return f"lineage:{lineages[0]}"


def _assign_components(
    records: Sequence[SourceRecord], spec: SplitMaterializationSpec
) -> dict[SplitName, list[int]]:
    policy = spec.split_policy
    components = _atomic_components(records)
    total_weight = sum(policy.weights.values())
    train_threshold = policy.weights["train"]
    validation_threshold = train_threshold + policy.weights["validation"]
    assigned: dict[SplitName, list[int]] = {name: [] for name in SPLIT_NAMES}
    for component in components:
        split = _bucket_split(
            _component_anchor(records, component),
            spec,
            total_weight,
            train_threshold,
            validation_threshold,
        )
        assigned[split].extend(component)
    _sort_and_require_nonempty(assigned, records)
    return assigned


def _bucket_split(
    anchor: str,
    spec: SplitMaterializationSpec,
    total_weight: int,
    train_threshold: int,
    validation_threshold: int,
) -> SplitName:
    policy = spec.split_policy
    material = f"{policy.algorithm_version}\0{policy.seed}\0{policy.salt}\0{anchor}".encode()
    bucket = int(sha256_bytes(material), 16) % total_weight
    if bucket < train_threshold:
        return "train"
    if bucket < validation_threshold:
        return "validation"
    return "held_out"


def _sort_and_require_nonempty(
    assigned: dict[SplitName, list[int]], records: Sequence[SourceRecord]
) -> None:
    for split in SPLIT_NAMES:
        assigned[split].sort(key=lambda index: records[index].member.canonical_id)
        if not assigned[split]:
            raise ValueError(
                f"split '{split}' is empty; use a larger curated source or "
                "a separately versioned salt"
            )


def leakage_audit(
    assignments: Mapping[SplitName, Sequence[int]], records: Sequence[SourceRecord]
) -> LeakageAudit:
    counts = {
        "exact_content_cross_split": _cross_split_count(
            assignments, records, lambda record: record.member.content_sha256
        ),
        "canonical_id_cross_split": _cross_split_count(
            assignments, records, lambda record: record.member.canonical_id
        ),
        "source_coordinate_cross_split": _cross_split_count(
            assignments, records, lambda record: record.member.source_coordinate
        ),
        "lineage_cross_split": _cross_split_count(
            assignments, records, lambda record: record.member.lineage_id
        ),
        "declared_group_cross_split": _cross_split_count(
            assignments, records, lambda record: record.member.group_id
        ),
    }
    if any(counts.values()):
        raise ValueError(f"deterministic leakage audit failed: {counts}")
    return LeakageAudit(
        exact_content_cross_split=counts["exact_content_cross_split"],
        canonical_id_cross_split=counts["canonical_id_cross_split"],
        source_coordinate_cross_split=counts["source_coordinate_cross_split"],
        lineage_cross_split=counts["lineage_cross_split"],
        declared_group_cross_split=counts["declared_group_cross_split"],
        deterministic_guarantees=(
            "canonical JSON content hashes do not cross splits",
            "canonical IDs do not cross splits",
            "source coordinates do not cross splits",
            "lineage IDs do not cross splits",
            "declared group IDs do not cross splits",
        ),
    )


def _cross_split_count(
    assignments: Mapping[SplitName, Sequence[int]],
    records: Sequence[SourceRecord],
    key: Callable[[SourceRecord], str | None],
) -> int:
    owners: dict[str, SplitName] = {}
    collisions: set[str] = set()
    for split, indexes in assignments.items():
        for index in indexes:
            value = key(records[index])
            if value is None:
                continue
            previous = owners.setdefault(value, split)
            if previous != split:
                collisions.add(value)
    return len(collisions)


def materialize_split(
    source_path: str | Path,
    output_dir: str | Path,
    spec: SplitMaterializationSpec,
) -> SplitManifest:
    """Materialize one immutable, deterministic three-way source-level split."""

    source = Path(source_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser()
    _validate_materialization_paths(source, destination)
    records = read_source_records(source, spec.canonical_identity)
    assignments = _assign_components(records, spec)
    artifacts, payloads = _build_artifacts(assignments, records)
    manifest = _build_manifest(
        source, spec, records, artifacts, leakage_audit(assignments, records)
    )
    _write_snapshot(destination, manifest, payloads)
    return manifest


def _validate_materialization_paths(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"refusing silent regeneration because output path already exists: {destination}"
        )
    if not source.is_file():
        raise ValueError(f"source must be a file: {source}")


def _build_artifacts(
    assignments: Mapping[SplitName, Sequence[int]], records: Sequence[SourceRecord]
) -> tuple[dict[SplitName, SplitArtifact], dict[SplitName, bytes]]:
    artifacts: dict[SplitName, SplitArtifact] = {}
    payloads: dict[SplitName, bytes] = {}
    for split, indexes in assignments.items():
        payload = b"".join(canonical_json_bytes(records[index].row) + b"\n" for index in indexes)
        payloads[split] = payload
        artifacts[split] = SplitArtifact(
            path=f"splits/{split}.jsonl",
            sha256=sha256_bytes(payload),
            record_count=len(indexes),
            members=tuple(records[index].member for index in indexes),
        )
    return artifacts, payloads


def _build_manifest(
    source: Path,
    spec: SplitMaterializationSpec,
    records: Sequence[SourceRecord],
    artifacts: dict[SplitName, SplitArtifact],
    audit: LeakageAudit,
) -> SplitManifest:
    return SplitManifest(
        source=SourceFile(
            repository=spec.source_repository,
            revision=spec.source_revision,
            dataset_version=spec.dataset_version,
            path=source.name,
            sha256=sha256_file(source),
            record_count=len(records),
        ),
        canonical_identity=spec.canonical_identity,
        split_policy=spec.split_policy,
        splits=artifacts,
        leakage_audit=audit,
        limitations=(
            "Deterministic gates do not claim semantic near-duplicate detection.",
            "Tokenizer and rendering statistics are immutable derivative sidecars.",
        ),
    )


def _write_snapshot(
    destination: Path,
    manifest: SplitManifest,
    payloads: Mapping[SplitName, bytes],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for split, payload in payloads.items():
        exclusive_write(destination / manifest.splits[split].path, payload)
    exclusive_write(destination / "split_manifest.json", _manifest_bytes(manifest))
    exclusive_write(destination / "split_report.md", _render_report(manifest).encode("utf-8"))


def exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}") from exc


def _manifest_bytes(manifest: SplitManifest) -> bytes:
    return canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"


def _render_report(manifest: SplitManifest) -> str:
    lines = _report_header(manifest)
    for split in SPLIT_NAMES:
        artifact = manifest.splits[split]
        lines.append(f"| {split} | {artifact.record_count} | `{artifact.sha256}` |")
    lines.extend(_report_footer(manifest))
    return "\n".join(lines)


def _report_header(manifest: SplitManifest) -> list[str]:
    return [
        "# Frozen split report",
        "",
        f"- Source: `{manifest.source.repository}@{manifest.source.revision}`",
        f"- Dataset version: `{manifest.source.dataset_version}`",
        f"- Source file: `{manifest.source.path}` (`{manifest.source.sha256}`)",
        f"- Source coverage: {manifest.source.record_count}/{manifest.source.record_count} records",
        f"- Split algorithm: `{manifest.split_policy.algorithm_version}`",
        f"- Seed/salt: `{manifest.split_policy.seed}` / `{manifest.split_policy.salt}`",
        "",
        "## Partitions",
        "",
        "| Split | Records | Source-level SHA-256 |",
        "|---|---:|---|",
    ]


def _report_footer(manifest: SplitManifest) -> list[str]:
    return [
        "",
        "## Leakage guarantees",
        "",
        *[f"- {item}" for item in manifest.leakage_audit.deterministic_guarantees],
        "",
        "## Exclusions",
        "",
        "- None. Every valid source record is materialized exactly once.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in manifest.limitations],
        "",
    ]
