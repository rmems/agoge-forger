"""Deterministic materialization for the canonical split schema."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._source_snapshot import copy_source_snapshot
from ._split_report import manifest_bytes, render_report
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
    validate_repository_relative_path,
)


@dataclass(frozen=True)
class SourceRecord:
    row: dict[str, Any]
    raw_line: bytes
    member: SplitMember


@dataclass(frozen=True)
class _SplitWritePlan:
    assignments: Mapping[SplitName, Sequence[int]]
    records: Sequence[SourceRecord]
    payloads: io.BufferedRandom
    payload_spans: Sequence[tuple[int, int]]


@dataclass
class TrainingRepresentationTracker:
    """Track one model-independent training representation across streams."""

    first: tuple[str, str] | None = None

    def observe(self, row: Mapping[str, Any], coordinate: str) -> None:
        representation = _training_representation(row, coordinate)
        if self.first is None:
            self.first = (representation, coordinate)
            return
        if representation != self.first[0]:
            raise ValueError(
                "source mixes model-dependent training representations: "
                f"{self.first[0]} at {self.first[1]} and "
                f"{representation} at {coordinate}; freeze one representation "
                "per split snapshot"
            )


@dataclass(frozen=True)
class _SourceLine:
    row: dict[str, Any]
    raw_line: bytes
    coordinate: str
    training_payload: Mapping[str, Any]


@dataclass(frozen=True)
class _BucketPolicy:
    spec: SplitMaterializationSpec
    total_weight: int
    train_threshold: int
    validation_threshold: int


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


def iter_source_records(
    source_path: Path,
    identity: CanonicalIdentityPolicy,
    *,
    source_coordinate_path: str,
    representation_tracker: TrainingRepresentationTracker | None = None,
) -> Iterator[SourceRecord]:
    """Yield source records while enforcing complete-source invariants."""

    coordinate_path = validate_repository_relative_path(source_coordinate_path)
    seen_ids: dict[str, str] = {}
    tracker = representation_tracker or TrainingRepresentationTracker()
    record_count = 0
    with source_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            coordinate = f"{coordinate_path}:{line_number}"
            row = _decode_source_row(raw_line, coordinate)
            record = _build_source_record(
                _SourceLine(
                    row=row,
                    raw_line=raw_line,
                    coordinate=coordinate,
                    training_payload=_content_hash_payload(row, identity, line_number),
                ),
                identity,
            )
            _reject_duplicate_identity(record.member, seen_ids)
            if identity.content_hash_policy == "normalized-training-payload-v1":
                tracker.observe(row, coordinate)
            record_count += 1
            yield record
    if record_count == 0:
        raise ValueError(f"source contains no JSONL records: {source_path}")


def read_source_records(
    source_path: Path,
    identity: CanonicalIdentityPolicy,
    *,
    source_coordinate_path: str,
) -> list[SourceRecord]:
    """Materialize :func:`iter_source_records` for assignment and validation."""

    return list(
        iter_source_records(
            source_path,
            identity,
            source_coordinate_path=source_coordinate_path,
        )
    )


def _training_representation(row: Mapping[str, Any], coordinate: str) -> str:
    representations = tuple(
        field_name for field_name in ("text", "messages", "instruction") if field_name in row
    )
    if len(representations) > 1:
        rendered = ", ".join(representations)
        raise ValueError(
            f"{coordinate}: source row declares multiple training representations: {rendered}"
        )
    if representations:
        representation = representations[0]
        if representation != "text":
            raise ValueError(
                f"{coordinate}: new split snapshots require pre-rendered 'text'; "
                f"'{representation}' depends on downstream rendering"
            )
        return representation
    # Keep normalize_row's existing missing-payload diagnostic as the canonical error.
    return "unknown"


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
    source: _SourceLine,
    identity: CanonicalIdentityPolicy,
) -> SourceRecord:
    row = source.row
    coordinate = source.coordinate
    canonical_id = _required_string(row, identity.canonical_id_field, coordinate)
    lineage_id = _optional_string(row, identity.lineage_id_field, coordinate) or canonical_id
    group_id = _optional_string(row, identity.group_id_field, coordinate)
    materialized_line = canonical_json_bytes(row) + b"\n"
    member = SplitMember(
        canonical_id=canonical_id,
        lineage_id=lineage_id,
        group_id=group_id,
        source_coordinate=coordinate,
        content_sha256=sha256_bytes(canonical_json_bytes(source.training_payload)),
        raw_line_sha256=sha256_bytes(source.raw_line),
        materialized_line_sha256=sha256_bytes(materialized_line),
    )
    return SourceRecord(row=row, raw_line=source.raw_line, member=member)


def _content_hash_payload(
    row: dict[str, Any], identity: CanonicalIdentityPolicy, line_number: int
) -> Mapping[str, Any]:
    """Return the versioned payload used for exact-content grouping.

    Ancillary source metadata remains covered by the raw and materialized line
    digests, but it must not let identical training examples cross partitions.
    The legacy branch exists only to validate already-frozen v1 manifests.
    """

    if identity.content_hash_policy == "canonical-json-excluding-identity-fields":
        return _without_identity_fields(row, identity)
    normalized = normalize_row(row, tokenizer=None, index=line_number)
    return {"text": normalized["text"]}


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


def _required_string(row: Mapping[str, Any], field_name: str, coordinate: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{coordinate}: required identity field '{field_name}' must be a string")
    return _canonical_identity_string(value, field_name, coordinate)


def _optional_string(row: Mapping[str, Any], field_name: str, coordinate: str) -> str | None:
    value = row.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{coordinate}: optional identity field '{field_name}' must be a string")
    return _canonical_identity_string(value, field_name, coordinate)


def _canonical_identity_string(value: str, field_name: str, coordinate: str) -> str:
    if value != value.strip():
        raise ValueError(
            f"{coordinate}: identity field '{field_name}' cannot contain surrounding whitespace"
        )
    return value


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


def assign_records(
    records: Sequence[SourceRecord], spec: SplitMaterializationSpec
) -> dict[SplitName, list[int]]:
    policy = spec.split_policy
    components = _atomic_components(records)
    total_weight = sum(policy.weights.values())
    train_threshold = policy.weights["train"]
    validation_threshold = train_threshold + policy.weights["validation"]
    bucket_policy = _BucketPolicy(
        spec,
        total_weight,
        train_threshold,
        validation_threshold,
    )
    assigned: dict[SplitName, list[int]] = {name: [] for name in SPLIT_NAMES}
    for component in components:
        split = _bucket_split(_component_anchor(records, component), bucket_policy)
        assigned[split].extend(component)
    _sort_and_require_nonempty(assigned, records)
    return assigned


def _bucket_split(anchor: str, bucket_policy: _BucketPolicy) -> SplitName:
    policy = bucket_policy.spec.split_policy
    material = f"{policy.algorithm_version}\0{policy.seed}\0{policy.salt}\0{anchor}".encode()
    bucket = int(sha256_bytes(material), 16) % bucket_policy.total_weight
    if bucket < bucket_policy.train_threshold:
        return "train"
    if bucket < bucket_policy.validation_threshold:
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
    builder = LeakageAuditBuilder()
    for split, indexes in assignments.items():
        for index in indexes:
            builder.observe(split, records[index].member)
    return builder.result()


@dataclass
class LeakageAuditBuilder:
    """Incrementally compute the canonical deterministic leakage audit."""

    owners: dict[str, dict[str, SplitName]] = field(
        default_factory=lambda: {
            "content": {},
            "canonical": {},
            "coordinate": {},
            "lineage": {},
            "group": {},
        }
    )
    collisions: dict[str, set[str]] = field(
        default_factory=lambda: {
            "content": set(),
            "canonical": set(),
            "coordinate": set(),
            "lineage": set(),
            "group": set(),
        }
    )

    def observe(self, split: SplitName, member: SplitMember) -> None:
        values = {
            "content": member.content_sha256,
            "canonical": member.canonical_id,
            "coordinate": member.source_coordinate,
            "lineage": member.lineage_id,
            "group": member.group_id,
        }
        for kind, value in values.items():
            if value is None:
                continue
            previous = self.owners[kind].setdefault(value, split)
            if previous != split:
                self.collisions[kind].add(value)

    def result(self) -> LeakageAudit:
        counts = {
            "exact_content_cross_split": len(self.collisions["content"]),
            "canonical_id_cross_split": len(self.collisions["canonical"]),
            "source_coordinate_cross_split": len(self.collisions["coordinate"]),
            "lineage_cross_split": len(self.collisions["lineage"]),
            "declared_group_cross_split": len(self.collisions["group"]),
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


def materialize_split(
    source_path: str | Path,
    output_dir: str | Path,
    spec: SplitMaterializationSpec,
) -> SplitManifest:
    """Materialize one immutable, deterministic three-way source-level split."""

    source = Path(source_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser()
    if spec.canonical_identity.content_hash_policy != "normalized-training-payload-v1":
        raise ValueError(
            "new split materializations require normalized-training-payload-v1 "
            "exact-content hashing"
        )
    _validate_materialization_paths(source, destination)
    with tempfile.TemporaryDirectory(prefix="agoge-source-snapshot-") as staging_dir:
        staging = Path(staging_dir)
        source_snapshot = staging / "source.jsonl"
        source_sha256 = copy_source_snapshot(source, source_snapshot)
        with (staging / "canonical-payloads.bin").open("w+b") as payloads:
            records, payload_spans = _stage_source_records(source_snapshot, spec, payloads)
            assignments = assign_records(records, spec)
            destination.mkdir(parents=True, exist_ok=False)
            artifacts = _write_split_artifacts(
                destination,
                _SplitWritePlan(assignments, records, payloads, payload_spans),
            )
            manifest = _build_manifest(
                _source_file(source_sha256, spec, len(records)),
                spec,
                artifacts,
                leakage_audit(assignments, records),
            )
            _write_snapshot_metadata(destination, manifest)
    return manifest


def _validate_materialization_paths(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"refusing silent regeneration because output path already exists: {destination}"
        )
    if not source.is_file():
        raise ValueError(f"source must be a file: {source}")


def _stage_source_records(
    source_snapshot: Path,
    spec: SplitMaterializationSpec,
    payloads: io.BufferedRandom,
) -> tuple[list[SourceRecord], list[tuple[int, int]]]:
    records: list[SourceRecord] = []
    spans: list[tuple[int, int]] = []
    for record in iter_source_records(
        source_snapshot,
        spec.canonical_identity,
        source_coordinate_path=spec.source_path,
    ):
        line = canonical_json_bytes(record.row) + b"\n"
        offset = payloads.tell()
        payloads.write(line)
        spans.append((offset, len(line)))
        records.append(
            SourceRecord(
                row={},
                raw_line=b"",
                member=record.member,
            )
        )
    payloads.flush()
    return records, spans


def _write_split_artifacts(
    destination: Path,
    plan: _SplitWritePlan,
) -> dict[SplitName, SplitArtifact]:
    artifacts: dict[SplitName, SplitArtifact] = {}
    for split, indexes in plan.assignments.items():
        relative_path = f"splits/{split}.jsonl"
        artifact_path = destination / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        try:
            with artifact_path.open("xb") as handle:
                for index in indexes:
                    offset, length = plan.payload_spans[index]
                    plan.payloads.seek(offset)
                    line = plan.payloads.read(length)
                    if len(line) != length:
                        raise OSError("staged canonical split payload was truncated")
                    digest.update(line)
                    handle.write(line)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite frozen artifact: {artifact_path}"
            ) from exc
        artifacts[split] = SplitArtifact(
            path=relative_path,
            sha256=digest.hexdigest(),
            record_count=len(indexes),
            members=tuple(plan.records[index].member for index in indexes),
        )
    return artifacts


def _source_file(
    source_sha256: str,
    spec: SplitMaterializationSpec,
    record_count: int,
) -> SourceFile:
    return SourceFile(
        repository=spec.source_repository,
        revision=spec.source_revision,
        dataset_version=spec.dataset_version,
        path=spec.source_path,
        sha256=source_sha256,
        record_count=record_count,
    )


def _build_manifest(
    source: SourceFile,
    spec: SplitMaterializationSpec,
    artifacts: dict[SplitName, SplitArtifact],
    audit: LeakageAudit,
) -> SplitManifest:
    return SplitManifest(
        source=source,
        canonical_identity=spec.canonical_identity,
        split_policy=spec.split_policy,
        splits=artifacts,
        leakage_audit=audit,
        limitations=(
            "Deterministic gates do not claim semantic near-duplicate detection.",
            "Tokenizer and rendering statistics are immutable derivative sidecars.",
        ),
    )


def _write_snapshot_metadata(
    destination: Path,
    manifest: SplitManifest,
) -> None:
    exclusive_write(destination / "split_manifest.json", manifest_bytes(manifest))
    exclusive_write(destination / "split_report.md", render_report(manifest).encode("utf-8"))


def exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}") from exc
