"""Immutable source-level split contracts for measured SFT experiments.

The split manifest is the single authoritative schema for partition identity.
Rendering and tokenization are deliberately derivative operations: they may
change model inputs, but never canonical task identity or split membership.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datasets import Dataset  # type: ignore[attr-defined]

from .datasets import normalize_row

SPLIT_MANIFEST_VERSION: Literal["agoge.split-manifest.v1"] = "agoge.split-manifest.v1"
SPLIT_ALGORITHM_VERSION: Literal["sha256-atomic-bucket-v1"] = "sha256-atomic-bucket-v1"
TOKEN_STATS_VERSION: Literal["agoge.token-stats.v1"] = "agoge.token-stats.v1"

SplitName = Literal["train", "validation", "held_out"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "held_out")


class FrozenModel(BaseModel):
    """Strict base class for versioned contract records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceFile(FrozenModel):
    repository: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    dataset_version: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=1)


class CanonicalIdentityPolicy(FrozenModel):
    canonical_id_field: str = Field(min_length=1)
    lineage_id_field: str = Field(min_length=1)
    group_id_field: str = Field(min_length=1)
    missing_lineage: Literal["canonical_id"] = "canonical_id"
    content_hash_policy: Literal["canonical-json-excluding-identity-fields"] = (
        "canonical-json-excluding-identity-fields"
    )
    source_coordinate_policy: Literal["source-path-plus-one-based-line"] = (
        "source-path-plus-one-based-line"
    )


class SplitPolicy(FrozenModel):
    algorithm_version: Literal["sha256-atomic-bucket-v1"] = SPLIT_ALGORITHM_VERSION
    seed: int
    salt: str = Field(min_length=1)
    weights: dict[SplitName, int]

    @model_validator(mode="after")
    def validate_weights(self) -> SplitPolicy:
        if set(self.weights) != set(SPLIT_NAMES):
            raise ValueError(f"weights must contain exactly {SPLIT_NAMES}")
        if any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("all split weights must be positive")
        return self


class SplitMember(FrozenModel):
    canonical_id: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    group_id: str | None = None
    source_coordinate: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_line_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialized_line_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SplitArtifact(FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=1)
    members: tuple[SplitMember, ...]

    @model_validator(mode="after")
    def count_members(self) -> SplitArtifact:
        if self.record_count != len(self.members):
            raise ValueError("split record_count does not match members")
        return self


class LeakageAudit(FrozenModel):
    status: Literal["passed"] = "passed"
    exact_content_cross_split: int = 0
    canonical_id_cross_split: int = 0
    source_coordinate_cross_split: int = 0
    lineage_cross_split: int = 0
    declared_group_cross_split: int = 0
    deterministic_guarantees: tuple[str, ...]

    @model_validator(mode="after")
    def fail_closed(self) -> LeakageAudit:
        counts = (
            self.exact_content_cross_split,
            self.canonical_id_cross_split,
            self.source_coordinate_cross_split,
            self.lineage_cross_split,
            self.declared_group_cross_split,
        )
        if any(count != 0 for count in counts):
            raise ValueError("a passed leakage audit cannot contain cross-split collisions")
        return self


class TokenStatisticsPolicy(FrozenModel):
    artifact_schema_version: Literal["agoge.token-stats.v1"] = TOKEN_STATS_VERSION
    storage: Literal["immutable-sidecar"] = "immutable-sidecar"
    invariant: Literal["must-not-modify-source-split-digests"] = (
        "must-not-modify-source-split-digests"
    )
    required_keys: tuple[str, ...] = (
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "serializer_id",
        "serializer_version",
        "serializer_sha256",
    )


class SplitManifest(FrozenModel):
    """Canonical, versioned source-level partition manifest."""

    schema_version: Literal["agoge.split-manifest.v1"] = SPLIT_MANIFEST_VERSION
    source: SourceFile
    canonical_identity: CanonicalIdentityPolicy
    split_policy: SplitPolicy
    splits: dict[SplitName, SplitArtifact]
    leakage_audit: LeakageAudit
    token_statistics: TokenStatisticsPolicy = Field(default_factory=TokenStatisticsPolicy)
    exclusions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_split_names_and_total(self) -> SplitManifest:
        if set(self.splits) != set(SPLIT_NAMES):
            raise ValueError(f"splits must contain exactly {SPLIT_NAMES}")
        total = sum(split.record_count for split in self.splits.values())
        if total != self.source.record_count:
            raise ValueError("split counts do not equal the source record count")
        members = [member for split in self.splits.values() for member in split.members]
        canonical_ids = [member.canonical_id for member in members]
        source_coordinates = [member.source_coordinate for member in members]
        if len(canonical_ids) != len(set(canonical_ids)):
            raise ValueError("canonical IDs must be globally unique across the manifest")
        if len(source_coordinates) != len(set(source_coordinates)):
            raise ValueError("source coordinates must be globally unique across the manifest")
        return self


class TokenizerLike(Protocol):
    def __call__(self, text: str) -> Any: ...


Serializer = Callable[[Mapping[str, Any]], str]


class TokenStatSplit(FrozenModel):
    record_count: int = Field(ge=1)
    total_tokens: int = Field(ge=0)
    minimum_tokens: int = Field(ge=0)
    maximum_tokens: int = Field(ge=0)
    truncated_records: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> TokenStatSplit:
        if self.minimum_tokens > self.maximum_tokens:
            raise ValueError("minimum_tokens cannot exceed maximum_tokens")
        if self.truncated_records > self.record_count:
            raise ValueError("truncated_records cannot exceed record_count")
        return self


class TokenStatistics(FrozenModel):
    schema_version: Literal["agoge.token-stats.v1"] = TOKEN_STATS_VERSION
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_split_sha256: dict[SplitName, str]
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_limit: int | None = Field(default=None, ge=1)
    splits: dict[SplitName, TokenStatSplit]

    @model_validator(mode="after")
    def validate_split_identity(self) -> TokenStatistics:
        if set(self.source_split_sha256) != set(SPLIT_NAMES):
            raise ValueError(f"source_split_sha256 must contain exactly {SPLIT_NAMES}")
        if set(self.splits) != set(SPLIT_NAMES):
            raise ValueError(f"splits must contain exactly {SPLIT_NAMES}")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.source_split_sha256.values()
        ):
            raise ValueError("source split digests must be lowercase SHA-256 values")
        return self


@dataclass(frozen=True)
class _SourceRecord:
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


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's canonical JSON encoding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_source_records(
    source_path: Path, identity: CanonicalIdentityPolicy
) -> list[_SourceRecord]:
    records: list[_SourceRecord] = []
    seen_ids: dict[str, str] = {}
    source_name = source_path.name

    with source_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            coordinate = f"{source_name}:{line_number}"
            try:
                decoded = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{coordinate}: source line is not UTF-8") from exc
            try:
                row = json.loads(decoded)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{coordinate}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(  # noqa: TRY004
                    f"{coordinate}: source row must be a JSON object"
                )
            # Validate the existing Agoge SFT row contract without rendering it.
            # The original source object remains the canonical stored record.
            normalize_row(row, tokenizer=None, index=line_number)

            canonical_id = _required_string(row, identity.canonical_id_field, coordinate)
            if canonical_id in seen_ids:
                raise ValueError(
                    f"duplicate canonical ID '{canonical_id}' at {coordinate} and "
                    f"{seen_ids[canonical_id]}"
                )
            seen_ids[canonical_id] = coordinate
            lineage_id = (
                _optional_string(row, identity.lineage_id_field, coordinate) or canonical_id
            )
            group_id = _optional_string(row, identity.group_id_field, coordinate)

            content_row = dict(row)
            for field_name in (
                identity.canonical_id_field,
                identity.lineage_id_field,
                identity.group_id_field,
            ):
                content_row.pop(field_name, None)

            member = SplitMember(
                canonical_id=canonical_id,
                lineage_id=lineage_id,
                group_id=group_id,
                source_coordinate=coordinate,
                content_sha256=sha256_bytes(canonical_json_bytes(content_row)),
                raw_line_sha256=sha256_bytes(raw_line),
                materialized_line_sha256=sha256_bytes(canonical_json_bytes(row) + b"\n"),
            )
            records.append(_SourceRecord(row=row, raw_line=raw_line, member=member))

    if not records:
        raise ValueError(f"source contains no JSONL records: {source_path}")
    return records


def _atomic_components(records: Sequence[_SourceRecord]) -> list[list[int]]:
    """Join records sharing lineage, declared group, or exact content."""

    union_find = _UnionFind(len(records))
    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        keys = [
            ("lineage", record.member.lineage_id),
            ("content", record.member.content_sha256),
        ]
        if record.member.group_id is not None:
            keys.append(("group", record.member.group_id))
        for key in keys:
            if key in seen:
                union_find.union(index, seen[key])
            else:
                seen[key] = index

    components: dict[int, list[int]] = {}
    for index in range(len(records)):
        components.setdefault(union_find.find(index), []).append(index)
    return list(components.values())


def _component_anchor(records: Sequence[_SourceRecord], component: Sequence[int]) -> str:
    groups = sorted(
        record.member.group_id
        for index in component
        if (record := records[index]).member.group_id is not None
    )
    if groups:
        return f"group:{groups[0]}"
    lineages = sorted(records[index].member.lineage_id for index in component)
    if lineages:
        return f"lineage:{lineages[0]}"
    return f"sample:{min(records[index].member.canonical_id for index in component)}"


def _assign_components(
    records: Sequence[_SourceRecord], components: Sequence[Sequence[int]], policy: SplitPolicy
) -> dict[SplitName, list[int]]:
    total_weight = sum(policy.weights.values())
    thresholds = (
        policy.weights["train"],
        policy.weights["train"] + policy.weights["validation"],
    )
    assigned: dict[SplitName, list[int]] = {name: [] for name in SPLIT_NAMES}

    for component in components:
        anchor = _component_anchor(records, component)
        assignment_material = (
            f"{policy.algorithm_version}\0{policy.seed}\0{policy.salt}\0{anchor}".encode()
        )
        bucket = int(sha256_bytes(assignment_material), 16) % total_weight
        if bucket < thresholds[0]:
            split: SplitName = "train"
        elif bucket < thresholds[1]:
            split = "validation"
        else:
            split = "held_out"
        assigned[split].extend(component)

    for split in SPLIT_NAMES:
        assigned[split].sort(key=lambda index: records[index].member.canonical_id)
        if not assigned[split]:
            raise ValueError(
                f"split '{split}' is empty under {policy.algorithm_version}; "
                "use a larger curated source or a separately versioned salt"
            )
    return assigned


def _cross_split_count(
    assignments: Mapping[SplitName, Sequence[int]],
    records: Sequence[_SourceRecord],
    key: Callable[[_SourceRecord], str | None],
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


def _leakage_audit(
    assignments: Mapping[SplitName, Sequence[int]], records: Sequence[_SourceRecord]
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


def _exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}") from exc


def _manifest_bytes(manifest: SplitManifest) -> bytes:
    return canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"


def materialize_split(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    source_repository: str,
    source_revision: str,
    dataset_version: str,
    seed: int,
    salt: str,
    train_weight: int = 80,
    validation_weight: int = 10,
    held_out_weight: int = 10,
    canonical_id_field: str = "canonical_id",
    lineage_id_field: str = "lineage_id",
    group_id_field: str = "group_id",
) -> SplitManifest:
    """Materialize one immutable, deterministic three-way source-level split."""

    source = Path(source_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise FileExistsError(
            f"refusing silent regeneration because output path already exists: {destination}"
        )
    if not source.is_file():
        raise ValueError(f"source must be a file: {source}")

    identity = CanonicalIdentityPolicy(
        canonical_id_field=canonical_id_field,
        lineage_id_field=lineage_id_field,
        group_id_field=group_id_field,
    )
    policy = SplitPolicy(
        seed=seed,
        salt=salt,
        weights={
            "train": train_weight,
            "validation": validation_weight,
            "held_out": held_out_weight,
        },
    )
    records = _read_source_records(source, identity)
    assignments = _assign_components(records, _atomic_components(records), policy)
    audit = _leakage_audit(assignments, records)

    split_payloads: dict[SplitName, bytes] = {}
    split_artifacts: dict[SplitName, SplitArtifact] = {}
    for split, indexes in assignments.items():
        payload = b"".join(canonical_json_bytes(records[index].row) + b"\n" for index in indexes)
        split_payloads[split] = payload
        split_artifacts[split] = SplitArtifact(
            path=f"splits/{split}.jsonl",
            sha256=sha256_bytes(payload),
            record_count=len(indexes),
            members=tuple(records[index].member for index in indexes),
        )

    manifest = SplitManifest(
        source=SourceFile(
            repository=source_repository,
            revision=source_revision,
            dataset_version=dataset_version,
            path=source.name,
            sha256=sha256_file(source),
            record_count=len(records),
        ),
        canonical_identity=identity,
        split_policy=policy,
        splits=split_artifacts,
        leakage_audit=audit,
        limitations=(
            "Deterministic gates do not claim semantic near-duplicate detection.",
            "Tokenizer and rendering statistics are immutable derivative sidecars.",
        ),
    )

    destination.mkdir(parents=True, exist_ok=False)
    for split, payload in split_payloads.items():
        _exclusive_write(destination / manifest.splits[split].path, payload)
    _exclusive_write(destination / "split_manifest.json", _manifest_bytes(manifest))
    _exclusive_write(destination / "split_report.md", _render_report(manifest).encode("utf-8"))
    return manifest


def _render_report(manifest: SplitManifest) -> str:
    lines = [
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
    for split in SPLIT_NAMES:
        artifact = manifest.splits[split]
        lines.append(f"| {split} | {artifact.record_count} | `{artifact.sha256}` |")
    lines.extend(
        [
            "",
            "## Leakage guarantees",
            "",
            *[f"- {guarantee}" for guarantee in manifest.leakage_audit.deterministic_guarantees],
            "",
            "## Exclusions",
            "",
            "- None. Every valid source record is materialized exactly once.",
            "",
            "## Limitations",
            "",
            *[f"- {limitation}" for limitation in manifest.limitations],
            "",
        ]
    )
    return "\n".join(lines)


def load_split_manifest(manifest_path: str | Path) -> SplitManifest:
    path = Path(manifest_path).expanduser().resolve(strict=True)
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid split manifest JSON: {path}") from exc
    return SplitManifest.model_validate(content)


def _resolve_split_path(manifest_path: Path, artifact: SplitArtifact) -> Path:
    root = manifest_path.parent.resolve()
    candidate = (root / artifact.path).resolve(strict=True)
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"split artifact escapes manifest directory: {artifact.path}")
    return candidate


def validate_split_manifest(
    manifest_path: str | Path, *, source_path: str | Path | None = None
) -> SplitManifest:
    """Validate schema, source identity, artifacts, members, and leakage gates."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = load_split_manifest(path)
    if source_path is not None:
        source = Path(source_path).expanduser().resolve(strict=True)
        actual_source_sha = sha256_file(source)
        if actual_source_sha != manifest.source.sha256:
            raise ValueError(
                f"source SHA-256 mismatch: expected {manifest.source.sha256}, "
                f"found {actual_source_sha}"
            )
        source_records = _read_source_records(source, manifest.canonical_identity)
        source_members = {record.member.canonical_id: record.member for record in source_records}
        manifest_members = {
            member.canonical_id: member
            for artifact in manifest.splits.values()
            for member in artifact.members
        }
        if source_members != manifest_members:
            raise ValueError("manifest membership metadata differs from the pinned source")

    observed: dict[SplitName, list[_SourceRecord]] = {name: [] for name in SPLIT_NAMES}
    for split, artifact in manifest.splits.items():
        artifact_path = _resolve_split_path(path, artifact)
        actual_digest = sha256_file(artifact_path)
        if actual_digest != artifact.sha256:
            raise ValueError(
                f"{split} digest mismatch: expected {artifact.sha256}, found {actual_digest}"
            )
        identity = manifest.canonical_identity
        records = _read_source_records(artifact_path, identity)
        if len(records) != artifact.record_count:
            raise ValueError(f"{split} record count mismatch")
        expected_members = [member.model_dump(mode="json") for member in artifact.members]
        actual_members = [
            {
                **record.member.model_dump(mode="json"),
                "source_coordinate": artifact.members[index].source_coordinate,
                "raw_line_sha256": artifact.members[index].raw_line_sha256,
            }
            for index, record in enumerate(records)
        ]
        if actual_members != expected_members:
            raise ValueError(f"{split} membership metadata does not match materialized records")
        observed[split] = [
            _SourceRecord(row=record.row, raw_line=record.raw_line, member=artifact.members[index])
            for index, record in enumerate(records)
        ]

    flattened: list[_SourceRecord] = []
    remapped: dict[SplitName, list[int]] = {name: [] for name in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        for record in observed[split]:
            remapped[split].append(len(flattened))
            flattened.append(record)
    audit = _leakage_audit(remapped, flattened)
    if audit != manifest.leakage_audit:
        raise ValueError("stored leakage audit differs from recomputed audit")
    return manifest


def _iter_materialized_records(
    path: Path, manifest: SplitManifest, split: SplitName
) -> Iterator[dict[str, Any]]:
    artifact_path = _resolve_split_path(path, manifest.splits[split])
    with artifact_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"materialized {split} row is not an object")
                yield row


def iter_frozen_records(manifest_path: str | Path, split: SplitName) -> Iterator[dict[str, Any]]:
    """Yield raw source records from a verified frozen partition."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = validate_split_manifest(path)
    yield from _iter_materialized_records(path, manifest, split)


def load_frozen_dataset(
    manifest_path: str | Path, split: SplitName, tokenizer: Any = None
) -> Dataset:
    """Training loader for a frozen split; it never derives new membership."""

    def generate() -> Iterator[dict[str, Any]]:
        for index, row in enumerate(iter_frozen_records(manifest_path, split), 1):
            yield normalize_row(row, tokenizer=tokenizer, index=index)

    return Dataset.from_generator(generate)


def _extract_token_ids(tokenizer: TokenizerLike, text: str) -> Sequence[Any]:
    encoded = tokenizer(text)
    if isinstance(encoded, Mapping):
        token_ids = encoded.get("input_ids")
    else:
        token_ids = encoded
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise TypeError("tokenizer must return a sequence or mapping with sequence input_ids")
    return token_ids


def write_token_statistics(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    tokenizer: TokenizerLike,
    serializer: Serializer,
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    serializer_id: str,
    serializer_version: str,
    serializer_sha256: str,
    context_limit: int | None = None,
) -> TokenStatistics:
    """Write model-specific statistics without mutating canonical split identity."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = validate_split_manifest(path)
    split_stats: dict[SplitName, TokenStatSplit] = {}
    for split in SPLIT_NAMES:
        lengths: list[int] = []
        for row in _iter_materialized_records(path, manifest, split):
            rendered = serializer(row)
            if not isinstance(rendered, str):
                raise TypeError("serializer must return a string")
            lengths.append(len(_extract_token_ids(tokenizer, rendered)))
        split_stats[split] = TokenStatSplit(
            record_count=len(lengths),
            total_tokens=sum(lengths),
            minimum_tokens=min(lengths),
            maximum_tokens=max(lengths),
            truncated_records=(
                sum(length > context_limit for length in lengths) if context_limit else 0
            ),
        )

    statistics = TokenStatistics(
        split_manifest_sha256=sha256_file(path),
        source_split_sha256={split: manifest.splits[split].sha256 for split in SPLIT_NAMES},
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        serializer_id=serializer_id,
        serializer_version=serializer_version,
        serializer_sha256=serializer_sha256,
        context_limit=context_limit,
        splits=split_stats,
    )
    _exclusive_write(
        Path(output_path).expanduser(),
        canonical_json_bytes(statistics.model_dump(mode="json")) + b"\n",
    )
    return statistics
