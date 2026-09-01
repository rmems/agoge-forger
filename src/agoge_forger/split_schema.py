"""Canonical schemas and digest helpers for immutable source-level splits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPLIT_MANIFEST_VERSION: Literal["agoge.split-manifest.v1"] = "agoge.split-manifest.v1"
SPLIT_ALGORITHM_VERSION: Literal["sha256-atomic-bucket-v1"] = "sha256-atomic-bucket-v1"
TOKEN_STATS_VERSION: Literal["agoge.token-stats.v1"] = "agoge.token-stats.v1"
IMMUTABLE_REVISION_PATTERN = r"^[0-9a-f]{40,64}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

SplitName = Literal["train", "validation", "held_out"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "held_out")


class FrozenModel(BaseModel):
    """Strict base class for versioned contract records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceFile(FrozenModel):
    repository: str = Field(min_length=1)
    revision: str = Field(pattern=IMMUTABLE_REVISION_PATTERN)
    dataset_version: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    record_count: int = Field(ge=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_relative_path(value)


class CanonicalIdentityPolicy(FrozenModel):
    canonical_id_field: str = Field(default="canonical_id", min_length=1)
    lineage_id_field: str = Field(default="lineage_id", min_length=1)
    group_id_field: str = Field(default="group_id", min_length=1)
    missing_lineage: Literal["canonical_id"] = "canonical_id"
    content_hash_policy: Literal[
        "canonical-json-excluding-identity-fields",
        "normalized-training-payload-v1",
    ] = "canonical-json-excluding-identity-fields"
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
        _require_split_names(self.weights, "weights")
        if any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("all split weights must be positive")
        return self


def _new_identity_policy() -> CanonicalIdentityPolicy:
    return CanonicalIdentityPolicy(content_hash_policy="normalized-training-payload-v1")


class SplitMaterializationSpec(FrozenModel):
    """Pinned provenance and policy supplied to one materialization."""

    source_repository: str = Field(min_length=1)
    source_revision: str = Field(pattern=IMMUTABLE_REVISION_PATTERN)
    dataset_version: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    split_policy: SplitPolicy
    canonical_identity: CanonicalIdentityPolicy = Field(default_factory=_new_identity_policy)

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        return validate_repository_relative_path(value)


class SplitMember(FrozenModel):
    canonical_id: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    group_id: str | None = None
    source_coordinate: str = Field(min_length=1)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    raw_line_sha256: str = Field(pattern=_SHA256_PATTERN)
    materialized_line_sha256: str = Field(pattern=_SHA256_PATTERN)


class SplitArtifact(FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    record_count: int = Field(ge=1)
    members: tuple[SplitMember, ...]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_relative_path(value)

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
        if any(count != 0 for count in _leakage_counts(self)):
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
        "tokenizer_sha256",
        "serializer_id",
        "serializer_version",
        "serializer_sha256",
    )


class SplitManifest(FrozenModel):
    """The single canonical source-level partition manifest."""

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
    def validate_membership(self) -> SplitManifest:
        _require_split_names(self.splits, "splits")
        _require_source_count(self)
        members = tuple(member for split in self.splits.values() for member in split.members)
        _require_unique((member.canonical_id for member in members), "canonical IDs")
        _require_unique((member.source_coordinate for member in members), "source coordinates")
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


class TokenStatisticsSpec(FrozenModel):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=IMMUTABLE_REVISION_PATTERN)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=IMMUTABLE_REVISION_PATTERN)
    tokenizer_sha256: str = Field(pattern=_SHA256_PATTERN)
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_limit: int | None = Field(default=None, ge=1)


class TokenStatistics(FrozenModel):
    schema_version: Literal["agoge.token-stats.v1"] = TOKEN_STATS_VERSION
    split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_split_sha256: dict[SplitName, str]
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=IMMUTABLE_REVISION_PATTERN)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=IMMUTABLE_REVISION_PATTERN)
    tokenizer_sha256: str = Field(pattern=_SHA256_PATTERN)
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_limit: int | None = Field(default=None, ge=1)
    splits: dict[SplitName, TokenStatSplit]

    @model_validator(mode="after")
    def validate_split_identity(self) -> TokenStatistics:
        _require_split_names(self.source_split_sha256, "source_split_sha256")
        _require_split_names(self.splits, "splits")
        if not all(_is_lower_sha256(digest) for digest in self.source_split_sha256.values()):
            raise ValueError("source split digests must be lowercase SHA-256 values")
        return self


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's canonical JSON encoding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def validate_repository_relative_path(value: str) -> str:
    """Require one canonical, portable path confined to a repository root."""

    _require_portable_path_text(value)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    _require_relative_path(posix_path, windows_path)
    _require_canonical_path(value, posix_path)
    return value


def _require_portable_path_text(value: str) -> None:
    text_error = (
        "repository-relative path must not contain surrounding whitespace or control characters"
    )
    if value != value.strip():
        raise ValueError(text_error)
    if any(ord(character) < 32 for character in value):
        raise ValueError(text_error)
    if "\\" in value:
        raise ValueError("repository-relative path must use POSIX '/' separators")


def _require_relative_path(posix_path: PurePosixPath, windows_path: PureWindowsPath) -> None:
    _require_posix_relative_path(posix_path)
    _require_windows_relative_path(windows_path)
    _require_no_parent_traversal(posix_path)


def _require_posix_relative_path(posix_path: PurePosixPath) -> None:
    absolute_error = "repository-relative path must not be absolute or use a drive prefix"
    if posix_path.is_absolute():
        raise ValueError(absolute_error)


def _require_windows_relative_path(windows_path: PureWindowsPath) -> None:
    absolute_error = "repository-relative path must not be absolute or use a drive prefix"
    if windows_path.is_absolute():
        raise ValueError(absolute_error)
    if windows_path.drive:
        raise ValueError(absolute_error)


def _require_no_parent_traversal(posix_path: PurePosixPath) -> None:
    if ".." in posix_path.parts:
        raise ValueError("repository-relative path must not escape the repository root")


def _require_canonical_path(value: str, posix_path: PurePosixPath) -> None:
    canonical = posix_path.as_posix()
    if canonical in {"", "."}:
        raise ValueError("repository-relative path must be a canonical POSIX path")
    if canonical != value:
        raise ValueError("repository-relative path must be a canonical POSIX path")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_split_names(value: Mapping[Any, Any], label: str) -> None:
    if set(value) != set(SPLIT_NAMES):
        raise ValueError(f"{label} must contain exactly {SPLIT_NAMES}")


def _require_source_count(manifest: SplitManifest) -> None:
    total = sum(split.record_count for split in manifest.splits.values())
    if total != manifest.source.record_count:
        raise ValueError("split counts do not equal the source record count")


def _require_unique(values: Iterable[str], label: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be globally unique across the manifest")


def _leakage_counts(audit: LeakageAudit) -> tuple[int, ...]:
    return (
        audit.exact_content_cross_split,
        audit.canonical_id_cross_split,
        audit.source_coordinate_cross_split,
        audit.lineage_cross_split,
        audit.declared_group_cross_split,
    )


def _is_lower_sha256(digest: str) -> bool:
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
