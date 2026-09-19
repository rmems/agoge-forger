"""Published mixture snapshot schemas."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .mixture_schema import (
    _REVISION_PATTERN,
    _SHA256_PATTERN,
    _SOURCE_ID_PATTERN,
    MIXTURE_MANIFEST_VERSION,
    ExclusionReason,
    MixturePolicy,
    MixtureTokenizerPin,
    ProvenanceClass,
)
from .split_schema import FrozenModel, validate_repository_relative_path


class MixtureMember(FrozenModel):
    source_id: str = Field(pattern=_SOURCE_ID_PATTERN)
    canonical_id: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    group_id: str | None = None
    source_coordinate: str = Field(min_length=1)
    accepted_tokens: int = Field(ge=0)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)


class MixtureExclusion(FrozenModel):
    source_id: str = Field(pattern=_SOURCE_ID_PATTERN)
    canonical_id: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    reason: ExclusionReason


class MixtureSourceResult(FrozenModel):
    source_id: str = Field(pattern=_SOURCE_ID_PATTERN)
    provenance_class: ProvenanceClass
    repository: str = Field(min_length=1)
    revision: str = Field(pattern=_REVISION_PATTERN)
    dataset_version: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    token_statistics_sha256: str = Field(pattern=_SHA256_PATTERN)
    token_ledger_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    weight: int = Field(ge=1)
    quota_tokens: int = Field(ge=0)
    available_accepted_tokens: int = Field(ge=0)
    accepted_tokens: int = Field(ge=0)
    example_count: int = Field(ge=0)
    excluded_example_count: int = Field(ge=0)
    underfilled: bool
    shortfall_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_fill(self) -> MixtureSourceResult:
        _require_fill_invariants(self)
        return self


class MixtureArtifact(FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    record_count: int = Field(ge=1)
    accepted_tokens: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_relative_path(value)


class MixtureLineageAudit(FrozenModel):
    status: Literal["passed"] = "passed"
    duplicate_lineage: int = 0
    duplicate_canonical_id: int = 0
    duplicate_content: int = 0
    duplicate_group: int = 0
    lineage_split_crossing: int = 0
    lineage_arm_crossing: int = 0

    @model_validator(mode="after")
    def fail_closed(self) -> MixtureLineageAudit:
        counts = (
            self.duplicate_lineage,
            self.duplicate_canonical_id,
            self.duplicate_content,
            self.duplicate_group,
            self.lineage_split_crossing,
            self.lineage_arm_crossing,
        )
        if any(count != 0 for count in counts):
            raise ValueError("a passed mixture lineage audit cannot contain collisions")
        return self


class MixtureManifest(FrozenModel):
    schema_version: Literal["agoge.mixture-manifest.v1"] = MIXTURE_MANIFEST_VERSION
    policy: MixturePolicy
    tokenizer: MixtureTokenizerPin
    sources: tuple[MixtureSourceResult, ...] = Field(min_length=1)
    artifact: MixtureArtifact
    selected: tuple[MixtureMember, ...]
    exclusions: tuple[MixtureExclusion, ...]
    lineage_audit: MixtureLineageAudit
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_membership(self) -> MixtureManifest:
        _require_manifest_membership(self)
        return self


def _require_fill_invariants(source: MixtureSourceResult) -> None:
    if source.accepted_tokens > source.quota_tokens:
        raise ValueError("accepted tokens cannot exceed the allocated quota")
    if source.accepted_tokens > source.available_accepted_tokens:
        raise ValueError("accepted tokens cannot exceed available accepted tokens")
    if source.underfilled != (source.shortfall_tokens > 0):
        raise ValueError("underfilled must match a positive token shortfall")
    if source.shortfall_tokens != source.quota_tokens - source.accepted_tokens:
        raise ValueError("shortfall_tokens must equal the unfilled quota")


def _require_manifest_membership(manifest: MixtureManifest) -> None:
    source_ids = _require_unique_source_ids(manifest)
    _require_selected_membership(manifest, source_ids)
    selected_tokens = _require_selected_token_totals(manifest)
    _require_attributed_source_tokens(manifest, source_ids)
    _require_unique((member.canonical_id for member in manifest.selected), "canonical IDs")
    _require_budget_alignment(manifest, selected_tokens)


def _require_unique_source_ids(manifest: MixtureManifest) -> tuple[str, ...]:
    source_ids = tuple(source.source_id for source in manifest.sources)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("mixture source_id values must be unique")
    return source_ids


def _require_selected_membership(manifest: MixtureManifest, source_ids: tuple[str, ...]) -> None:
    declared = set(source_ids)
    if any(member.source_id not in declared for member in manifest.selected):
        raise ValueError("selected members must belong to a declared mixture source")
    if manifest.artifact.record_count != len(manifest.selected):
        raise ValueError("mixture record_count does not match selected members")


def _require_selected_token_totals(manifest: MixtureManifest) -> int:
    selected_tokens = sum(member.accepted_tokens for member in manifest.selected)
    if selected_tokens != manifest.artifact.accepted_tokens:
        raise ValueError("mixture accepted tokens do not match selected members")
    if selected_tokens != sum(source.accepted_tokens for source in manifest.sources):
        raise ValueError("mixture accepted tokens do not match per-source totals")
    return selected_tokens


def _require_budget_alignment(manifest: MixtureManifest, selected_tokens: int) -> None:
    budget = manifest.policy.accepted_token_budget
    if sum(source.quota_tokens for source in manifest.sources) != budget:
        raise ValueError("source quotas must sum to the accepted-token budget")
    if selected_tokens > budget:
        raise ValueError("mixture accepted tokens cannot exceed the accepted-token budget")


def _require_attributed_source_tokens(
    manifest: MixtureManifest, source_ids: tuple[str, ...]
) -> None:
    attributed = {
        source_id: sum(
            member.accepted_tokens for member in manifest.selected if member.source_id == source_id
        )
        for source_id in source_ids
    }
    if any(source.accepted_tokens != attributed[source.source_id] for source in manifest.sources):
        raise ValueError("per-source accepted tokens do not match selected members")


def _require_unique(values: Iterable[str], label: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be globally unique across the mixture")
