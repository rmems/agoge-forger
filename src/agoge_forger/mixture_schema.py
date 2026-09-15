"""Canonical schemas for deterministic token-budget mixture composition."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .split_schema import (
    SPLIT_NAMES,
    FrozenModel,
    SplitName,
    TokenStatistics,
    validate_repository_relative_path,
)

MIXTURE_MANIFEST_VERSION: Literal["agoge.mixture-manifest.v1"] = "agoge.mixture-manifest.v1"
MIXTURE_ALGORITHM_VERSION: Literal["sha256-token-budget-v1"] = "sha256-token-budget-v1"
TOKEN_LEDGER_VERSION: Literal["agoge.token-ledger.v1"] = "agoge.token-ledger.v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SOURCE_ID_PATTERN = r"^[a-z][a-z0-9_-]*$"

ProvenanceClass = Literal["SYNTHETIC_FACTORY", "EXTERNAL_PUBLIC_SWE", "PROMETHEUS_REAL"]
ExclusionReason = Literal["truncated", "exceeds-source-quota", "quota-remainder"]


class MixturePolicy(FrozenModel):
    algorithm_version: Literal["sha256-token-budget-v1"] = MIXTURE_ALGORITHM_VERSION
    seed: int
    salt: str = Field(min_length=1)
    accepted_token_budget: int = Field(ge=1)
    experiment_arm: str = Field(min_length=1)
    consumed_split: SplitName = "train"


class MixtureSourceSpec(FrozenModel):
    source_id: str = Field(pattern=_SOURCE_ID_PATTERN)
    provenance_class: ProvenanceClass
    weight: int = Field(ge=1)
    split_manifest: str = Field(min_length=1)
    token_statistics: str = Field(min_length=1)
    token_ledger: str = Field(min_length=1)


class MixtureCompositionSpec(FrozenModel):
    """Pinned mixture inputs: frozen splits, token sidecars, and policy."""

    policy: MixturePolicy
    sources: tuple[MixtureSourceSpec, ...] = Field(min_length=1)
    reserved_lineage_ids: tuple[str, ...] = ()
    reserved_group_ids: tuple[str, ...] = ()
    reserved_experiment_arm: str | None = None

    @model_validator(mode="after")
    def require_unique_source_ids(self) -> MixtureCompositionSpec:
        source_ids = tuple(source.source_id for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        return self


class MixtureTokenizerPin(FrozenModel):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    tokenizer_sha256: str = Field(pattern=_SHA256_PATTERN)
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_limit: int | None = Field(default=None, ge=1)


class TokenLedgerRecord(FrozenModel):
    canonical_id: str = Field(min_length=1)
    split: SplitName
    accepted_tokens: int = Field(ge=0)
    truncated: bool


class TokenLedger(FrozenModel):
    schema_version: Literal["agoge.token-ledger.v1"] = TOKEN_LEDGER_VERSION
    split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_split_sha256: dict[SplitName, str]
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    tokenizer_sha256: str = Field(pattern=_SHA256_PATTERN)
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_limit: int | None = Field(default=None, ge=1)
    records: tuple[TokenLedgerRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_records(self) -> TokenLedger:
        keys = tuple((record.split, record.canonical_id) for record in self.records)
        if len(keys) != len(set(keys)):
            raise ValueError("token ledger records must be unique by split and canonical_id")
        if set(self.source_split_sha256) != set(SPLIT_NAMES):
            raise ValueError(f"source_split_sha256 must contain exactly {SPLIT_NAMES}")
        if not all(_is_lower_sha256(digest) for digest in self.source_split_sha256.values()):
            raise ValueError("source split digests must be lowercase SHA-256 values")
        return self


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
    revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
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
        if self.accepted_tokens > self.quota_tokens:
            raise ValueError("accepted tokens cannot exceed the allocated quota")
        if self.accepted_tokens > self.available_accepted_tokens:
            raise ValueError("accepted tokens cannot exceed available accepted tokens")
        if self.underfilled != (self.shortfall_tokens > 0):
            raise ValueError("underfilled must match a positive token shortfall")
        if self.underfilled and self.shortfall_tokens != self.quota_tokens - self.accepted_tokens:
            raise ValueError("shortfall_tokens must equal the unfilled quota")
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
        source_ids = tuple(source.source_id for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("mixture source_id values must be unique")
        if self.artifact.record_count != len(self.selected):
            raise ValueError("mixture record_count does not match selected members")
        selected_tokens = sum(member.accepted_tokens for member in self.selected)
        if selected_tokens != self.artifact.accepted_tokens:
            raise ValueError("mixture accepted tokens do not match selected members")
        if selected_tokens != sum(source.accepted_tokens for source in self.sources):
            raise ValueError("mixture accepted tokens do not match per-source totals")
        _require_unique((member.canonical_id for member in self.selected), "canonical IDs")
        budget = self.policy.accepted_token_budget
        if sum(source.quota_tokens for source in self.sources) != budget:
            raise ValueError("source quotas must sum to the accepted-token budget")
        if self.artifact.accepted_tokens > budget:
            raise ValueError("mixture accepted tokens cannot exceed the accepted-token budget")
        return self


def tokenizer_pin_from_statistics(statistics: TokenStatistics) -> MixtureTokenizerPin:
    return MixtureTokenizerPin(
        model_id=statistics.model_id,
        model_revision=statistics.model_revision,
        tokenizer_id=statistics.tokenizer_id,
        tokenizer_revision=statistics.tokenizer_revision,
        tokenizer_sha256=statistics.tokenizer_sha256,
        serializer_id=statistics.serializer_id,
        serializer_version=statistics.serializer_version,
        serializer_sha256=statistics.serializer_sha256,
        context_limit=statistics.context_limit,
    )


def _is_lower_sha256(digest: str) -> bool:
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _require_unique(values: Iterable[str], label: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be globally unique across the mixture")
