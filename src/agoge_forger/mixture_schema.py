"""Canonical schemas for deterministic token-budget mixture composition."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .split_schema import FrozenModel, SplitName, TokenStatistics

MIXTURE_MANIFEST_VERSION: Literal["agoge.mixture-manifest.v1"] = "agoge.mixture-manifest.v1"
MIXTURE_ALGORITHM_VERSION: Literal["sha256-token-budget-v1"] = "sha256-token-budget-v1"
TOKEN_LEDGER_VERSION: Literal["agoge.token-ledger.v1"] = "agoge.token-ledger.v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REVISION_PATTERN = r"^[0-9a-f]{40,64}$"
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
    reserved_canonical_ids: tuple[str, ...] = ()
    reserved_content_sha256: tuple[str, ...] = ()
    reserved_experiment_arm: str | None = None

    @model_validator(mode="after")
    def require_unique_source_ids(self) -> MixtureCompositionSpec:
        source_ids = tuple(source.source_id for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        return self


class MixtureTokenizerPin(FrozenModel):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=_REVISION_PATTERN)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=_REVISION_PATTERN)
    tokenizer_sha256: str = Field(pattern=_SHA256_PATTERN)
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_limit: int | None = Field(default=None, ge=1)


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
