"""Token-ledger schemas derived from frozen splits."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .mixture_schema import (
    _REVISION_PATTERN,
    _SHA256_PATTERN,
    TOKEN_LEDGER_VERSION,
    _is_lower_sha256,
)
from .split_schema import SPLIT_NAMES, FrozenModel, SplitName


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
    tokenizer_revision: str = Field(pattern=_REVISION_PATTERN)
    tokenizer_sha256: str = Field(pattern=_SHA256_PATTERN)
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_limit: int | None = Field(default=None, ge=1)
    records: tuple[TokenLedgerRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_records(self) -> TokenLedger:
        _require_unique_ledger_keys(self.records)
        _require_complete_source_digests(self.source_split_sha256)
        return self


def _require_unique_ledger_keys(records: tuple[TokenLedgerRecord, ...]) -> None:
    keys = tuple((record.split, record.canonical_id) for record in records)
    if len(keys) != len(set(keys)):
        raise ValueError("token ledger records must be unique by split and canonical_id")


def _require_complete_source_digests(source_split_sha256: dict[SplitName, str]) -> None:
    if set(source_split_sha256) != set(SPLIT_NAMES):
        raise ValueError(f"source_split_sha256 must contain exactly {SPLIT_NAMES}")
    if not all(_is_lower_sha256(digest) for digest in source_split_sha256.values()):
        raise ValueError("source split digests must be lowercase SHA-256 values")
