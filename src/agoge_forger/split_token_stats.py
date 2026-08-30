"""Immutable model-specific token statistics derived from frozen splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._token_provenance import (
    SerializerBinding,
    TokenizerBinding,
    derive_serializer_provenance,
    derive_tokenizer_provenance,
    derive_tokenizer_sha256,
)
from .split_loaders import iter_materialized_records
from .split_materialize import exclusive_write
from .split_schema import (
    SPLIT_NAMES,
    Serializer,
    SplitName,
    TokenizerLike,
    TokenStatistics,
    TokenStatisticsSpec,
    TokenStatSplit,
    canonical_json_bytes,
    sha256_file,
)
from .split_validation import validate_split_manifest


@dataclass(frozen=True)
class TokenStatisticsDerivation:
    tokenizer: TokenizerBinding
    serializer: SerializerBinding
    spec: TokenStatisticsSpec

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, TokenizerBinding):
            raise TypeError("tokenizer must be a TokenizerBinding")
        if not isinstance(self.serializer, SerializerBinding):
            raise TypeError("serializer must be a SerializerBinding")


def write_token_statistics(
    manifest_path: str | Path,
    output_path: str | Path,
    derivation: TokenStatisticsDerivation,
) -> TokenStatistics:
    """Write model-specific statistics without mutating canonical split identity."""

    spec = _verified_spec(derivation)
    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = validate_split_manifest(path)
    counter = _TokenCounter(derivation.tokenizer, derivation.serializer, spec.context_limit)
    split_stats = {split: counter.for_split(path, manifest, split) for split in SPLIT_NAMES}
    spec = _verified_spec(derivation)
    statistics = TokenStatistics(
        split_manifest_sha256=sha256_file(path),
        source_split_sha256={split: manifest.splits[split].sha256 for split in SPLIT_NAMES},
        splits=split_stats,
        **spec.model_dump(),
    )
    exclusive_write(
        Path(output_path).expanduser(),
        canonical_json_bytes(statistics.model_dump(mode="json")) + b"\n",
    )
    return statistics


def _verified_spec(derivation: TokenStatisticsDerivation) -> TokenStatisticsSpec:
    tokenizer_id, tokenizer_revision = derive_tokenizer_provenance(
        derivation.tokenizer.implementation
    )
    serializer_id, serializer_version, serializer_sha256 = derive_serializer_provenance(
        derivation.serializer.implementation
    )
    tokenizer_sha256 = derive_tokenizer_sha256(derivation.tokenizer.implementation)
    bound_provenance = {
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_sha256": tokenizer_sha256,
        "serializer_id": serializer_id,
        "serializer_version": serializer_version,
        "serializer_sha256": serializer_sha256,
    }
    binding_provenance = {
        "tokenizer_id": derivation.tokenizer.tokenizer_id,
        "tokenizer_revision": derivation.tokenizer.tokenizer_revision,
        "tokenizer_sha256": derivation.tokenizer.tokenizer_sha256,
        "serializer_id": derivation.serializer.serializer_id,
        "serializer_version": derivation.serializer.serializer_version,
        "serializer_sha256": derivation.serializer.serializer_sha256,
    }
    for provenance_field, bound_value in bound_provenance.items():
        if binding_provenance[provenance_field] != bound_value:
            raise ValueError(f"{provenance_field} changed after the callable provenance was bound")
        declared_value = getattr(derivation.spec, provenance_field)
        if declared_value != bound_value:
            raise ValueError(
                f"{provenance_field} does not match the bound callable provenance: "
                f"declared {declared_value!r}, bound {bound_value!r}"
            )
    return derivation.spec.model_copy(update=bound_provenance)


@dataclass(frozen=True)
class _TokenCounter:
    tokenizer: TokenizerLike
    serializer: Serializer
    context_limit: int | None

    def for_split(self, manifest_path: Path, manifest: Any, split: SplitName) -> TokenStatSplit:
        lengths = [
            len(_extract_token_ids(self.tokenizer, _render(self.serializer, row)))
            for row in iter_materialized_records(manifest_path, manifest, split)
        ]
        truncated = (
            sum(length > self.context_limit for length in lengths) if self.context_limit else 0
        )
        return TokenStatSplit(
            record_count=len(lengths),
            total_tokens=sum(lengths),
            minimum_tokens=min(lengths),
            maximum_tokens=max(lengths),
            truncated_records=truncated,
        )


def _render(serializer: Serializer, row: Mapping[str, Any]) -> str:
    rendered = serializer(row)
    if not isinstance(rendered, str):
        raise TypeError("serializer must return a string")
    return rendered


def _extract_token_ids(tokenizer: TokenizerLike, text: str) -> Sequence[Any]:
    encoded = tokenizer(text)
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise TypeError("tokenizer must return a sequence or mapping with sequence input_ids")
    return token_ids
