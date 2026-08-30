"""Immutable model-specific token statistics derived from frozen splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from .split_validation import iter_materialized_records, validate_split_manifest


def write_token_statistics(
    manifest_path: str | Path,
    output_path: str | Path,
    tokenizer: TokenizerLike,
    serializer: Serializer,
    spec: TokenStatisticsSpec,
) -> TokenStatistics:
    """Write model-specific statistics without mutating canonical split identity."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = validate_split_manifest(path)
    counter = _TokenCounter(tokenizer, serializer, spec.context_limit)
    split_stats = {split: counter.for_split(path, manifest, split) for split in SPLIT_NAMES}
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
