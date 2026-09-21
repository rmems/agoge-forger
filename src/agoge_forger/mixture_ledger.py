"""Immutable per-record token ledgers derived from frozen splits."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ._atomic_file import publish_bytes_noreplace, write_fsynced_bytes
from ._strict_json import decode_json_object
from .mixture_ledger_schema import TokenLedger, TokenLedgerRecord
from .mixture_schema import TOKEN_LEDGER_VERSION
from .split_contract import (
    SPLIT_NAMES,
    SplitManifest,
    SplitMember,
    SplitName,
    TokenStatistics,
    TokenStatSplit,
)
from .split_loaders import iter_materialized_records
from .split_schema import canonical_json_bytes, sha256_bytes
from .split_token_stats import TokenStatisticsDerivation, _verified_spec, measure_rendered_tokens
from .split_validation import validate_split_manifest_snapshot


def write_token_ledger(
    manifest_path: str | Path,
    output_path: str | Path,
    derivation: TokenStatisticsDerivation,
) -> TokenLedger:
    """Write per-record token lengths without mutating canonical split identity."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest_snapshot = path.read_bytes()
    manifest = validate_split_manifest_snapshot(path, manifest_snapshot)
    records = tuple(
        _ledger_record(member.canonical_id, split, row, derivation)
        for split in SPLIT_NAMES
        for member, row in _zip_split_rows(path, manifest, split)
    )
    spec = _verified_spec(derivation)
    ledger = TokenLedger(
        schema_version=TOKEN_LEDGER_VERSION,
        split_manifest_sha256=sha256_bytes(manifest_snapshot),
        source_split_sha256={split: manifest.splits[split].sha256 for split in SPLIT_NAMES},
        tokenizer_id=spec.tokenizer_id,
        tokenizer_revision=spec.tokenizer_revision,
        tokenizer_sha256=spec.tokenizer_sha256,
        serializer_id=spec.serializer_id,
        serializer_version=spec.serializer_version,
        serializer_sha256=spec.serializer_sha256,
        context_limit=spec.context_limit,
        records=records,
    )
    publish_bytes_noreplace(
        Path(output_path).expanduser(),
        ledger_bytes(ledger),
        refusal="refusing to overwrite frozen artifact",
        writer=write_fsynced_bytes,
    )
    return ledger


def load_token_ledger(path: str | Path) -> tuple[TokenLedger, str]:
    ledger_path = Path(path).expanduser().resolve(strict=True)
    snapshot = ledger_path.read_bytes()
    value = decode_json_object(snapshot, str(ledger_path), object_label="token ledger")
    return TokenLedger.model_validate(value), sha256_bytes(snapshot)


def load_token_statistics(path: str | Path) -> tuple[TokenStatistics, str]:
    stats_path = Path(path).expanduser().resolve(strict=True)
    snapshot = stats_path.read_bytes()
    value = decode_json_object(snapshot, str(stats_path), object_label="token statistics")
    return TokenStatistics.model_validate(value), sha256_bytes(snapshot)


def ledger_bytes(ledger: TokenLedger) -> bytes:
    return canonical_json_bytes(ledger.model_dump(mode="json")) + b"\n"


def require_ledger_matches_statistics(ledger: TokenLedger, statistics: TokenStatistics) -> None:
    _require_ledger_manifest_identity(ledger, statistics)
    _require_matching_tokenizer(ledger, statistics)
    for split in SPLIT_NAMES:
        _require_ledger_split_statistics(ledger, statistics, split)


def _require_ledger_manifest_identity(ledger: TokenLedger, statistics: TokenStatistics) -> None:
    if ledger.split_manifest_sha256 != statistics.split_manifest_sha256:
        raise ValueError("token ledger split-manifest digest does not match token statistics")
    if ledger.source_split_sha256 != statistics.source_split_sha256:
        raise ValueError("token ledger source-split digests do not match token statistics")


def _require_ledger_split_statistics(
    ledger: TokenLedger, statistics: TokenStatistics, split: SplitName
) -> None:
    split_records = tuple(record for record in ledger.records if record.split == split)
    stats = statistics.splits[split]
    _require_ledger_split_record_count(split, split_records, stats)
    _require_ledger_split_token_totals(split, split_records, stats)
    _require_ledger_split_token_bounds(split, split_records, stats)


def _require_ledger_split_record_count(
    split: SplitName, split_records: tuple[TokenLedgerRecord, ...], stats: TokenStatSplit
) -> None:
    if len(split_records) != stats.record_count:
        raise ValueError(f"token ledger {split} record count does not match token statistics")


def _require_ledger_split_token_totals(
    split: SplitName, split_records: tuple[TokenLedgerRecord, ...], stats: TokenStatSplit
) -> None:
    total = sum(record.accepted_tokens for record in split_records)
    truncated = sum(record.truncated for record in split_records)
    if total != stats.total_tokens:
        raise ValueError(f"token ledger {split} token total does not match token statistics")
    if truncated != stats.truncated_records:
        raise ValueError(f"token ledger {split} truncated count does not match token statistics")


def _require_ledger_split_token_bounds(
    split: SplitName, split_records: tuple[TokenLedgerRecord, ...], stats: TokenStatSplit
) -> None:
    lengths = tuple(record.accepted_tokens for record in split_records)
    if min(lengths) != stats.minimum_tokens or max(lengths) != stats.maximum_tokens:
        raise ValueError(f"token ledger {split} token bounds do not match token statistics")


def _ledger_record(
    canonical_id: str,
    split: SplitName,
    row: Mapping[str, Any],
    derivation: TokenStatisticsDerivation,
) -> TokenLedgerRecord:
    length, truncated = measure_rendered_tokens(
        derivation.tokenizer,
        derivation.serializer,
        row,
        derivation.spec.context_limit,
    )
    return TokenLedgerRecord(
        canonical_id=canonical_id,
        split=split,
        accepted_tokens=length,
        truncated=truncated,
    )


def _zip_split_rows(
    manifest_path: Path, manifest: SplitManifest, split: SplitName
) -> Iterator[tuple[SplitMember, dict[str, Any]]]:
    artifact = manifest.splits[split]
    rows = tuple(iter_materialized_records(manifest_path, manifest, split))
    if len(rows) != len(artifact.members):
        raise ValueError(f"frozen {split} membership does not match materialized records")
    return zip(artifact.members, rows, strict=True)


def _require_matching_tokenizer(ledger: TokenLedger, statistics: TokenStatistics) -> None:
    fields = (
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_sha256",
        "serializer_id",
        "serializer_version",
        "serializer_sha256",
        "context_limit",
    )
    for field_name in fields:
        if getattr(ledger, field_name) != getattr(statistics, field_name):
            raise ValueError(f"token ledger {field_name} does not match token statistics")
