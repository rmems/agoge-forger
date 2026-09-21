"""Load and pin frozen mixture sources without publishing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .mixture_allocate import SelectableRecord
from .mixture_ledger import (
    load_token_ledger,
    load_token_statistics,
    require_ledger_matches_statistics,
)
from .mixture_ledger_schema import TokenLedger
from .mixture_schema import MixtureSourceSpec, MixtureTokenizerPin, tokenizer_pin_from_statistics
from .path_safety import resolve_existing_path
from .split_contract import SPLIT_NAMES, SplitManifest, SplitName, TokenStatistics
from .split_loaders import iter_materialized_records
from .split_schema import canonical_json_bytes, sha256_bytes
from .split_validation import validate_split_manifest_snapshot


@dataclass(frozen=True)
class _ConsumedSplit:
    source_id: str
    manifest_path: Path
    manifest: SplitManifest
    ledger: TokenLedger
    consumed: SplitName


@dataclass(frozen=True)
class LoadedSource:
    spec: MixtureSourceSpec
    manifest_path: Path
    manifest: SplitManifest
    manifest_sha256: str
    statistics: TokenStatistics
    statistics_sha256: str
    ledger: TokenLedger
    ledger_sha256: str
    consumed_split: SplitName
    records: tuple[SelectableRecord, ...]


def load_source(
    source: MixtureSourceSpec,
    spec_dir: str | Path | None,
    consumed: SplitName,
) -> LoadedSource:
    base = Path(spec_dir).expanduser() if spec_dir is not None else Path.cwd()
    manifest_path = _resolve_input(source.split_manifest, base)
    statistics_path = _resolve_input(source.token_statistics, base)
    ledger_path = _resolve_input(source.token_ledger, base)
    manifest_snapshot = manifest_path.read_bytes()
    manifest = validate_split_manifest_snapshot(manifest_path, manifest_snapshot)
    manifest_sha256 = sha256_bytes(manifest_snapshot)
    statistics, statistics_sha256 = load_token_statistics(statistics_path)
    ledger, ledger_sha256 = load_token_ledger(ledger_path)
    require_sidecar_identity(manifest, manifest_sha256, statistics, ledger)
    require_ledger_matches_statistics(ledger, statistics)
    records = consumed_records(
        _ConsumedSplit(source.source_id, manifest_path, manifest, ledger, consumed)
    )
    return LoadedSource(
        spec=source,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        statistics=statistics,
        statistics_sha256=statistics_sha256,
        ledger=ledger,
        ledger_sha256=ledger_sha256,
        consumed_split=consumed,
        records=records,
    )


def require_tokenizer_agreement(sources: Sequence[LoadedSource]) -> MixtureTokenizerPin:
    pin = tokenizer_pin_from_statistics(sources[0].statistics)
    pin_fields = (
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_sha256",
        "serializer_id",
        "serializer_version",
        "serializer_sha256",
        "context_limit",
    )
    for source in sources[1:]:
        other = tokenizer_pin_from_statistics(source.statistics)
        for field_name in pin_fields:
            if getattr(pin, field_name) != getattr(other, field_name):
                raise ValueError(
                    "tokenizer-revision mismatch: source "
                    f"{sources[0].spec.source_id} {field_name}="
                    f"{getattr(pin, field_name)!r} does not match "
                    f"{source.spec.source_id} {getattr(other, field_name)!r}"
                )
    return pin


def require_sidecar_identity(
    manifest: SplitManifest,
    manifest_sha256: str,
    statistics: TokenStatistics,
    ledger: TokenLedger,
) -> None:
    _require_sidecar_manifest_digest(
        statistics.split_manifest_sha256, manifest_sha256, "statistics"
    )
    _require_sidecar_manifest_digest(ledger.split_manifest_sha256, manifest_sha256, "ledger")
    expected = {split: manifest.splits[split].sha256 for split in SPLIT_NAMES}
    _require_sidecar_split_digests(statistics.source_split_sha256, expected, "statistics")
    _require_sidecar_split_digests(ledger.source_split_sha256, expected, "ledger")


def _require_sidecar_manifest_digest(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f"token {label} split-manifest digest does not match the frozen manifest")


def _require_sidecar_split_digests(
    actual: dict[SplitName, str], expected: dict[SplitName, str], label: str
) -> None:
    if actual != expected:
        raise ValueError(f"token {label} source-split digests do not match the frozen manifest")


def consumed_records(context: _ConsumedSplit) -> tuple[SelectableRecord, ...]:
    token_by_id = {
        record.canonical_id: record
        for record in context.ledger.records
        if record.split == context.consumed
    }
    artifact = context.manifest.splits[context.consumed]
    rows = tuple(
        iter_materialized_records(context.manifest_path, context.manifest, context.consumed)
    )
    if len(rows) != len(artifact.members):
        raise ValueError(
            f"{context.source_id}: frozen {context.consumed} membership does not match "
            "materialized rows"
        )
    if set(token_by_id) != {member.canonical_id for member in artifact.members}:
        raise ValueError(
            f"{context.source_id}: token ledger does not cover the consumed "
            f"{context.consumed} split"
        )
    return tuple(
        SelectableRecord(
            source_id=context.source_id,
            canonical_id=member.canonical_id,
            lineage_id=member.lineage_id,
            group_id=member.group_id,
            source_coordinate=member.source_coordinate,
            content_sha256=member.content_sha256,
            accepted_tokens=token_by_id[member.canonical_id].accepted_tokens,
            truncated=token_by_id[member.canonical_id].truncated,
            line=canonical_json_bytes(row) + b"\n",
        )
        for member, row in zip(artifact.members, rows, strict=True)
    )


def _resolve_input(value: str, spec_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = spec_dir / candidate
    return resolve_existing_path(str(candidate), must_be_file=True)
