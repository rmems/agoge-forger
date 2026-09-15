"""Deterministic materialization of exact accepted-token mixtures."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._atomic_directory import require_rename_noreplace_support
from ._mixture_report import manifest_bytes, render_report
from ._source_snapshot import nearest_existing_output_ancestor
from .mixture_allocate import (
    MixtureGroup,
    SelectableRecord,
    SourceSelection,
    allocate_token_quotas,
    select_groups_for_quota,
)
from .mixture_ledger import (
    load_token_ledger,
    load_token_statistics,
    require_ledger_matches_statistics,
)
from .mixture_schema import (
    MixtureArtifact,
    MixtureCompositionSpec,
    MixtureLineageAudit,
    MixtureManifest,
    MixtureMember,
    MixtureSourceResult,
    MixtureSourceSpec,
    MixtureTokenizerPin,
    TokenLedger,
    tokenizer_pin_from_statistics,
)
from .path_safety import resolve_existing_path
from .split_contract import SPLIT_NAMES, SplitManifest, SplitMember, SplitName, TokenStatistics
from .split_loaders import iter_materialized_records
from .split_materialize import (
    SourceRecord,
    _atomic_components,
    _component_anchor,
    _publish_snapshot,
    exclusive_write,
)
from .split_schema import canonical_json_bytes, sha256_file
from .split_validation import validate_split_manifest


@dataclass(frozen=True)
class _LoadedSource:
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


@dataclass
class _IdentityOwners:
    owners: dict[str, dict[str, tuple[str, SplitName, str]]] = field(
        default_factory=lambda: {
            "lineage": {},
            "canonical": {},
            "content": {},
            "group": {},
        }
    )


def compose_mixture(
    spec: MixtureCompositionSpec,
    output_dir: str | Path,
    *,
    spec_dir: str | Path | None = None,
) -> MixtureManifest:
    """Materialize one immutable, deterministic token-budget mixture."""

    destination = Path(output_dir).expanduser()
    _refuse_existing_destination(destination)
    consumed = spec.policy.consumed_split
    loaded = tuple(_load_source(source, spec_dir, consumed) for source in spec.sources)
    tokenizer = _require_tokenizer_agreement(loaded)
    _require_lineage_isolation(loaded, spec)
    source_ids = tuple(source.spec.source_id for source in loaded)
    weights = {source.spec.source_id: source.spec.weight for source in loaded}
    quotas = allocate_token_quotas(spec.policy.accepted_token_budget, weights, source_ids)
    selections = tuple(
        _select_source(source, quotas[source.spec.source_id], spec) for source in loaded
    )
    return _publish_mixture(destination, spec, tokenizer, loaded, selections)


def _publish_mixture(
    destination: Path,
    spec: MixtureCompositionSpec,
    tokenizer: MixtureTokenizerPin,
    loaded: Sequence[_LoadedSource],
    selections: Sequence[SourceSelection],
) -> MixtureManifest:
    staging_parent = nearest_existing_output_ancestor(destination)
    require_rename_noreplace_support(staging_parent)
    with tempfile.TemporaryDirectory(
        prefix=".agoge-mixture-staging-",
        dir=staging_parent,
    ) as staging_dir:
        staged_destination = Path(staging_dir) / "snapshot"
        staged_destination.mkdir()
        artifact = _write_mixture_artifact(staged_destination, selections)
        manifest = _build_manifest(spec, tokenizer, loaded, selections, artifact)
        exclusive_write(staged_destination / "mixture_manifest.json", manifest_bytes(manifest))
        exclusive_write(
            staged_destination / "mixture_report.md",
            render_report(manifest).encode("utf-8"),
        )
        _publish_snapshot(staged_destination, destination)
    return manifest


def _refuse_existing_destination(destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(
            f"refusing silent regeneration because output path already exists: {destination}"
        )


def _load_source(
    source: MixtureSourceSpec,
    spec_dir: str | Path | None,
    consumed: SplitName,
) -> _LoadedSource:
    base = Path(spec_dir).expanduser() if spec_dir is not None else Path.cwd()
    manifest_path = _resolve_input(source.split_manifest, base)
    statistics_path = _resolve_input(source.token_statistics, base)
    ledger_path = _resolve_input(source.token_ledger, base)
    manifest = validate_split_manifest(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    statistics, statistics_sha256 = load_token_statistics(statistics_path)
    ledger, ledger_sha256 = load_token_ledger(ledger_path)
    _require_sidecar_identity(manifest, manifest_sha256, statistics, ledger)
    require_ledger_matches_statistics(ledger, statistics)
    records = _consumed_records(source.source_id, manifest_path, manifest, ledger, consumed)
    return _LoadedSource(
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


def _resolve_input(value: str, spec_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = spec_dir / candidate
    return resolve_existing_path(str(candidate), must_be_file=True)


def _require_sidecar_identity(
    manifest: SplitManifest,
    manifest_sha256: str,
    statistics: TokenStatistics,
    ledger: TokenLedger,
) -> None:
    if statistics.split_manifest_sha256 != manifest_sha256:
        raise ValueError(
            "token statistics split-manifest digest does not match the frozen manifest"
        )
    if ledger.split_manifest_sha256 != manifest_sha256:
        raise ValueError("token ledger split-manifest digest does not match the frozen manifest")
    expected = {split: manifest.splits[split].sha256 for split in SPLIT_NAMES}
    if statistics.source_split_sha256 != expected:
        raise ValueError("token statistics source-split digests do not match the frozen manifest")
    if ledger.source_split_sha256 != expected:
        raise ValueError("token ledger source-split digests do not match the frozen manifest")


def _consumed_records(
    source_id: str,
    manifest_path: Path,
    manifest: SplitManifest,
    ledger: TokenLedger,
    consumed: SplitName,
) -> tuple[SelectableRecord, ...]:
    token_by_id = {
        record.canonical_id: record for record in ledger.records if record.split == consumed
    }
    artifact = manifest.splits[consumed]
    rows = tuple(iter_materialized_records(manifest_path, manifest, consumed))
    if len(rows) != len(artifact.members):
        raise ValueError(
            f"{source_id}: frozen {consumed} membership does not match materialized rows"
        )
    if set(token_by_id) != {member.canonical_id for member in artifact.members}:
        raise ValueError(f"{source_id}: token ledger does not cover the consumed {consumed} split")
    return tuple(
        SelectableRecord(
            source_id=source_id,
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


def _require_tokenizer_agreement(sources: Sequence[_LoadedSource]) -> MixtureTokenizerPin:
    pin = tokenizer_pin_from_statistics(sources[0].statistics)
    pin_fields = (
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_sha256",
        "serializer_id",
        "serializer_version",
        "serializer_sha256",
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


def _require_lineage_isolation(
    sources: Sequence[_LoadedSource], spec: MixtureCompositionSpec
) -> None:
    owners = _IdentityOwners()
    arm = spec.policy.experiment_arm
    for source in sources:
        for split in SPLIT_NAMES:
            for member in source.manifest.splits[split].members:
                _observe_identity(owners, source.spec.source_id, split, arm, member)
    _reject_reserved_identities(sources, spec)


def _reject_reserved_identities(
    sources: Sequence[_LoadedSource], spec: MixtureCompositionSpec
) -> None:
    reserved_arm = spec.reserved_experiment_arm or "reserved"
    reserved_lineages = set(spec.reserved_lineage_ids)
    reserved_groups = set(spec.reserved_group_ids)
    for source in sources:
        for record in source.records:
            if record.lineage_id in reserved_lineages:
                raise ValueError(
                    "lineage groups cannot cross experiment arms: "
                    f"{record.lineage_id} reserved for {reserved_arm}"
                )
            if record.group_id is not None and record.group_id in reserved_groups:
                raise ValueError(
                    "lineage groups cannot cross experiment arms: "
                    f"group {record.group_id} reserved for {reserved_arm}"
                )


def _observe_identity(
    owners: _IdentityOwners,
    source_id: str,
    split: SplitName,
    arm: str,
    member: Any,
) -> None:
    values = {
        "lineage": member.lineage_id,
        "canonical": member.canonical_id,
        "content": member.content_sha256,
        "group": member.group_id,
    }
    for kind, value in values.items():
        if value is None:
            continue
        previous = owners.owners[kind].get(value)
        if previous is None:
            owners.owners[kind][value] = (source_id, split, arm)
            continue
        _reject_identity_collision(kind, value, previous, (source_id, split, arm))


def _reject_identity_collision(
    kind: str,
    value: str,
    previous: tuple[str, SplitName, str],
    current: tuple[str, SplitName, str],
) -> None:
    prev_source, prev_split, prev_arm = previous
    source_id, split, arm = current
    if prev_split != split:
        raise ValueError(
            "lineage groups cannot cross train/validation/held-out: "
            f"{kind} {value} in {prev_source}:{prev_split} and {source_id}:{split}"
        )
    if prev_arm != arm:
        raise ValueError(
            f"lineage groups cannot cross experiment arms: {kind} {value} in {prev_arm} and {arm}"
        )
    if prev_source != source_id:
        label = "duplicate-lineage" if kind == "lineage" else f"duplicate {kind}"
        raise ValueError(f"{label}: {value} in {prev_source} and {source_id}")


def _select_source(
    source: _LoadedSource,
    quota: int,
    spec: MixtureCompositionSpec,
) -> SourceSelection:
    groups = _atomic_groups(source.records, source.spec.source_id)
    return select_groups_for_quota(groups, quota, spec.policy, source.spec.source_id)


def _atomic_groups(records: Sequence[SelectableRecord], source_id: str) -> tuple[MixtureGroup, ...]:
    source_records = tuple(
        SourceRecord(row={}, raw_line=b"", member=_member_from_selectable(record))
        for record in records
    )
    groups = []
    for component in _atomic_components(source_records):
        members = tuple(records[index] for index in component)
        truncated = any(record.truncated for record in members)
        accepted = 0 if truncated else sum(record.accepted_tokens for record in members)
        groups.append(
            MixtureGroup(
                source_id=source_id,
                anchor=_component_anchor(source_records, component),
                records=members,
                accepted_tokens=accepted,
                truncated=truncated,
            )
        )
    return tuple(groups)


def _member_from_selectable(record: SelectableRecord) -> SplitMember:
    return SplitMember(
        canonical_id=record.canonical_id,
        lineage_id=record.lineage_id,
        group_id=record.group_id,
        source_coordinate=record.source_coordinate,
        content_sha256=record.content_sha256,
        raw_line_sha256="0" * 64,
        materialized_line_sha256="0" * 64,
    )


def _write_mixture_artifact(
    destination: Path,
    selections: Sequence[SourceSelection],
) -> MixtureArtifact:
    selected = sorted(
        (record for selection in selections for record in selection.records),
        key=lambda record: (record.source_id, record.canonical_id),
    )
    if not selected:
        raise ValueError(
            "mixture selected no records; lower the budget or inspect underfilled sources"
        )
    relative_path = "mixture.jsonl"
    payload = b"".join(record.line for record in selected)
    exclusive_write(destination / relative_path, payload)
    return MixtureArtifact(
        path=relative_path,
        sha256=sha256_file(destination / relative_path),
        record_count=len(selected),
        accepted_tokens=sum(record.accepted_tokens for record in selected),
    )


def _build_manifest(
    spec: MixtureCompositionSpec,
    tokenizer: MixtureTokenizerPin,
    loaded: Sequence[_LoadedSource],
    selections: Sequence[SourceSelection],
    artifact: MixtureArtifact,
) -> MixtureManifest:
    by_source = {selection.source_id: selection for selection in selections}
    source_results = tuple(
        _source_result(source, by_source[source.spec.source_id]) for source in loaded
    )
    selected = tuple(
        MixtureMember(
            source_id=record.source_id,
            canonical_id=record.canonical_id,
            lineage_id=record.lineage_id,
            group_id=record.group_id,
            source_coordinate=record.source_coordinate,
            accepted_tokens=record.accepted_tokens,
            content_sha256=record.content_sha256,
        )
        for selection in selections
        for record in selection.records
    )
    selected = tuple(sorted(selected, key=lambda member: (member.source_id, member.canonical_id)))
    exclusions = tuple(
        exclusion
        for selection in selections
        for exclusion in sorted(
            selection.exclusions,
            key=lambda item: (item.source_id, item.canonical_id, item.reason),
        )
    )
    return MixtureManifest(
        policy=spec.policy,
        tokenizer=tokenizer,
        sources=source_results,
        artifact=artifact,
        selected=selected,
        exclusions=exclusions,
        lineage_audit=MixtureLineageAudit(),
        limitations=(
            "Underfilled sources are reported and never silently oversampled.",
            "Deterministic gates do not claim semantic near-duplicate detection.",
            "Accepted-token budgets are tokenizer-revision specific.",
        ),
    )


def _source_result(source: _LoadedSource, selection: SourceSelection) -> MixtureSourceResult:
    return MixtureSourceResult(
        source_id=source.spec.source_id,
        provenance_class=source.spec.provenance_class,
        repository=source.manifest.source.repository,
        revision=source.manifest.source.revision,
        dataset_version=source.manifest.source.dataset_version,
        source_sha256=source.manifest.source.sha256,
        split_manifest_sha256=source.manifest_sha256,
        token_statistics_sha256=source.statistics_sha256,
        token_ledger_sha256=source.ledger_sha256,
        split_artifact_sha256=source.manifest.splits[source.consumed_split].sha256,
        weight=source.spec.weight,
        quota_tokens=selection.quota_tokens,
        available_accepted_tokens=selection.available_accepted_tokens,
        accepted_tokens=selection.accepted_tokens,
        example_count=len(selection.records),
        excluded_example_count=len(selection.exclusions),
        underfilled=selection.underfilled,
        shortfall_tokens=selection.shortfall_tokens,
    )
