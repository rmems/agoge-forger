"""Deterministic materialization of exact accepted-token mixtures."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
from .mixture_identity import require_lineage_isolation
from .mixture_load import LoadedSource, load_source, require_tokenizer_agreement
from .mixture_result_schema import (
    MixtureArtifact,
    MixtureLineageAudit,
    MixtureManifest,
    MixtureMember,
    MixtureSourceResult,
)
from .mixture_schema import MixtureCompositionSpec, MixtureTokenizerPin
from .split_contract import SplitMember
from .split_materialize import (
    SourceRecord,
    _atomic_components,
    _component_anchor,
    _publish_snapshot,
    exclusive_write,
)
from .split_schema import sha256_file


@dataclass(frozen=True)
class _MixturePublication:
    destination: Path
    spec: MixtureCompositionSpec
    tokenizer: MixtureTokenizerPin
    loaded: Sequence[LoadedSource]
    selections: Sequence[SourceSelection]


@dataclass(frozen=True)
class _ManifestBuild:
    publication: _MixturePublication
    artifact: MixtureArtifact


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
    loaded = tuple(load_source(source, spec_dir, consumed) for source in spec.sources)
    tokenizer = require_tokenizer_agreement(loaded)
    require_lineage_isolation(loaded, spec)
    source_ids = tuple(source.spec.source_id for source in loaded)
    weights = {source.spec.source_id: source.spec.weight for source in loaded}
    quotas = allocate_token_quotas(spec.policy.accepted_token_budget, weights, source_ids)
    selections = tuple(
        _select_source(source, quotas[source.spec.source_id], spec) for source in loaded
    )
    publication = _MixturePublication(destination, spec, tokenizer, loaded, selections)
    return _publish_mixture(publication)


def _publish_mixture(publication: _MixturePublication) -> MixtureManifest:
    destination = publication.destination
    staging_parent = nearest_existing_output_ancestor(destination)
    require_rename_noreplace_support(staging_parent)
    with tempfile.TemporaryDirectory(
        prefix=".agoge-mixture-staging-",
        dir=staging_parent,
    ) as staging_dir:
        staged_destination = Path(staging_dir) / "snapshot"
        staged_destination.mkdir()
        artifact = _write_mixture_artifact(staged_destination, publication.selections)
        manifest = _build_manifest(_ManifestBuild(publication, artifact))
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
    _refuse_symlinked_ancestors(destination)


def _refuse_symlinked_ancestors(destination: Path) -> None:
    current = destination.expanduser().absolute().parent
    while True:
        if current.is_symlink():
            raise ValueError(f"refusing to publish through a symlinked ancestor: {current}")
        if current.parent == current:
            return
        current = current.parent


def _select_source(
    source: LoadedSource,
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


def _build_manifest(build: _ManifestBuild) -> MixtureManifest:
    publication = build.publication
    spec = publication.spec
    tokenizer = publication.tokenizer
    loaded = publication.loaded
    selections = publication.selections
    artifact = build.artifact
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


def _source_result(source: LoadedSource, selection: SourceSelection) -> MixtureSourceResult:
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
