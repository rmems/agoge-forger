"""Lineage, canonical, group, and content isolation for mixture sources."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .mixture_allocate import SelectableRecord
from .mixture_load import LoadedSource
from .mixture_schema import MixtureCompositionSpec
from .split_contract import SPLIT_NAMES, SplitName


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


def require_lineage_isolation(
    sources: Sequence[LoadedSource], spec: MixtureCompositionSpec
) -> None:
    owners = _IdentityOwners()
    arm = spec.policy.experiment_arm
    for source in sources:
        for split in SPLIT_NAMES:
            for member in source.manifest.splits[split].members:
                _observe_identity(owners, source.spec.source_id, split, arm, member)
    reject_reserved_identities(sources, spec)


def reject_reserved_identities(
    sources: Sequence[LoadedSource], spec: MixtureCompositionSpec
) -> None:
    reserved_arm = spec.reserved_experiment_arm or "reserved"
    reserved = {
        "lineage": set(spec.reserved_lineage_ids),
        "group": set(spec.reserved_group_ids),
        "canonical": set(spec.reserved_canonical_ids),
        "content": set(spec.reserved_content_sha256),
    }
    for source in sources:
        for record in source.records:
            reject_reserved_record(record, reserved, reserved_arm)


def reject_reserved_record(
    record: SelectableRecord,
    reserved: dict[str, set[str]],
    reserved_arm: str,
) -> None:
    values = {
        "lineage": record.lineage_id,
        "group": record.group_id,
        "canonical": record.canonical_id,
        "content": record.content_sha256,
    }
    for kind, value in values.items():
        if value is not None and value in reserved[kind]:
            raise ValueError(
                "lineage groups cannot cross experiment arms: "
                f"{kind} {value} reserved for {reserved_arm}"
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
