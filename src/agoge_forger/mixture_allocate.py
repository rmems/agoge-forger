"""Deterministic quota allocation and first-fit group selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .mixture_result_schema import MixtureExclusion
from .mixture_schema import ExclusionReason, MixturePolicy
from .split_schema import sha256_bytes


@dataclass(frozen=True)
class SelectableRecord:
    source_id: str
    canonical_id: str
    lineage_id: str
    group_id: str | None
    source_coordinate: str
    content_sha256: str
    accepted_tokens: int
    truncated: bool
    line: bytes


@dataclass(frozen=True)
class MixtureGroup:
    source_id: str
    anchor: str
    records: tuple[SelectableRecord, ...]
    accepted_tokens: int
    truncated: bool


@dataclass(frozen=True)
class SourceSelection:
    source_id: str
    quota_tokens: int
    available_accepted_tokens: int
    accepted_tokens: int
    shortfall_tokens: int
    groups: tuple[MixtureGroup, ...]
    exclusions: tuple[MixtureExclusion, ...]

    @property
    def underfilled(self) -> bool:
        return self.shortfall_tokens > 0

    @property
    def records(self) -> tuple[SelectableRecord, ...]:
        return tuple(record for group in self.groups for record in group.records)


def allocate_token_quotas(
    budget: int,
    weights: Mapping[str, int],
    source_ids: Sequence[str],
) -> dict[str, int]:
    """Allocate an exact integer budget with largest-remainder (Hamilton) rounding."""

    if budget < 1:
        raise ValueError("accepted-token budget must be positive")
    if set(weights) != set(source_ids):
        raise ValueError("quota weights must cover each source_id exactly once")
    if any(weights[source_id] <= 0 for source_id in source_ids):
        raise ValueError("all source weights must be positive")
    total_weight = sum(weights[source_id] for source_id in source_ids)
    floors: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    for source_id in source_ids:
        floor, remainder = divmod(budget * weights[source_id], total_weight)
        floors[source_id] = floor
        remainders.append((remainder, source_id))
    leftover = budget - sum(floors.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for source_id in (item[1] for item in remainders[:leftover]):
        floors[source_id] += 1
    if sum(floors.values()) != budget:
        raise ValueError("source quotas must sum to the accepted-token budget")
    return floors


def select_groups_for_quota(
    groups: Sequence[MixtureGroup],
    quota: int,
    policy: MixturePolicy,
    source_id: str,
) -> SourceSelection:
    ordered = sorted(
        groups,
        key=lambda group: (_selection_digest(policy, source_id, group.anchor), group.anchor),
    )
    selected: list[MixtureGroup] = []
    exclusions: list[MixtureExclusion] = []
    remaining = quota
    available = 0
    for group in ordered:
        if group.truncated:
            exclusions.extend(_group_exclusions(group, "truncated"))
            continue
        available += group.accepted_tokens
        reason = _quota_exclusion_reason(group.accepted_tokens, quota, remaining)
        if reason is not None:
            exclusions.extend(_group_exclusions(group, reason))
            continue
        selected.append(group)
        remaining -= group.accepted_tokens
    accepted = quota - remaining
    return SourceSelection(
        source_id=source_id,
        quota_tokens=quota,
        available_accepted_tokens=available,
        accepted_tokens=accepted,
        shortfall_tokens=remaining,
        groups=tuple(selected),
        exclusions=tuple(exclusions),
    )


def _quota_exclusion_reason(tokens: int, quota: int, remaining: int) -> ExclusionReason | None:
    if tokens > quota:
        return "exceeds-source-quota"
    if tokens > remaining:
        return "quota-remainder"
    return None


def _group_exclusions(group: MixtureGroup, reason: ExclusionReason) -> tuple[MixtureExclusion, ...]:
    return tuple(
        MixtureExclusion(
            source_id=group.source_id,
            canonical_id=record.canonical_id,
            lineage_id=record.lineage_id,
            reason=reason,
        )
        for record in group.records
    )


def _selection_digest(policy: MixturePolicy, source_id: str, anchor: str) -> str:
    material = (
        f"{policy.algorithm_version}\0{policy.seed}\0{policy.salt}\0{source_id}\0{anchor}".encode()
    )
    return sha256_bytes(material)
