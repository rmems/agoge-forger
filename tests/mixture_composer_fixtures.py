"""Shared frozen-fixture builders for mixture composer tests."""

from dataclasses import dataclass
from pathlib import Path

from agoge_forger.mixture_contract import MixtureCompositionSpec, MixturePolicy, MixtureSourceSpec
from agoge_forger.mixture_ledger_schema import TokenLedger, TokenLedgerRecord
from agoge_forger.split_contract import (
    SPLIT_NAMES,
    SplitMaterializationSpec,
    SplitPolicy,
    TokenStatistics,
    TokenStatSplit,
    canonical_json_bytes,
    materialize_split,
    sha256_file,
    validate_split_manifest,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"
MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40
TOKENIZER_SHA256 = "d" * 64
SERIALIZER_SHA256 = "c" * 64
PROVENANCE = {
    "synthetic": "SYNTHETIC_FACTORY",
    "public": "EXTERNAL_PUBLIC_SWE",
    "prometheus": "PROMETHEUS_REAL",
}


@dataclass(frozen=True)
class SidecarWriteOptions:
    tokens: int = 10
    tokenizer_revision: str = TOKENIZER_REVISION
    context_limit: int = 64
    truncated_ids: frozenset[str] = frozenset()


def write_source(path: Path, prefix: str, count: int, *, lineage=None) -> None:
    rows = [
        {
            "canonical_id": f"{prefix}-{index:03d}",
            "lineage_id": lineage(index)
            if lineage is not None
            else f"{prefix}-lineage-{index:03d}",
            "text": f"{prefix} sample {index} unique evidence {index * 17}.",
        }
        for index in range(count)
    ]
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def freeze_source(tmp_path: Path, source_id: str, count: int = 36, *, lineage=None) -> Path:
    source = tmp_path / f"{source_id}.jsonl"
    output = tmp_path / f"{source_id}-split"
    write_source(source, source_id, count, lineage=lineage)
    materialize_split(
        source,
        output,
        SplitMaterializationSpec(
            source_repository=f"rmems/{source_id}",
            source_revision=REVISION,
            dataset_version=f"{source_id}-v1",
            source_path=f"data/{source_id}.jsonl",
            split_policy=SplitPolicy(
                seed=20260915,
                salt="agoge-issue-103-mixture-v1",
                weights={"train": 6, "validation": 2, "held_out": 2},
            ),
        ),
    )
    return output


def write_sidecars(
    snapshot: Path,
    options: SidecarWriteOptions | None = None,
) -> None:
    opts = options or SidecarWriteOptions()
    manifest_path = snapshot / "split_manifest.json"
    manifest = validate_split_manifest(manifest_path)
    records: list[TokenLedgerRecord] = []
    splits: dict[str, TokenStatSplit] = {}
    for split in SPLIT_NAMES:
        lengths = []
        truncated_count = 0
        for member in manifest.splits[split].members:
            truncated = member.canonical_id in opts.truncated_ids
            lengths.append(opts.tokens)
            truncated_count += int(truncated)
            records.append(
                TokenLedgerRecord(
                    canonical_id=member.canonical_id,
                    split=split,
                    accepted_tokens=opts.tokens,
                    truncated=truncated,
                )
            )
        splits[split] = TokenStatSplit(
            record_count=len(manifest.splits[split].members),
            total_tokens=sum(lengths),
            minimum_tokens=min(lengths),
            maximum_tokens=max(lengths),
            truncated_records=truncated_count,
        )
    statistics = TokenStatistics(
        split_manifest_sha256=sha256_file(manifest_path),
        source_split_sha256={split: manifest.splits[split].sha256 for split in SPLIT_NAMES},
        model_id="fake/model",
        model_revision=MODEL_REVISION,
        tokenizer_id="fake/tokenizer",
        tokenizer_revision=opts.tokenizer_revision,
        tokenizer_sha256=TOKENIZER_SHA256,
        serializer_id="plain-text",
        serializer_version="1",
        serializer_sha256=SERIALIZER_SHA256,
        context_limit=opts.context_limit,
        splits=splits,
    )
    ledger = TokenLedger(
        split_manifest_sha256=statistics.split_manifest_sha256,
        source_split_sha256=statistics.source_split_sha256,
        tokenizer_id=statistics.tokenizer_id,
        tokenizer_revision=statistics.tokenizer_revision,
        tokenizer_sha256=statistics.tokenizer_sha256,
        serializer_id=statistics.serializer_id,
        serializer_version=statistics.serializer_version,
        serializer_sha256=statistics.serializer_sha256,
        context_limit=statistics.context_limit,
        records=tuple(records),
    )
    (snapshot / "token_stats.json").write_bytes(
        canonical_json_bytes(statistics.model_dump(mode="json")) + b"\n"
    )
    (snapshot / "token_ledger.json").write_bytes(
        canonical_json_bytes(ledger.model_dump(mode="json")) + b"\n"
    )


def source_spec(source_id: str, snapshot: Path, weight: int) -> MixtureSourceSpec:
    return MixtureSourceSpec(
        source_id=source_id,
        provenance_class=PROVENANCE[source_id],
        weight=weight,
        split_manifest=str(snapshot / "split_manifest.json"),
        token_statistics=str(snapshot / "token_stats.json"),
        token_ledger=str(snapshot / "token_ledger.json"),
    )


def compose_spec(
    sources: tuple[tuple[str, Path, int], ...],
    *,
    budget: int,
    experiment_arm: str = "F",
    reserved_lineage_ids: tuple[str, ...] = (),
    reserved_canonical_ids: tuple[str, ...] = (),
    reserved_content_sha256: tuple[str, ...] = (),
    reserved_experiment_arm: str | None = None,
) -> MixtureCompositionSpec:
    return MixtureCompositionSpec(
        policy=MixturePolicy(
            seed=20260915,
            salt="matched-budget-v1",
            accepted_token_budget=budget,
            experiment_arm=experiment_arm,
        ),
        sources=tuple(
            source_spec(source_id, snapshot, weight) for source_id, snapshot, weight in sources
        ),
        reserved_lineage_ids=reserved_lineage_ids,
        reserved_canonical_ids=reserved_canonical_ids,
        reserved_content_sha256=reserved_content_sha256,
        reserved_experiment_arm=reserved_experiment_arm,
    )


def prepare_sources(
    tmp_path: Path, names: tuple[str, ...] = ("synthetic", "public", "prometheus")
) -> tuple[tuple[str, Path, int], ...]:
    prepared = []
    for name in names:
        snapshot = freeze_source(tmp_path, name)
        write_sidecars(snapshot)
        prepared.append((name, snapshot, 1))
    return tuple(prepared)
