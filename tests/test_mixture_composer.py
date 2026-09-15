from pathlib import Path

import pytest

from agoge_forger.mixture_contract import (
    MixtureCompositionSpec,
    MixturePolicy,
    MixtureSourceSpec,
    allocate_token_quotas,
    compose_mixture,
    write_token_ledger,
)
from agoge_forger.mixture_ledger import load_token_ledger, require_ledger_matches_statistics
from agoge_forger.mixture_schema import TokenLedger, TokenLedgerRecord
from agoge_forger.split_contract import (
    SPLIT_NAMES,
    SerializerBinding,
    SplitMaterializationSpec,
    SplitPolicy,
    TokenizerBinding,
    TokenStatistics,
    TokenStatisticsDerivation,
    TokenStatisticsSpec,
    TokenStatSplit,
    canonical_json_bytes,
    materialize_split,
    sha256_bytes,
    sha256_file,
    validate_split_manifest,
    write_token_statistics,
)
from tests.test_split_contract import CharacterTokenizer, plain_text_serializer

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


def _write_source(path: Path, prefix: str, count: int, *, lineage=None) -> None:
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


def _freeze_source(tmp_path: Path, source_id: str, count: int = 36, *, lineage=None) -> Path:
    source = tmp_path / f"{source_id}.jsonl"
    output = tmp_path / f"{source_id}-split"
    _write_source(source, source_id, count, lineage=lineage)
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


def _write_sidecars(
    snapshot: Path,
    *,
    tokens: int = 10,
    tokenizer_revision: str = TOKENIZER_REVISION,
    truncated_ids: frozenset[str] = frozenset(),
) -> None:
    manifest_path = snapshot / "split_manifest.json"
    manifest = validate_split_manifest(manifest_path)
    records: list[TokenLedgerRecord] = []
    splits: dict[str, TokenStatSplit] = {}
    for split in SPLIT_NAMES:
        lengths = []
        truncated_count = 0
        for member in manifest.splits[split].members:
            truncated = member.canonical_id in truncated_ids
            lengths.append(tokens)
            truncated_count += int(truncated)
            records.append(
                TokenLedgerRecord(
                    canonical_id=member.canonical_id,
                    split=split,
                    accepted_tokens=tokens,
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
        tokenizer_revision=tokenizer_revision,
        tokenizer_sha256=TOKENIZER_SHA256,
        serializer_id="plain-text",
        serializer_version="1",
        serializer_sha256=SERIALIZER_SHA256,
        context_limit=64,
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


def _source_spec(source_id: str, snapshot: Path, weight: int) -> MixtureSourceSpec:
    return MixtureSourceSpec(
        source_id=source_id,
        provenance_class=PROVENANCE[source_id],
        weight=weight,
        split_manifest=str(snapshot / "split_manifest.json"),
        token_statistics=str(snapshot / "token_stats.json"),
        token_ledger=str(snapshot / "token_ledger.json"),
    )


def _compose_spec(
    sources: tuple[tuple[str, Path, int], ...],
    *,
    budget: int,
    experiment_arm: str = "F",
    reserved_lineage_ids: tuple[str, ...] = (),
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
            _source_spec(source_id, snapshot, weight) for source_id, snapshot, weight in sources
        ),
        reserved_lineage_ids=reserved_lineage_ids,
        reserved_experiment_arm=reserved_experiment_arm,
    )


def _prepare_sources(
    tmp_path: Path, names: tuple[str, ...] = ("synthetic", "public", "prometheus")
):
    prepared = []
    for name in names:
        snapshot = _freeze_source(tmp_path, name)
        _write_sidecars(snapshot, tokens=10)
        prepared.append((name, snapshot, 1))
    return tuple(prepared)


def test_allocate_token_quotas_uses_largest_remainder_rounding():
    quotas = allocate_token_quotas(
        100,
        {"prometheus": 1, "public": 1, "synthetic": 1},
        ("prometheus", "public", "synthetic"),
    )
    assert sum(quotas.values()) == 100
    assert quotas == {"prometheus": 34, "public": 33, "synthetic": 33}

    weighted = allocate_token_quotas(100, {"public": 1, "prometheus": 2}, ("public", "prometheus"))
    assert weighted == {"public": 33, "prometheus": 67}


def test_exact_budget_mixture_from_frozen_fixtures(tmp_path):
    sources = _prepare_sources(tmp_path)
    output = tmp_path / "mixture"
    manifest = compose_mixture(_compose_spec(sources, budget=30), output)

    assert manifest.artifact.accepted_tokens == 30
    assert all(source.accepted_tokens == 10 for source in manifest.sources)
    assert all(not source.underfilled for source in manifest.sources)
    assert manifest.artifact.record_count == 3
    assert (output / "mixture.jsonl").is_file()
    assert "Accepted-token budget: 30" in (output / "mixture_report.md").read_text()


def test_source_underfill_is_reported_and_not_oversampled(tmp_path):
    sources = _prepare_sources(tmp_path)
    output = tmp_path / "mixture"
    manifest = compose_mixture(_compose_spec(sources, budget=10_000), output)

    assert all(source.underfilled for source in manifest.sources)
    assert all(source.shortfall_tokens > 0 for source in manifest.sources)
    for source, (_, snapshot, _) in zip(manifest.sources, sources, strict=True):
        train_count = (
            validate_split_manifest(snapshot / "split_manifest.json").splits["train"].record_count
        )
        assert source.example_count == train_count
        assert source.accepted_tokens == 10 * train_count
        assert source.accepted_tokens == source.available_accepted_tokens
        assert source.accepted_tokens < source.quota_tokens
    assert manifest.artifact.accepted_tokens < 10_000


def test_duplicate_lineage_is_rejected(tmp_path):
    synthetic = _freeze_source(tmp_path, "synthetic")
    public = _freeze_source(
        tmp_path,
        "public",
        lineage=lambda index: (
            "synthetic-lineage-000" if index == 0 else f"public-lineage-{index:03d}"
        ),
    )
    _write_sidecars(synthetic, tokens=10)
    _write_sidecars(public, tokens=10)

    with pytest.raises(
        ValueError, match="duplicate-lineage|cannot cross train/validation/held-out"
    ):
        compose_mixture(
            _compose_spec(
                (("synthetic", synthetic, 1), ("public", public, 1)),
                budget=20,
            ),
            tmp_path / "mixture",
        )
    assert not (tmp_path / "mixture").exists()


def test_tokenizer_revision_mismatch_is_rejected(tmp_path):
    synthetic = _freeze_source(tmp_path, "synthetic")
    public = _freeze_source(tmp_path, "public")
    _write_sidecars(synthetic, tokens=10)
    _write_sidecars(public, tokens=10, tokenizer_revision="e" * 40)

    with pytest.raises(ValueError, match="tokenizer-revision mismatch"):
        compose_mixture(
            _compose_spec(
                (("synthetic", synthetic, 1), ("public", public, 1)),
                budget=20,
            ),
            tmp_path / "mixture",
        )


def test_lineage_groups_are_not_split_to_fill_a_quota(tmp_path):
    snapshot = _freeze_source(
        tmp_path,
        "synthetic",
        lineage=lambda index: f"synthetic-lineage-{index // 2:03d}",
    )
    _write_sidecars(snapshot, tokens=10)
    spec_under = _compose_spec((("synthetic", snapshot, 1),), budget=15, experiment_arm="B")
    with pytest.raises(ValueError, match="mixture selected no records"):
        compose_mixture(spec_under, tmp_path / "mixture-under")

    manifest = compose_mixture(
        _compose_spec((("synthetic", snapshot, 1),), budget=20, experiment_arm="B"),
        tmp_path / "mixture-pair",
    )
    assert manifest.artifact.accepted_tokens == 20
    assert manifest.artifact.record_count == 2
    assert len({member.lineage_id for member in manifest.selected}) == 1


def test_reserved_lineage_cannot_cross_experiment_arms(tmp_path):
    snapshot = _freeze_source(tmp_path, "synthetic")
    _write_sidecars(snapshot, tokens=10)
    train_lineage = (
        validate_split_manifest(snapshot / "split_manifest.json").splits["train"].members[0]
    )
    reserved = train_lineage.lineage_id

    with pytest.raises(ValueError, match="cannot cross experiment arms"):
        compose_mixture(
            _compose_spec(
                (("synthetic", snapshot, 1),),
                budget=10,
                reserved_lineage_ids=(reserved,),
                reserved_experiment_arm="E",
            ),
            tmp_path / "mixture",
        )


def test_deterministic_rebuild_is_byte_identical(tmp_path):
    sources = _prepare_sources(tmp_path)
    spec = _compose_spec(sources, budget=30)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = compose_mixture(spec, first)
    second_manifest = compose_mixture(spec, second)

    assert first_manifest == second_manifest
    for name in ("mixture.jsonl", "mixture_manifest.json", "mixture_report.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_compose_mixture_refuses_silent_regeneration(tmp_path):
    snapshot = _freeze_source(tmp_path, "synthetic")
    _write_sidecars(snapshot, tokens=10)
    spec = _compose_spec((("synthetic", snapshot, 1),), budget=10)
    compose_mixture(spec, tmp_path / "mixture")

    with pytest.raises(FileExistsError, match="refusing silent regeneration"):
        compose_mixture(spec, tmp_path / "mixture")


def test_truncated_records_are_excluded_not_oversampled(tmp_path):
    snapshot = _freeze_source(tmp_path, "synthetic")
    train_members = (
        validate_split_manifest(snapshot / "split_manifest.json").splits["train"].members
    )
    truncated_id = train_members[0].canonical_id
    _write_sidecars(snapshot, tokens=10, truncated_ids=frozenset({truncated_id}))
    manifest = compose_mixture(
        _compose_spec((("synthetic", snapshot, 1),), budget=10, experiment_arm="B"),
        tmp_path / "mixture-truncated",
    )
    selected_ids = {member.canonical_id for member in manifest.selected}
    assert truncated_id not in selected_ids
    assert any(
        item.canonical_id == truncated_id and item.reason == "truncated"
        for item in manifest.exclusions
    )
    assert manifest.artifact.accepted_tokens == 10


def test_write_token_ledger_matches_token_statistics(tmp_path):
    snapshot = _freeze_source(tmp_path, "synthetic")
    tokenizer = TokenizerBinding(implementation=CharacterTokenizer())
    serializer = SerializerBinding(implementation=plain_text_serializer)
    derivation = TokenStatisticsDerivation(
        tokenizer=tokenizer,
        serializer=serializer,
        spec=TokenStatisticsSpec(
            model_id="fake/model-family-a",
            model_revision=MODEL_REVISION,
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_revision=tokenizer.tokenizer_revision,
            tokenizer_sha256=tokenizer.tokenizer_sha256,
            serializer_id=serializer.serializer_id,
            serializer_version=serializer.serializer_version,
            serializer_sha256=serializer.serializer_sha256,
            context_limit=64,
        ),
    )
    statistics = write_token_statistics(
        snapshot / "split_manifest.json",
        snapshot / "character-token-stats.json",
        derivation,
    )
    ledger = write_token_ledger(
        snapshot / "split_manifest.json",
        snapshot / "character-token-ledger.json",
        derivation,
    )
    require_ledger_matches_statistics(ledger, statistics)
    loaded, digest = load_token_ledger(snapshot / "character-token-ledger.json")
    assert loaded == ledger
    assert digest == sha256_bytes((snapshot / "character-token-ledger.json").read_bytes())
