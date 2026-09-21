import pytest

from agoge_forger.mixture_contract import (
    allocate_token_quotas,
    compose_mixture,
    write_token_ledger,
)
from agoge_forger.mixture_ledger import load_token_ledger, require_ledger_matches_statistics
from agoge_forger.split_contract import (
    SerializerBinding,
    TokenizerBinding,
    TokenStatisticsDerivation,
    TokenStatisticsSpec,
    sha256_bytes,
    validate_split_manifest,
    write_token_statistics,
)
from tests.mixture_composer_fixtures import (
    MODEL_REVISION,
    ComposeSpecOptions,
    SidecarWriteOptions,
    compose_spec,
    freeze_source,
    prepare_sources,
    write_sidecars,
)
from tests.test_split_contract import CharacterTokenizer, plain_text_serializer


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
    sources = prepare_sources(tmp_path)
    output = tmp_path / "mixture"
    manifest = compose_mixture(compose_spec(sources, ComposeSpecOptions(budget=30)), output)

    assert manifest.artifact.accepted_tokens == 30
    assert all(source.accepted_tokens == 10 for source in manifest.sources)
    assert all(not source.underfilled for source in manifest.sources)
    assert manifest.artifact.record_count == 3
    assert (output / "mixture.jsonl").is_file()
    assert "Accepted-token budget: 30" in (output / "mixture_report.md").read_text()


def test_source_underfill_is_reported_and_not_oversampled(tmp_path):
    sources = prepare_sources(tmp_path)
    output = tmp_path / "mixture"
    manifest = compose_mixture(compose_spec(sources, ComposeSpecOptions(budget=10_000)), output)

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
    synthetic = freeze_source(tmp_path, "synthetic")
    public = freeze_source(
        tmp_path,
        "public",
        lineage=lambda index: (
            "synthetic-lineage-000" if index == 0 else f"public-lineage-{index:03d}"
        ),
    )
    write_sidecars(synthetic)
    write_sidecars(public)

    with pytest.raises(
        ValueError, match="duplicate-lineage|cannot cross train/validation/held-out"
    ):
        compose_mixture(
            compose_spec(
                (("synthetic", synthetic, 1), ("public", public, 1)),
                ComposeSpecOptions(budget=20),
            ),
            tmp_path / "mixture",
        )
    assert not (tmp_path / "mixture").exists()


def test_tokenizer_revision_mismatch_is_rejected(tmp_path):
    synthetic = freeze_source(tmp_path, "synthetic")
    public = freeze_source(tmp_path, "public")
    write_sidecars(synthetic)
    write_sidecars(public, SidecarWriteOptions(tokenizer_revision="e" * 40))

    with pytest.raises(ValueError, match="tokenizer-revision mismatch"):
        compose_mixture(
            compose_spec(
                (("synthetic", synthetic, 1), ("public", public, 1)),
                ComposeSpecOptions(budget=20),
            ),
            tmp_path / "mixture",
        )


def test_lineage_groups_are_not_split_to_fill_a_quota(tmp_path):
    snapshot = freeze_source(
        tmp_path,
        "synthetic",
        lineage=lambda index: f"synthetic-lineage-{index // 2:03d}",
    )
    write_sidecars(snapshot)
    spec_under = compose_spec(
        (("synthetic", snapshot, 1),),
        ComposeSpecOptions(budget=15, experiment_arm="B"),
    )
    with pytest.raises(ValueError, match="mixture selected no records"):
        compose_mixture(spec_under, tmp_path / "mixture-under")

    manifest = compose_mixture(
        compose_spec(
            (("synthetic", snapshot, 1),),
            ComposeSpecOptions(budget=20, experiment_arm="B"),
        ),
        tmp_path / "mixture-pair",
    )
    assert manifest.artifact.accepted_tokens == 20
    assert manifest.artifact.record_count == 2
    assert len({member.lineage_id for member in manifest.selected}) == 1


def test_reserved_lineage_cannot_cross_experiment_arms(tmp_path):
    snapshot = freeze_source(tmp_path, "synthetic")
    write_sidecars(snapshot)
    train_lineage = (
        validate_split_manifest(snapshot / "split_manifest.json").splits["train"].members[0]
    )
    reserved = train_lineage.lineage_id

    with pytest.raises(ValueError, match="cannot cross experiment arms"):
        compose_mixture(
            compose_spec(
                (("synthetic", snapshot, 1),),
                ComposeSpecOptions(
                    budget=10,
                    reserved_lineage_ids=(reserved,),
                    reserved_experiment_arm="E",
                ),
            ),
            tmp_path / "mixture",
        )


def test_deterministic_rebuild_is_byte_identical(tmp_path):
    sources = prepare_sources(tmp_path)
    spec = compose_spec(sources, ComposeSpecOptions(budget=30))
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = compose_mixture(spec, first)
    second_manifest = compose_mixture(spec, second)

    assert first_manifest == second_manifest
    for name in ("mixture.jsonl", "mixture_manifest.json", "mixture_report.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_compose_mixture_refuses_silent_regeneration(tmp_path):
    snapshot = freeze_source(tmp_path, "synthetic")
    write_sidecars(snapshot)
    spec = compose_spec((("synthetic", snapshot, 1),), ComposeSpecOptions(budget=10))
    compose_mixture(spec, tmp_path / "mixture")

    with pytest.raises(FileExistsError, match="refusing silent regeneration"):
        compose_mixture(spec, tmp_path / "mixture")


def test_truncated_records_are_excluded_not_oversampled(tmp_path):
    snapshot = freeze_source(tmp_path, "synthetic")
    train_members = (
        validate_split_manifest(snapshot / "split_manifest.json").splits["train"].members
    )
    truncated_id = train_members[0].canonical_id
    write_sidecars(snapshot, SidecarWriteOptions(truncated_ids=frozenset({truncated_id})))
    manifest = compose_mixture(
        compose_spec(
            (("synthetic", snapshot, 1),),
            ComposeSpecOptions(budget=10, experiment_arm="B"),
        ),
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
    snapshot = freeze_source(tmp_path, "synthetic")
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
