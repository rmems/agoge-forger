import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agoge_forger.split_contract import (
    SPLIT_NAMES,
    SerializerBinding,
    SplitMaterializationSpec,
    SplitPolicy,
    TokenizerBinding,
    TokenStatisticsDerivation,
    TokenStatisticsSpec,
    canonical_json_bytes,
    iter_frozen_records,
    load_frozen_dataset,
    materialize_split,
    sha256_bytes,
    validate_split_manifest,
    write_token_statistics,
)


def _write_source(path: Path, count: int = 90) -> None:
    rows = [
        {
            "canonical_id": f"sample-{index:03d}",
            "lineage_id": f"lineage-{index // 2:03d}",
            "group_id": f"family-{index // 3:03d}",
            "text": f"Explain deterministic sample {index} with unique evidence {index * 17}.",
        }
        for index in range(count)
    ]
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _materialize(source: Path, output: Path):
    spec = SplitMaterializationSpec(
        source_repository="rmems/synthetic-factory",
        source_revision="0123456789abcdef0123456789abcdef01234567",
        dataset_version="curated-sft-v1",
        source_path="data/curated.jsonl",
        split_policy=SplitPolicy(
            seed=20260830,
            salt="agoge-issue-99-v1",
            weights={"train": 6, "validation": 2, "held_out": 2},
        ),
    )
    return materialize_split(source, output, spec)


def test_one_command_materializes_repeatable_three_way_split(tmp_path, run_freeze_split):
    source = tmp_path / "curated.jsonl"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_source(source)

    stdout = run_freeze_split(
        [
            "--source",
            str(source),
            "--source-path",
            "data/curated.jsonl",
            "--output-dir",
            str(first),
            "--source-repository",
            "rmems/synthetic-factory",
            "--source-revision",
            "0123456789abcdef0123456789abcdef01234567",
            "--dataset-version",
            "curated-sft-v1",
            "--seed",
            "20260830",
            "--salt",
            "agoge-issue-99-v1",
            "--train-weight",
            "6",
            "--validation-weight",
            "2",
            "--held-out-weight",
            "2",
        ]
    )
    assert "split_manifest.json" in stdout
    assert "split_report.md" in stdout

    first_manifest = validate_split_manifest(first / "split_manifest.json", source_path=source)
    second_manifest = _materialize(source, second)

    assert first_manifest == second_manifest
    assert (first / "split_manifest.json").read_bytes() == (
        second / "split_manifest.json"
    ).read_bytes()
    assert first_manifest.source.record_count == 90
    assert first_manifest.leakage_audit.status == "passed"
    assert set(first_manifest.splits) == set(SPLIT_NAMES)
    assert all(first_manifest.splits[name].record_count > 0 for name in SPLIT_NAMES)
    assert "Source coverage: 90/90" in (first / "split_report.md").read_text()

    loaded_ids = {
        row["canonical_id"]
        for split in SPLIT_NAMES
        for row in iter_frozen_records(first / "split_manifest.json", split)
    }
    assert loaded_ids == {f"sample-{index:03d}" for index in range(90)}
    train_dataset = load_frozen_dataset(first / "split_manifest.json", "train")
    assert len(train_dataset) == first_manifest.splits["train"].record_count


def test_frozen_output_refuses_silent_regeneration(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    _write_source(source)
    _materialize(source, output)

    with pytest.raises(FileExistsError, match="refusing silent regeneration"):
        _materialize(source, output)


def test_frozen_dataset_cache_identity_changes_with_manifest_and_split_digests(
    tmp_path, monkeypatch
):
    source = tmp_path / "curated.jsonl"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_source(source)
    first_manifest = _materialize(source, first)
    captured = []

    def from_generator(generate, *, gen_kwargs):
        captured.append(gen_kwargs)
        return list(generate(**gen_kwargs))

    monkeypatch.setattr("agoge_forger.split_loaders.Dataset.from_generator", from_generator)
    load_frozen_dataset(first / "split_manifest.json", "train")

    source.write_bytes(
        source.read_bytes().replace(b"deterministic sample 0", b"materially changed 0")
    )
    second_manifest = _materialize(source, second)
    (first / "split_manifest.json").write_bytes((second / "split_manifest.json").read_bytes())
    for split in SPLIT_NAMES:
        (first / first_manifest.splits[split].path).write_bytes(
            (second / second_manifest.splits[split].path).read_bytes()
        )
    load_frozen_dataset(first / "split_manifest.json", "train")

    assert captured[0]["request"].manifest_sha256 != captured[1]["request"].manifest_sha256
    assert captured[0]["request"].split_sha256 != captured[1]["request"].split_sha256


def test_frozen_training_dataset_projects_away_heterogeneous_metadata(tmp_path, monkeypatch):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    rows = [
        {
            "canonical_id": f"sample-{index:03d}",
            "lineage_id": f"lineage-{index:03d}",
            "text": f"Training row {index}",
            "metadata": index if index % 2 else {"source": index},
        }
        for index in range(30)
    ]
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    _materialize(source, output)

    def from_generator(generate, *, gen_kwargs):
        return list(generate(**gen_kwargs))

    monkeypatch.setattr("agoge_forger.split_loaders.Dataset.from_generator", from_generator)

    dataset = load_frozen_dataset(output / "split_manifest.json", "train")

    assert dataset
    assert all(set(row) == {"text"} for row in dataset)


def test_source_and_materialized_mutations_are_detected(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    _write_source(source)
    manifest = _materialize(source, output)
    manifest_path = output / "split_manifest.json"

    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        validate_split_manifest(manifest_path, source_path=source)

    split_path = output / manifest.splits["train"].path
    split_path.write_bytes(split_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="train digest mismatch"):
        validate_split_manifest(manifest_path)


def test_source_validation_recomputes_pinned_split_ownership(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    _write_source(source)
    manifest = _materialize(source, output)
    manifest_path = output / "split_manifest.json"
    raw_manifest = json.loads(manifest_path.read_text())

    train_path = output / manifest.splits["train"].path
    held_path = output / manifest.splits["held_out"].path
    train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    held_rows = [json.loads(line) for line in held_path.read_text().splitlines()]
    train_rows[0], held_rows[0] = held_rows[0], train_rows[0]
    train_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in train_rows)
    held_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in held_rows)
    train_path.write_bytes(train_payload)
    held_path.write_bytes(held_payload)

    train_members = raw_manifest["splits"]["train"]["members"]
    held_members = raw_manifest["splits"]["held_out"]["members"]
    train_members[0], held_members[0] = held_members[0], train_members[0]
    raw_manifest["splits"]["train"]["sha256"] = sha256_bytes(train_payload)
    raw_manifest["splits"]["held_out"]["sha256"] = sha256_bytes(held_payload)
    manifest_path.write_bytes(canonical_json_bytes(raw_manifest) + b"\n")

    with pytest.raises(ValueError, match="split ownership differs"):
        validate_split_manifest(manifest_path, source_path=source)


def test_cross_split_exact_content_leakage_fails_closed(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    _write_source(source)
    manifest = _materialize(source, output)

    train_row = next(iter(iter_frozen_records(output / "split_manifest.json", "train")))
    held_path = output / manifest.splits["held_out"].path
    held_rows = [json.loads(line) for line in held_path.read_text().splitlines()]
    held_rows[0]["text"] = train_row["text"]
    held_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in held_rows)
    held_path.write_bytes(held_payload)

    raw_manifest = json.loads((output / "split_manifest.json").read_text())
    train_content_sha = raw_manifest["splits"]["train"]["members"][0]["content_sha256"]
    raw_manifest["splits"]["held_out"]["sha256"] = sha256_bytes(held_payload)
    raw_manifest["splits"]["held_out"]["members"][0]["content_sha256"] = train_content_sha
    raw_manifest["splits"]["held_out"]["members"][0]["materialized_line_sha256"] = sha256_bytes(
        canonical_json_bytes(held_rows[0]) + b"\n"
    )
    (output / "split_manifest.json").write_bytes(canonical_json_bytes(raw_manifest) + b"\n")

    with pytest.raises(ValueError, match="deterministic leakage audit failed"):
        validate_split_manifest(output / "split_manifest.json")


def test_ancillary_metadata_cannot_split_identical_training_content(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    _write_source(source)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0].update(
        lineage_id="metadata-lineage-a",
        group_id="metadata-group-a",
        quality_score=0.1,
    )
    rows[1].update(
        lineage_id="metadata-lineage-b",
        group_id="metadata-group-b",
        quality_score=0.9,
        text=rows[0]["text"],
    )
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))

    manifest = _materialize(source, output)
    members = {
        member.canonical_id: (split, member)
        for split, artifact in manifest.splits.items()
        for member in artifact.members
    }
    first_split, first = members[rows[0]["canonical_id"]]
    second_split, second = members[rows[1]["canonical_id"]]

    assert first.content_sha256 == second.content_sha256
    assert first.raw_line_sha256 != second.raw_line_sha256
    assert first_split == second_split


def test_mixed_training_representations_fail_closed(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    _write_source(source)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0].update(
        lineage_id="text-message-a",
        group_id="text-message-a",
        text="User: Q\nAssistant: A",
    )
    rows[1].pop("text")
    rows[1].update(
        lineage_id="text-message-b",
        group_id="text-message-b",
        messages=[
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ],
    )
    rows[2].update(
        lineage_id="text-instruction-a",
        group_id="text-instruction-a",
        text="Instruction: Q\nOutput: A",
    )
    rows[3].pop("text")
    rows[3].update(
        lineage_id="text-instruction-b",
        group_id="text-instruction-b",
        instruction="Q",
        output="A",
    )
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))

    with pytest.raises(ValueError, match="new split snapshots require pre-rendered 'text'"):
        _materialize(source, output)
    assert not output.exists()


def test_v1_manifest_omitting_legacy_content_policy_keeps_legacy_semantics(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    _write_source(source)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for index, row in enumerate(rows):
        row["quality_score"] = index / len(rows)
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    _materialize(source, output)

    manifest_path = output / "split_manifest.json"
    raw_manifest = json.loads(manifest_path.read_text())
    legacy_digests = {
        row["canonical_id"]: sha256_bytes(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"canonical_id", "lineage_id", "group_id"}
                }
            )
        )
        for row in rows
    }
    for artifact in raw_manifest["splits"].values():
        for member in artifact["members"]:
            member["content_sha256"] = legacy_digests[member["canonical_id"]]
    raw_manifest["canonical_identity"].pop("content_hash_policy")
    manifest_path.write_bytes(canonical_json_bytes(raw_manifest) + b"\n")

    validated = validate_split_manifest(manifest_path, source_path=source)
    assert (
        validated.canonical_identity.content_hash_policy
        == "canonical-json-excluding-identity-fields"
    )


def test_lineage_and_declared_group_membership_remain_atomic(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    _write_source(source)
    manifest = _materialize(source, output)

    lineage_owners: dict[str, str] = {}
    group_owners: dict[str, str] = {}
    for split, artifact in manifest.splits.items():
        for member in artifact.members:
            assert lineage_owners.setdefault(member.lineage_id, split) == split
            assert member.group_id is not None
            assert group_owners.setdefault(member.group_id, split) == split


class CharacterTokenizer:
    name_or_path = "fake/character-tokenizer"
    _commit_hash = "b" * 40

    def __call__(self, text: str):
        return {"input_ids": list(text.encode("utf-8"))}


class WordPieceTokenizer:
    name_or_path = "fake/word-piece-tokenizer"
    _commit_hash = "d" * 40

    def __call__(self, text: str):
        return [piece for word in text.split() for piece in (word[:2], word[2:]) if piece]


def plain_text_serializer(row):
    return str(row["text"])


plain_text_serializer.serializer_id = "plain-text"
plain_text_serializer.serializer_version = "1"


def tagged_uppercase_serializer(row):
    return f"<instruction>\n{row['text'].upper()}\n</instruction>"


tagged_uppercase_serializer.serializer_id = "tagged-uppercase"
tagged_uppercase_serializer.serializer_version = "9"


def test_materially_distinct_tokenizer_and_serializer_stats_do_not_change_splits(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "frozen"
    _write_source(source)
    manifest = _materialize(source, output)
    manifest_path = output / "split_manifest.json"
    manifest_before = manifest_path.read_bytes()
    split_digests = {name: manifest.splits[name].sha256 for name in SPLIT_NAMES}
    character_tokenizer = TokenizerBinding(implementation=CharacterTokenizer())
    plain_serializer = SerializerBinding(implementation=plain_text_serializer)

    character_stats = write_token_statistics(
        manifest_path,
        output / "character-token-stats.json",
        TokenStatisticsDerivation(
            tokenizer=character_tokenizer,
            serializer=plain_serializer,
            spec=TokenStatisticsSpec(
                model_id="fake/model-family-a",
                model_revision="a" * 40,
                tokenizer_id=character_tokenizer.tokenizer_id,
                tokenizer_revision=character_tokenizer.tokenizer_revision,
                tokenizer_sha256=character_tokenizer.tokenizer_sha256,
                serializer_id=plain_serializer.serializer_id,
                serializer_version=plain_serializer.serializer_version,
                serializer_sha256=plain_serializer.serializer_sha256,
                context_limit=64,
            ),
        ),
    )
    word_tokenizer = TokenizerBinding(implementation=WordPieceTokenizer())
    tagged_serializer = SerializerBinding(implementation=tagged_uppercase_serializer)
    word_stats = write_token_statistics(
        manifest_path,
        output / "word-token-stats.json",
        TokenStatisticsDerivation(
            tokenizer=word_tokenizer,
            serializer=tagged_serializer,
            spec=TokenStatisticsSpec(
                model_id="fake/model-family-b",
                model_revision="c" * 40,
                tokenizer_id=word_tokenizer.tokenizer_id,
                tokenizer_revision=word_tokenizer.tokenizer_revision,
                tokenizer_sha256=word_tokenizer.tokenizer_sha256,
                serializer_id=tagged_serializer.serializer_id,
                serializer_version=tagged_serializer.serializer_version,
                serializer_sha256=tagged_serializer.serializer_sha256,
                context_limit=16,
            ),
        ),
    )

    assert character_stats.source_split_sha256 == split_digests
    assert word_stats.source_split_sha256 == split_digests
    assert manifest_path.read_bytes() == manifest_before
    assert character_stats.splits["train"].total_tokens != word_stats.splits["train"].total_tokens
    assert character_stats.tokenizer_revision != word_stats.tokenizer_revision
    assert character_stats.tokenizer_sha256 == character_tokenizer.tokenizer_sha256
    assert word_stats.tokenizer_sha256 == word_tokenizer.tokenizer_sha256
    assert character_stats.tokenizer_sha256 != word_stats.tokenizer_sha256
    assert character_stats.model_id != word_stats.model_id
    assert character_stats.serializer_sha256 != word_stats.serializer_sha256


@pytest.mark.parametrize(
    ("field", "floating_revision"),
    [("model_revision", "main"), ("tokenizer_revision", "latest")],
)
def test_token_statistics_requires_immutable_revisions(field, floating_revision):
    values = {
        "model_id": "fake/model",
        "model_revision": "a" * 40,
        "tokenizer_id": "fake/tokenizer",
        "tokenizer_revision": "b" * 40,
        "tokenizer_sha256": "d" * 64,
        "serializer_id": "plain-text",
        "serializer_version": "1",
        "serializer_sha256": "c" * 64,
    }
    values[field] = floating_revision

    with pytest.raises(ValidationError, match=field):
        TokenStatisticsSpec(**values)


def test_duplicate_canonical_identity_is_rejected_before_materialization(tmp_path):
    source = tmp_path / "bad.jsonl"
    rows = [
        {"canonical_id": "same", "lineage_id": "one", "text": "first"},
        {"canonical_id": "same", "lineage_id": "two", "text": "second"},
    ]
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))

    with pytest.raises(ValueError, match="duplicate canonical ID"):
        _materialize(source, tmp_path / "never-created")
