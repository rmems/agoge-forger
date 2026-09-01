import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from agoge_forger import split_materialize
from agoge_forger.datasets import normalize_row
from agoge_forger.split_contract import (
    CanonicalIdentityPolicy,
    SourceFile,
    SplitMaterializationSpec,
    SplitPolicy,
    canonical_json_bytes,
    validate_split_manifest,
)
from agoge_forger.split_materialize import iter_source_records, materialize_split

REVISION = "0123456789abcdef0123456789abcdef01234567"
SOURCE_PATH = "datasets/curated/sft.jsonl"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _text_rows(count: int = 90) -> list[dict[str, object]]:
    return [
        {
            "canonical_id": f"sample-{index:03d}",
            "lineage_id": f"lineage-{index // 2:03d}",
            "group_id": f"family-{index // 3:03d}",
            "text": f"Deterministic source row {index} with evidence {index * 17}.",
        }
        for index in range(count)
    ]


def _spec(source_path: str = SOURCE_PATH) -> SplitMaterializationSpec:
    return SplitMaterializationSpec(
        source_repository="rmems/synthetic-factory",
        source_revision=REVISION,
        dataset_version="curated-sft-v1",
        source_path=source_path,
        split_policy=SplitPolicy(
            seed=20260830,
            salt="agoge-issue-99-v1",
            weights={"train": 6, "validation": 2, "held_out": 2},
        ),
    )


class _TemplateTokenizer:
    chat_template = "pinned-test-template"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        return "<chat>" + "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )


def test_tokenizer_equivalent_text_and_messages_cannot_share_a_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "download.jsonl"
    output = tmp_path / "snapshot"
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    message_row: dict[str, object] = {
        "canonical_id": "message-row",
        "lineage_id": "message-lineage",
        "messages": messages,
    }
    rendered = normalize_row(message_row, tokenizer=_TemplateTokenizer(), index=2)["text"]
    rows = [
        {
            "canonical_id": "text-row",
            "lineage_id": "text-lineage",
            "text": rendered,
        },
        message_row,
    ]
    _write_rows(source, rows)

    with pytest.raises(
        ValueError,
        match=r"source mixes model-dependent training representations: text .* messages",
    ):
        materialize_split(source, output, _spec())

    assert not output.exists()


@pytest.mark.parametrize(
    "source_path",
    [
        "/absolute/source.jsonl",
        "C:/datasets/source.jsonl",
        "../source.jsonl",
        "datasets/../source.jsonl",
        r"datasets\source.jsonl",
        "./datasets/source.jsonl",
        "datasets//source.jsonl",
        "datasets/source.jsonl/",
    ],
)
def test_repository_source_path_must_be_portable_and_confined(source_path: str) -> None:
    with pytest.raises(ValidationError, match="repository-relative path"):
        _spec(source_path)
    with pytest.raises(ValidationError, match="repository-relative path"):
        SourceFile(
            repository="rmems/synthetic-factory",
            revision=REVISION,
            dataset_version="curated-sft-v1",
            path=source_path,
            sha256="0" * 64,
            record_count=1,
        )


def test_cli_records_and_reports_explicit_repository_source_path(tmp_path: Path) -> None:
    source = tmp_path / "cache" / "downloaded.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())

    result = subprocess.run(
        [
            sys.executable,
            "scripts/freeze_split.py",
            "--source",
            str(source),
            "--source-path",
            SOURCE_PATH,
            "--output-dir",
            str(output),
            "--source-repository",
            "rmems/synthetic-factory",
            "--source-revision",
            REVISION,
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = validate_split_manifest(output / "split_manifest.json", source_path=source)
    coordinates = {
        member.source_coordinate
        for artifact in manifest.splits.values()
        for member in artifact.members
    }
    assert manifest.source.path == SOURCE_PATH
    assert all(coordinate.startswith(f"{SOURCE_PATH}:") for coordinate in coordinates)
    assert f"rmems/synthetic-factory@{REVISION}:{SOURCE_PATH}" in result.stdout
    assert f"Source file: `{SOURCE_PATH}`" in (output / "split_report.md").read_text()


def test_streaming_source_reader_rejects_duplicate_ids_when_exhausted(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.jsonl"
    rows = [
        {"canonical_id": "same", "lineage_id": "first", "text": "First"},
        {"canonical_id": "same", "lineage_id": "second", "text": "Second"},
    ]
    _write_rows(source, rows)
    records = iter_source_records(
        source,
        CanonicalIdentityPolicy(content_hash_policy="normalized-training-payload-v1"),
        source_coordinate_path=SOURCE_PATH,
    )

    assert next(records).member.source_coordinate == f"{SOURCE_PATH}:1"
    with pytest.raises(ValueError, match="duplicate canonical ID 'same'"):
        next(records)


def test_streaming_source_reader_rejects_an_empty_source(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("\n \n", encoding="utf-8")

    with pytest.raises(ValueError, match="source contains no JSONL records"):
        list(
            iter_source_records(
                source,
                CanonicalIdentityPolicy(content_hash_policy="normalized-training-payload-v1"),
                source_coordinate_path=SOURCE_PATH,
            )
        )


def test_materialization_hashes_the_same_source_snapshot_it_splits(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())
    original_payload = source.read_bytes()
    replacement_rows = [
        {**row, "text": f"Replacement row {index}"} for index, row in enumerate(_text_rows())
    ]
    replacement_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in replacement_rows)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(replacement_payload)
    original_iter = split_materialize.iter_source_records

    def iterate_then_replace(*args, **kwargs):
        yield from original_iter(*args, **kwargs)
        os.replace(replacement, source)

    monkeypatch.setattr(split_materialize, "iter_source_records", iterate_then_replace)

    manifest = materialize_split(source, output, _spec())

    assert manifest.source.sha256 == split_materialize.sha256_bytes(original_payload)
    assert validate_split_manifest(output / "split_manifest.json") == manifest
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        validate_split_manifest(output / "split_manifest.json", source_path=source)


def test_materialization_assigns_metadata_without_retaining_source_payloads(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())
    original_assign = split_materialize.assign_records
    observed_metadata_only = False

    def assert_metadata_only(records, spec):
        nonlocal observed_metadata_only
        assert all(record.row == {} and record.raw_line == b"" for record in records)
        observed_metadata_only = True
        return original_assign(records, spec)

    monkeypatch.setattr(split_materialize, "assign_records", assert_metadata_only)

    manifest = materialize_split(source, output, _spec())

    assert observed_metadata_only
    assert validate_split_manifest(output / "split_manifest.json") == manifest
