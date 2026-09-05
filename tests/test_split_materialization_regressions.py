import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from agoge_forger import split_materialize, split_validation
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


@pytest.mark.parametrize("field", ["canonical_id", "lineage_id", "group_id"])
def test_materialization_rejects_noncanonical_identity_whitespace(
    tmp_path: Path, field: str
) -> None:
    source = tmp_path / "download.jsonl"
    output = tmp_path / "snapshot"
    rows = _text_rows()
    rows[0][field] = f" {rows[0][field]} "
    _write_rows(source, rows)

    with pytest.raises(ValueError, match=rf"identity field '{field}'.*surrounding whitespace"):
        materialize_split(source, output, _spec())

    assert not output.exists()


def test_new_materialization_requires_declared_lineage_id(tmp_path: Path) -> None:
    source = tmp_path / "download.jsonl"
    output = tmp_path / "snapshot"
    rows = _text_rows()
    del rows[0]["lineage_id"]
    _write_rows(source, rows)

    with pytest.raises(ValueError, match="identity field 'lineage_id' must be a string"):
        materialize_split(source, output, _spec())

    assert not output.exists()


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


def test_new_snapshot_requires_prerendered_text(tmp_path: Path) -> None:
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
        match=r"new split snapshots require pre-rendered 'text'; 'messages'",
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


def test_materialization_records_explicit_repository_source_path(tmp_path: Path) -> None:
    source = tmp_path / "cache" / "downloaded.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())

    manifest = materialize_split(source, output, _spec())
    manifest = validate_split_manifest(output / "split_manifest.json", source_path=source)
    coordinates = {
        member.source_coordinate
        for artifact in manifest.splits.values()
        for member in artifact.members
    }
    assert manifest.source.path == SOURCE_PATH
    assert all(coordinate.startswith(f"{SOURCE_PATH}:") for coordinate in coordinates)
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


@pytest.mark.parametrize("field", ["canonical_id", "lineage_id", "group_id", "text"])
def test_streaming_source_reader_rejects_duplicate_json_keys(tmp_path: Path, field: str) -> None:
    source = tmp_path / "duplicate-key.jsonl"
    source.write_text(
        f'{{"canonical_id":"sample","lineage_id":"lineage","group_id":"group",'
        f'"text":"payload",'
        f'"{field}":"ambiguous"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"{SOURCE_PATH}:1: duplicate JSON object key '{field}'",
    ):
        list(
            iter_source_records(
                source,
                CanonicalIdentityPolicy(content_hash_policy="normalized-training-payload-v1"),
                source_coordinate_path=SOURCE_PATH,
            )
        )


def test_streaming_source_reader_rejects_nested_duplicate_json_keys(tmp_path: Path) -> None:
    source = tmp_path / "nested-duplicate-key.jsonl"
    source.write_text(
        '{"canonical_id":"sample","lineage_id":"lineage","text":"payload",'
        '"metadata":{"license":"first","license":"second"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key 'license'"):
        list(
            iter_source_records(
                source,
                CanonicalIdentityPolicy(content_hash_policy="normalized-training-payload-v1"),
                source_coordinate_path=SOURCE_PATH,
            )
        )


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


def test_source_validation_hashes_and_parses_one_snapshot(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())
    manifest = materialize_split(source, output, _spec())
    replacement = tmp_path / "replacement.jsonl"
    replacement_rows = [
        {**row, "text": f"Replacement row {index}"} for index, row in enumerate(_text_rows())
    ]
    _write_rows(replacement, replacement_rows)
    original_read = split_validation.read_source_records

    def read_then_replace(snapshot, *args, **kwargs):
        assert snapshot != source
        os.replace(replacement, source)
        return original_read(snapshot, *args, **kwargs)

    monkeypatch.setattr(split_validation, "read_source_records", read_then_replace)

    assert validate_split_manifest(output / "split_manifest.json", source_path=source) == manifest


def test_materialization_rejects_source_changed_while_snapshotting(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())
    original_open = Path.open
    mutated = False

    class MutatingReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def fileno(self):
            return self.handle.fileno()

        def read(self, size=-1):
            nonlocal mutated
            chunk = self.handle.read(size)
            if not mutated:
                mutated = True
                with original_open(source, "r+b") as writer:
                    writer.seek(0)
                    writer.write(b" ")
                    writer.flush()
                    os.fsync(writer.fileno())
            return chunk

    def open_and_mutate(path, mode="r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        if path == source and mode == "rb":
            return MutatingReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", open_and_mutate)

    with pytest.raises(ValueError, match="source changed while creating immutable snapshot"):
        materialize_split(source, output, _spec())

    assert mutated
    assert not output.exists()


def test_source_snapshot_rehash_detects_same_metadata_content_race(
    tmp_path: Path, monkeypatch
) -> None:
    from agoge_forger import _source_snapshot

    source = tmp_path / "source.jsonl"
    snapshot = tmp_path / "snapshot.jsonl"
    original = b"original-source-payload\n"
    replacement = b"mutated--source-payload\n"
    assert len(original) == len(replacement)
    source.write_bytes(original)

    original_hash = _source_snapshot._sha256_path

    def mutate_then_hash(path: Path) -> str:
        if path == source:
            metadata = source.stat()
            source.write_bytes(replacement)
            os.utime(source, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        return original_hash(path)

    monkeypatch.setattr(_source_snapshot, "_sha256_path", mutate_then_hash)

    with pytest.raises(ValueError, match="source changed while creating immutable snapshot"):
        _source_snapshot.copy_source_snapshot(source, snapshot)


def test_materialization_stages_on_output_filesystem(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "missing-parent" / "snapshot"
    _write_rows(source, _text_rows())
    original_copy = split_materialize.copy_source_snapshot
    observed_staging: list[Path] = []

    def capture_staging(source_path, snapshot_path):
        staging = snapshot_path.parent
        observed_staging.append(staging)
        assert staging.parent == tmp_path
        assert staging.stat().st_dev == tmp_path.stat().st_dev
        return original_copy(source_path, snapshot_path)

    monkeypatch.setattr(split_materialize, "copy_source_snapshot", capture_staging)

    materialize_split(source, output, _spec())

    assert len(observed_staging) == 1
    assert not observed_staging[0].exists()


def test_materialization_does_not_publish_partial_snapshot_on_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())
    original_write_metadata = split_materialize._write_snapshot_metadata

    def fail_after_manifest(destination, manifest):
        split_materialize.exclusive_write(
            destination / "split_manifest.json",
            split_materialize.manifest_bytes(manifest),
        )
        raise OSError("synthetic metadata write failure")

    monkeypatch.setattr(split_materialize, "_write_snapshot_metadata", fail_after_manifest)
    with pytest.raises(OSError, match="synthetic metadata write failure"):
        materialize_split(source, output, _spec())

    assert not output.exists()
    monkeypatch.setattr(split_materialize, "_write_snapshot_metadata", original_write_metadata)
    manifest = materialize_split(source, output, _spec())
    assert validate_split_manifest(output / "split_manifest.json") == manifest


def test_materialization_does_not_replace_dangling_destination_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="output path already exists"):
        materialize_split(source, output, _spec())

    assert output.is_symlink()


def test_materialization_does_not_replace_concurrently_created_destination(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())
    original_write_metadata = split_materialize._write_snapshot_metadata

    def create_destination_after_writes(destination, manifest):
        original_write_metadata(destination, manifest)
        output.mkdir()

    monkeypatch.setattr(
        split_materialize,
        "_write_snapshot_metadata",
        create_destination_after_writes,
    )
    with pytest.raises(FileExistsError, match="output path already exists"):
        materialize_split(source, output, _spec())

    assert output.is_dir()
    assert not any(output.iterdir())


def test_materialization_rejects_unsupported_atomic_publication_before_staging(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "snapshot"
    _write_rows(source, _text_rows())
    copy_source = pytest.fail

    def unsupported(staging_parent):
        assert staging_parent == tmp_path
        raise OSError("atomic publication unsupported")

    monkeypatch.setattr(split_materialize, "require_rename_noreplace_support", unsupported)
    monkeypatch.setattr(split_materialize, "copy_source_snapshot", copy_source)

    with pytest.raises(OSError, match="atomic publication unsupported"):
        materialize_split(source, output, _spec())

    assert not output.exists()


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
