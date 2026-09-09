from pathlib import Path

from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.split_contract import canonical_json_bytes


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


def _invoke_freeze_split(source: Path, output: Path):
    return CliRunner().invoke(
        app,
        [
            "freeze-split",
            "--source",
            str(source),
            "--source-path",
            "data/curated.jsonl",
            "--output-dir",
            str(output),
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
        ],
    )


def _assert_cli_error(result, caplog) -> None:
    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert any("symlink" in message.lower() for message in caplog.messages)


def test_freeze_split_cli_does_not_precreate_destination(tmp_path):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "nested" / "snapshot"
    _write_source(source)

    result = _invoke_freeze_split(source, output)

    assert result.exit_code == 0, result.output
    assert (output / "split_manifest.json").is_file()
    assert (output / "split_report.md").is_file()


def test_freeze_split_cli_refuses_dangling_output_symlink(tmp_path, caplog):
    """A dangling --output-dir symlink used to resolve to its target, so the
    immutable snapshot was published somewhere other than the named path."""
    source = tmp_path / "curated.jsonl"
    elsewhere = tmp_path / "elsewhere" / "hijacked"
    output = tmp_path / "snapshot"
    _write_source(source)
    output.symlink_to(elsewhere, target_is_directory=True)

    with caplog.at_level("ERROR", logger="agoge"):
        result = _invoke_freeze_split(source, output)

    _assert_cli_error(result, caplog)
    assert output.is_symlink()
    assert output.readlink() == elsewhere
    assert not elsewhere.exists()
    assert not (tmp_path / "elsewhere").exists()


def test_freeze_split_cli_refuses_redirected_output_symlink(tmp_path, caplog):
    source = tmp_path / "curated.jsonl"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    output = tmp_path / "snapshot"
    _write_source(source)
    output.symlink_to(elsewhere, target_is_directory=True)

    with caplog.at_level("ERROR", logger="agoge"):
        result = _invoke_freeze_split(source, output)

    _assert_cli_error(result, caplog)
    assert output.is_symlink()
    assert not (elsewhere / "split_manifest.json").exists()
    assert list(elsewhere.iterdir()) == []


def test_freeze_split_cli_refuses_symlinked_output_parent(tmp_path, caplog):
    """A symlinked parent is the same publish-elsewhere hole as a leaf link."""
    source = tmp_path / "curated.jsonl"
    real_parent = tmp_path / "real-parent"
    linked_parent = tmp_path / "linked-parent"
    output = linked_parent / "snapshot"
    _write_source(source)
    real_parent.mkdir()
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with caplog.at_level("ERROR", logger="agoge"):
        result = _invoke_freeze_split(source, output)

    _assert_cli_error(result, caplog)
    assert linked_parent.is_symlink()
    assert not (real_parent / "snapshot").exists()
    assert list(real_parent.iterdir()) == []


def test_freeze_split_cli_does_not_clobber_existing_destination(tmp_path, caplog):
    source = tmp_path / "curated.jsonl"
    output = tmp_path / "snapshot"
    marker = output / "keep-me.txt"
    _write_source(source)
    output.mkdir()
    marker.write_text("untouched")

    with caplog.at_level("ERROR", logger="agoge"):
        result = _invoke_freeze_split(source, output)

    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert marker.read_text() == "untouched"
    assert not (output / "split_manifest.json").exists()
    assert caplog.messages
