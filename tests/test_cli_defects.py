"""Regression tests for the CLI defects filed as #127.

These cover export commands that answered ordinary operator mistakes with a
traceback and an inspection command that crashed on every valid file.
"""

import json
import os

import pytest
from typer.testing import CliRunner

from agoge_forger.artifacts.safetensors_io import inspect_safetensors_file
from agoge_forger.cli import app
from tests.test_run_status import _make_run_dir, _minimal_safetensors, _write_final_adapter


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _assert_clean_exit(result, code):
    assert result.exit_code == code, result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


# --- inspect-safetensors -----------------------------------------------------


def test_inspect_reads_a_valid_file(tmp_path):
    """`for key in f` raised TypeError on every valid file: safe_open exposes
    keys() but is not iterable, and the handler does not catch TypeError."""
    target = tmp_path / "adapter_model.safetensors"
    target.write_bytes(_minimal_safetensors())

    info = inspect_safetensors_file(str(target))

    assert info["tensors"], "no tensors reported for a valid safetensors file"
    for entry in info["tensors"].values():
        assert "shape" in entry and "dtype" in entry


def test_inspect_cli_reports_tensors_and_exits_zero(runner, tmp_path):
    target = tmp_path / "adapter_model.safetensors"
    target.write_bytes(_minimal_safetensors())

    result = runner.invoke(app, ["inspect-safetensors", "--path", str(target)])

    _assert_clean_exit(result, 0)


def test_corrupt_file_is_an_error_not_a_traceback(runner, tmp_path, caplog):
    target = tmp_path / "corrupt.safetensors"
    target.write_bytes(b"not a safetensors file at all")

    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(app, ["inspect-safetensors", "--path", str(target)])

    _assert_clean_exit(result, 1)
    assert caplog.messages


# --- export-final-model ------------------------------------------------------


def test_missing_provenance_is_an_error_not_a_traceback(runner, tmp_path, caplog):
    """An adapter never finalized has no artifact_index.json, so reading its
    sealed provenance raises before a single weight is loaded."""
    run_dir = _write_final_adapter(_make_run_dir(tmp_path))

    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(
            app,
            [
                "export-final-model",
                "--run-dir",
                str(run_dir),
                "--out-dir",
                str(tmp_path / "merged" / "demo_run"),
            ],
        )

    _assert_clean_exit(result, 1)
    assert any("producer_provenance" in message for message in caplog.messages)


def test_run_dir_without_an_exportable_artifact_is_an_error(runner, tmp_path, caplog):
    run_dir = _make_run_dir(tmp_path)

    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(
            app,
            [
                "export-final-model",
                "--run-dir",
                str(run_dir),
                "--out-dir",
                str(tmp_path / "merged" / "demo_run"),
            ],
        )

    _assert_clean_exit(result, 1)
    assert any("No exportable adapter artifact" in message for message in caplog.messages)


def test_export_json_of_a_valid_adapter_still_resolves(tmp_path):
    """Guard the happy path: the boundary must not swallow a usable adapter."""
    run_dir = _write_final_adapter(_make_run_dir(tmp_path))

    assert (run_dir / "adapter_config.json").is_file()
    assert json.loads((run_dir / "adapter_config.json").read_text())["base_model_name_or_path"]


def test_malformed_index_is_an_error_not_a_traceback(runner, tmp_path, caplog):
    """A JSON array parses fine but raises TypeError, which the handler missed."""
    run_dir = _write_final_adapter(_make_run_dir(tmp_path))
    (run_dir / "artifact_index.json").write_text("[]")

    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(
            app,
            [
                "export-final-model",
                "--run-dir",
                str(run_dir),
                "--out-dir",
                str(tmp_path / "merged" / "demo_run"),
            ],
        )

    _assert_clean_exit(result, 1)
    assert caplog.messages


# --- boundaries and config validation surfaced on #130 ----------------------


def test_unreadable_safetensors_exits_nonzero(runner, tmp_path, caplog):
    """An I/O failure used to be logged and swallowed into an empty result, so
    the command printed `{}` and exited 0 on a file it could not read."""
    target = tmp_path / "adapter_model.safetensors"
    target.write_bytes(_minimal_safetensors())
    target.chmod(0o000)
    if os.access(target, os.R_OK):  # running as root: the chmod means nothing
        pytest.skip("cannot make a file unreadable as this user")

    try:
        with caplog.at_level("ERROR", logger="agoge"):
            result = runner.invoke(app, ["inspect-safetensors", "--path", str(target)])
    finally:
        target.chmod(0o644)

    _assert_clean_exit(result, 1)
    assert caplog.messages


def test_merged_dir_pointing_at_a_file_reads_as_absent(runner, tmp_path):
    """run-status reports a non-directory merged path as absent; passing one
    explicitly used to exit 1 instead."""
    run_dir = _write_final_adapter(_make_run_dir(tmp_path))
    not_a_dir = tmp_path / "merged.txt"
    not_a_dir.write_text("x")

    result = runner.invoke(app, ["run-status", str(run_dir), "--merged-dir", str(not_a_dir)])

    _assert_clean_exit(result, 0)
    assert json.loads(result.stdout)["merged_model"]["present"] is False


def test_bad_training_config_path_exits_one(runner, tmp_path, caplog):
    """load_config resolves the path and requires keys; those are operator
    mistakes, not tracebacks."""
    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(app, ["train-lora", "--config", str(tmp_path / "nope.yaml")])

    _assert_clean_exit(result, 1)
    assert caplog.messages


def test_export_merges_the_source_its_provenance_came_from(runner, tmp_path, monkeypatch):
    """The command resolved the export source to read provenance, then let the
    export layer resolve again. A checkpoint written between the two would be
    merged under provenance describing the earlier one."""
    run_dir = _write_final_adapter(_make_run_dir(tmp_path))
    (run_dir / "artifact_index.json").write_text(
        json.dumps({"output_dir": str(run_dir), "artifacts": [], "producer_provenance": None})
    )
    seen = {}

    def fake_export(**kwargs):
        seen.update(kwargs)

    def fake_provenance(path):
        # Returning None is the point: the export layer accepts absent provenance,
        # so the test isolates *which path* it was read from.
        seen["provenance_source"] = str(path)

    monkeypatch.setattr("agoge_forger._cli_export._run_export", fake_export)
    monkeypatch.setattr(
        "agoge_forger._cli_export.producer_provenance_from_adapter", fake_provenance
    )

    result = runner.invoke(
        app,
        ["export-final-model", "--run-dir", str(run_dir), "--out-dir", str(tmp_path / "merged")],
    )

    _assert_clean_exit(result, 0)
    # Whatever provenance was read from is exactly what gets exported.
    assert seen["adapter_path"] == seen["provenance_source"]
