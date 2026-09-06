"""CLI surface for `agoge cleanup-run`: exit codes, dry-run, force, formats."""

import json

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from tests.test_run_status import _make_run_dir, _write_checkpoint, _write_final_adapter


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _run_with_checkpoints(tmp_path, steps=(50, 100), final_adapter=True):
    run_dir = _make_run_dir(tmp_path)
    for step in steps:
        _write_checkpoint(run_dir, step)
    if final_adapter:
        _write_final_adapter(run_dir)
    return run_dir


def _checkpoint_names(run_dir):
    return sorted(entry.name for entry in run_dir.iterdir() if entry.name.startswith("checkpoint-"))


def _assert_clean_exit(result, code):
    assert result.exit_code == code, result.stdout
    # A crash would surface here as something other than the exception click
    # raises for a controlled exit.
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_dry_run_reports_candidates_without_deleting(runner, tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    result = runner.invoke(app, ["cleanup-run", str(run_dir), "--dry-run", "--format", "json"])

    _assert_clean_exit(result, 0)
    report = json.loads(result.stdout)
    assert report["dry_run"] is True
    assert len(report["removed"]) == 2
    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-50"]


def test_execute_removes_checkpoints(runner, tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    result = runner.invoke(app, ["cleanup-run", str(run_dir), "--format", "json"])

    _assert_clean_exit(result, 0)
    assert json.loads(result.stdout)["dry_run"] is False
    assert _checkpoint_names(run_dir) == []


def test_keep_latest_is_honored_through_the_cli(runner, tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    result = runner.invoke(app, ["cleanup-run", str(run_dir), "--keep-latest", "1"])

    _assert_clean_exit(result, 0)
    assert _checkpoint_names(run_dir) == ["checkpoint-100"]


def test_table_format_is_rendered(runner, tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    result = runner.invoke(app, ["cleanup-run", str(run_dir), "--dry-run"])

    _assert_clean_exit(result, 0)
    assert "gib_reclaimed:" in result.stdout
    assert "checkpoints_removed:" in result.stdout
    # The human default also narrates each candidate.
    assert "[dry-run]" in result.stdout


def test_json_output_is_a_clean_document(runner, tmp_path):
    """--format json must stay pipeable: the logger renders to stdout too."""
    run_dir = _run_with_checkpoints(tmp_path)

    result = runner.invoke(app, ["cleanup-run", str(run_dir), "--format", "json"])

    _assert_clean_exit(result, 0)
    assert result.stdout.lstrip().startswith("{")
    assert "[dry-run]" not in result.stdout
    assert json.loads(result.stdout)["bytes_reclaimed"] > 0


def test_refuses_without_a_final_artifact(runner, tmp_path, caplog):
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)

    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(app, ["cleanup-run", str(run_dir)])

    _assert_clean_exit(result, 1)
    assert any("only recoverable artifact" in message for message in caplog.messages)
    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-50"]


def test_force_proceeds_and_warns(runner, tmp_path, caplog):
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)

    with caplog.at_level("WARNING", logger="agoge"):
        result = runner.invoke(app, ["cleanup-run", str(run_dir), "--force"])

    _assert_clean_exit(result, 0)
    assert _checkpoint_names(run_dir) == []
    assert any("no longer be resumed or exported" in message for message in caplog.messages)


def test_missing_path_exits_one_without_a_traceback(runner, tmp_path, caplog):
    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(app, ["cleanup-run", str(tmp_path / "nope")])

    _assert_clean_exit(result, 1)
    assert caplog.messages


def test_parent_traversal_exits_one(runner, tmp_path, caplog):
    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(app, ["cleanup-run", "../escape"])

    _assert_clean_exit(result, 1)
    assert any("must not contain" in message for message in caplog.messages)


def test_file_argument_is_rejected(runner, tmp_path, caplog):
    target = tmp_path / "not_a_dir.txt"
    target.write_text("x")

    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(app, ["cleanup-run", str(target)])

    _assert_clean_exit(result, 1)
    assert caplog.messages


def test_invalid_format_is_rejected_before_running(runner, tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    result = runner.invoke(app, ["cleanup-run", str(run_dir), "--format", "yaml"])

    assert result.exit_code != 0
    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-50"]


def test_run_status_still_reports_export_ready_after_cli_cleanup(runner, tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    _assert_clean_exit(runner.invoke(app, ["cleanup-run", str(run_dir), "--format", "json"]), 0)
    status = runner.invoke(app, ["run-status", str(run_dir)])

    _assert_clean_exit(status, 0)
    report = json.loads(status.stdout)
    assert report["export"]["ready"] is True
    assert report["checkpoints"]["valid_count"] == 0
