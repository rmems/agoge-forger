"""JSON/table output and base-revision tests for run-status."""

import json

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.run_status import build_run_status, format_run_status_table
from tests.test_run_status import (
    _make_run_dir,
    _write_checkpoint,
    _write_final_adapter,
    _write_merged_model,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# 10. CLI output contract
# --------------------------------------------------------------------------


def test_cli_json_format_matches_build_run_status(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 30)
    _write_final_adapter(run_dir)
    _write_merged_model(tmp_path / "merged" / run_dir.name)

    result = runner.invoke(app, ["run-status", str(run_dir), "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == build_run_status(str(run_dir))


def test_cli_defaults_to_json_output(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 30)

    result = runner.invoke(app, ["run-status", str(run_dir)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == build_run_status(str(run_dir))


def test_cli_table_format_is_aligned_text_not_json(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 30)
    _write_final_adapter(run_dir)

    result = runner.invoke(app, ["run-status", str(run_dir), "--format", "table"])

    assert result.exit_code == 0
    assert "resume_ready:" in result.stdout
    assert "export_ready:" in result.stdout
    assert "schema_version:" in result.stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_table_escapes_ansi_controls_in_base_model(tmp_path):
    """Table cells must not emit raw ANSI / Cc controls from adapter metadata."""
    run_dir = _make_run_dir(tmp_path)
    ansi_model = "\x1b[31mevil-model\x1b[0m"
    _write_final_adapter(run_dir, base_model=ansi_model)

    table = format_run_status_table(build_run_status(str(run_dir)))

    assert "\x1b" not in table
    assert "\\u001b[31mevil-model\\u001b[0m" in table
    assert "evil-model" in table


def test_table_escapes_ansi_controls_in_run_name_and_run_dir(tmp_path):
    """run_name / run_dir are not _or_dash fields and must still be escaped."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    report = build_run_status(str(run_dir))
    report["run_name"] = "\x1b[31mevil\x1b[0m"
    report["run_dir"] = str(tmp_path / "\x1b[31mevil\x1b[0m")

    table = format_run_status_table(report)

    assert "\x1b" not in table
    assert "\\u001b[31mevil\\u001b[0m" in table


def test_table_escapes_unicode_format_controls(tmp_path):
    """Bidi and other Cf controls must not visually reorder table fields."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir, base_model="safe\u202ereversed")

    table = format_run_status_table(build_run_status(str(run_dir)))

    assert "\u202e" not in table
    assert "safe\\u202ereversed" in table


def test_table_escapes_lone_unicode_surrogates(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    report = build_run_status(str(run_dir))
    report["base_model"] = "bad\ud800value"

    table = format_run_status_table(report)

    assert "bad\\ud800value" in table
    assert b"bad\\ud800value" in table.encode("utf-8")


def test_table_renderer_covers_every_report_row(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 30)

    table = format_run_status_table(build_run_status(str(run_dir)))
    labels = [line.split(":", 1)[0] for line in table.splitlines()]

    assert "resume_ready" in labels
    assert "export_ready" in labels
    assert "checkpoint_steps" in labels
    # Aligned block: every value starts at the same column.
    starts = {len(line) - len(line.split(":", 1)[1].lstrip()) for line in table.splitlines()}
    assert len(starts) == 1


def test_cli_invalid_format_exits_nonzero(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)

    result = runner.invoke(app, ["run-status", str(run_dir), "--format", "yaml"])

    assert result.exit_code != 0


# --------------------------------------------------------------------------
# 11. base_revision
# --------------------------------------------------------------------------


def test_base_revision_is_surfaced_when_pinned(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir, revision="deadbeefcafe")

    report = build_run_status(str(run_dir))

    assert report["base_model"] == str(run_dir / ".test-base-model")
    assert report["base_revision"] == "deadbeefcafe"


def test_base_revision_is_null_when_absent(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 40)

    report = build_run_status(str(run_dir))

    assert report["base_model"] == str(run_dir / "checkpoint-40" / ".test-base-model")
    assert report["base_revision"] is None
