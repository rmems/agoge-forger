"""Tests for `agoge run-status` and the `run_status` report builder.

The JSON document is a published contract (operators pipe it into `jq`, and the
polyglot side tools read it), so the schema assertions here are deliberately
exact: an added, renamed or dropped key must break a test rather than silently
change what downstream consumers see.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.run_status import (
    SCHEMA_VERSION,
    build_run_status,
    find_merged_model_dir,
    format_run_status_table,
    is_merged_model_dir,
)

TOP_LEVEL_KEYS = {
    "schema_version",
    "run_dir",
    "run_name",
    "allow_unsafe_serialization",
    "checkpoints",
    "final_adapter",
    "merged_model",
    "base_model",
    "base_revision",
    "resume",
    "export",
}
CHECKPOINTS_KEYS = {"valid_count", "steps", "latest_step", "latest_path"}
FINAL_ADAPTER_KEYS = {"present", "path"}
MERGED_MODEL_KEYS = {"present", "path"}
RESUME_KEYS = {"ready", "checkpoint_path"}
EXPORT_KEYS = {"ready", "source_path", "source_kind"}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_adapter_config(directory, base_model="Qwen/Qwen3.5-0.5B", revision=None):
    payload = {}
    if base_model is not None:
        payload["base_model_name_or_path"] = base_model
    if revision is not None:
        payload["revision"] = revision
    (directory / "adapter_config.json").write_text(json.dumps(payload))


def _write_checkpoint(root, step, base_model="Qwen/Qwen3.5-0.5B", revision=None):
    checkpoint_dir = root / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    (checkpoint_dir / "adapter_model.safetensors").write_text("weights")
    _write_adapter_config(checkpoint_dir, base_model=base_model, revision=revision)
    return checkpoint_dir


def _write_final_adapter(root, base_model="Qwen/Qwen3.5-0.5B", revision=None):
    (root / "adapter_model.safetensors").write_text("final-weights")
    _write_adapter_config(root, base_model=base_model, revision=revision)
    return root


def _write_legacy_bin_adapter(root, base_model="Qwen/Qwen3.5-0.5B"):
    (root / "adapter_model.bin").write_text("legacy-weights")
    _write_adapter_config(root, base_model=base_model)
    return root


def _write_merged_model(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type": "llama"}')
    (path / "model.safetensors").write_text("merged-weights")
    return path


def _make_run_dir(tmp_path, name="demo_run"):
    """Build the conventional `<root>/adapters/<run_name>` run directory."""
    run_dir = tmp_path / "adapters" / name
    run_dir.mkdir(parents=True)
    return run_dir


# --------------------------------------------------------------------------
# 1. Schema stability
# --------------------------------------------------------------------------


def test_report_key_sets_are_exact(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 10)
    _write_final_adapter(run_dir)
    _write_merged_model(tmp_path / "merged" / run_dir.name)

    report = build_run_status(str(run_dir))

    assert set(report) == TOP_LEVEL_KEYS
    assert set(report["checkpoints"]) == CHECKPOINTS_KEYS
    assert set(report["final_adapter"]) == FINAL_ADAPTER_KEYS
    assert set(report["merged_model"]) == MERGED_MODEL_KEYS
    assert set(report["resume"]) == RESUME_KEYS
    assert set(report["export"]) == EXPORT_KEYS


def test_schema_version_is_one(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)

    assert SCHEMA_VERSION == 1
    assert build_run_status(str(run_dir))["schema_version"] == SCHEMA_VERSION == 1


def test_report_survives_json_round_trip(tmp_path):
    """No `Path` objects may leak into the report: it must be `json.dumps`-able."""
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 20)
    _write_final_adapter(run_dir)
    _write_merged_model(tmp_path / "merged" / run_dir.name)

    report = build_run_status(str(run_dir))

    assert json.loads(json.dumps(report)) == report


def test_report_identifies_run_dir_and_name(tmp_path):
    run_dir = _make_run_dir(tmp_path, name="my_run")

    report = build_run_status(str(run_dir))

    assert report["run_dir"] == str(run_dir.resolve())
    assert report["run_name"] == "my_run"


# --------------------------------------------------------------------------
# 2. Empty but inspectable run directory
# --------------------------------------------------------------------------


def test_empty_run_dir_reports_every_key_with_null_values(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    report = build_run_status(str(run_dir))

    assert set(report) == TOP_LEVEL_KEYS
    assert report["checkpoints"]["valid_count"] == 0
    assert report["checkpoints"]["steps"] == []
    assert report["checkpoints"]["latest_step"] is None
    assert report["checkpoints"]["latest_path"] is None
    assert report["final_adapter"] == {"present": False, "path": None}
    assert report["merged_model"] == {"present": False, "path": None}
    assert report["base_model"] is None
    assert report["base_revision"] is None
    assert report["resume"] == {"ready": False, "checkpoint_path": None}
    assert report["export"] == {"ready": False, "source_path": None, "source_kind": None}


def test_empty_run_dir_cli_exits_zero(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)

    result = runner.invoke(app, ["run-status", str(run_dir)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["checkpoints"]["valid_count"] == 0


# --------------------------------------------------------------------------
# 3. Checkpoints only
# --------------------------------------------------------------------------


def test_checkpoints_only_run_is_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    # Written newest-first to prove the report sorts by step, not by mtime.
    latest = _write_checkpoint(run_dir, 100, base_model="Qwen/Qwen3.5-0.5B")
    _write_checkpoint(run_dir, 50, base_model="Qwen/Qwen3.5-0.5B")

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["valid_count"] == 2
    assert report["checkpoints"]["steps"] == [50, 100]
    assert report["checkpoints"]["latest_step"] == 100
    assert report["checkpoints"]["latest_path"] == str(latest.resolve())
    assert report["resume"] == {"ready": True, "checkpoint_path": str(latest.resolve())}
    assert report["final_adapter"] == {"present": False, "path": None}
    assert report["export"]["ready"] is True
    assert report["export"]["source_kind"] == "checkpoint"
    assert report["export"]["source_path"] == str(latest.resolve())
    assert report["base_model"] == "Qwen/Qwen3.5-0.5B"


def test_latest_checkpoint_is_always_drawn_from_the_reported_steps(tmp_path):
    """`latest_step`/`latest_path` must describe a checkpoint the report lists.

    They are derived from the same single scan as `steps` and `valid_count`, so
    a checkpoint appearing or disappearing between two scans cannot yield a
    report whose latest checkpoint is missing from its own list.
    """
    run_dir = _make_run_dir(tmp_path)
    for step in (25, 50, 100):
        _write_checkpoint(run_dir, step)

    checkpoints = build_run_status(str(run_dir))["checkpoints"]

    assert checkpoints["latest_step"] in checkpoints["steps"]
    assert checkpoints["latest_step"] == checkpoints["steps"][-1]
    assert checkpoints["valid_count"] == len(checkpoints["steps"])
    assert checkpoints["latest_path"].endswith(f"checkpoint-{checkpoints['latest_step']}")


# --------------------------------------------------------------------------
# 4. Final adapter at the run root
# --------------------------------------------------------------------------


def test_final_adapter_wins_over_checkpoints_as_export_source(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    latest = _write_checkpoint(run_dir, 100)
    _write_final_adapter(run_dir)

    report = build_run_status(str(run_dir))

    assert report["final_adapter"] == {"present": True, "path": str(run_dir.resolve())}
    assert report["export"]["ready"] is True
    assert report["export"]["source_kind"] == "final_adapter"
    assert report["export"]["source_path"] == str(run_dir.resolve())
    # Resume still points at the checkpoint, not the final adapter.
    assert report["resume"]["checkpoint_path"] == str(latest.resolve())


def test_final_adapter_without_checkpoints_is_export_ready_but_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)

    report = build_run_status(str(run_dir))

    assert report["export"]["ready"] is True
    assert report["export"]["source_kind"] == "final_adapter"
    assert report["resume"] == {"ready": False, "checkpoint_path": None}
    assert report["checkpoints"]["valid_count"] == 0
    assert report["checkpoints"]["steps"] == []


# --------------------------------------------------------------------------
# 5. Invalid checkpoints are skipped
# --------------------------------------------------------------------------


def test_invalid_checkpoints_are_not_counted(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 50)
    latest = _write_checkpoint(run_dir, 100)

    # Missing trainer_state.json.
    no_state = run_dir / "checkpoint-75"
    no_state.mkdir()
    (no_state / "adapter_model.safetensors").write_text("weights")
    _write_adapter_config(no_state)

    # Has trainer state but no adapter artifact at all.
    no_adapter = run_dir / "checkpoint-125"
    no_adapter.mkdir()
    (no_adapter / "trainer_state.json").write_text("{}")

    # Not a `checkpoint-N` directory.
    (run_dir / "checkpoint-final").mkdir()

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["valid_count"] == 2
    assert report["checkpoints"]["steps"] == [50, 100]
    assert report["checkpoints"]["latest_step"] == 100
    assert report["resume"]["checkpoint_path"] == str(latest.resolve())


# --------------------------------------------------------------------------
# 6. Merged model discovery
# --------------------------------------------------------------------------


def test_merged_model_found_in_conventional_sibling_layout(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)

    report = build_run_status(str(run_dir))

    assert report["merged_model"] == {"present": True, "path": str(merged.resolve())}
    assert find_merged_model_dir(run_dir.resolve()) == merged.resolve()


def test_merged_model_absent_when_never_exported(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)

    report = build_run_status(str(run_dir))

    assert report["merged_model"] == {"present": False, "path": None}
    assert find_merged_model_dir(run_dir.resolve()) is None


def test_merged_dir_without_config_json_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_text("merged-weights")

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_merged_dir_without_safetensors_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_merged_dir_with_only_nested_safetensors_is_not_a_merged_model(tmp_path):
    """A tree holding adapters further down is not an exported merged model.

    A merged model keeps its weights at the directory root; a run tree that only
    contains `checkpoint-N/adapter_model.safetensors` must not read as already
    merged, or an operator would skip an export that never happened.
    """
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    _write_checkpoint(merged, 10)
    (merged / "config.json").write_text('{"model_type": "llama"}')

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_sharded_merged_model_is_recognised(tmp_path):
    """`save_pretrained` shards keep every weight file at the root."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "model.safetensors.index.json").write_text("{}")
    (merged / "model-00001-of-00002.safetensors").write_text("shard-1")
    (merged / "model-00002-of-00002.safetensors").write_text("shard-2")

    assert is_merged_model_dir(merged) is True
    assert build_run_status(str(run_dir))["merged_model"] == {
        "present": True,
        "path": str(merged.resolve()),
    }


def test_explicit_merged_dir_is_honored(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    elsewhere = _write_merged_model(tmp_path / "elsewhere" / "custom_merge")

    report = build_run_status(str(run_dir), merged_dir=str(elsewhere))
    assert report["merged_model"] == {"present": True, "path": str(elsewhere.resolve())}

    result = runner.invoke(app, ["run-status", str(run_dir), "--merged-dir", str(elsewhere)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {
        "present": True,
        "path": str(elsewhere.resolve()),
    }


def test_explicit_missing_merged_dir_reports_absent_and_exits_zero(runner, tmp_path):
    """A not-yet-exported merged model is an answer, not an error: still exit 0."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    missing = tmp_path / "elsewhere" / "not_exported_yet"

    report = build_run_status(str(run_dir), merged_dir=str(missing))
    assert report["merged_model"] == {"present": False, "path": None}

    result = runner.invoke(app, ["run-status", str(run_dir), "--merged-dir", str(missing)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {"present": False, "path": None}


# --------------------------------------------------------------------------
# 7. Safetensors policy
# --------------------------------------------------------------------------


def test_legacy_bin_adapter_is_rejected_by_default(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_legacy_bin_adapter(run_dir)

    report = build_run_status(str(run_dir))

    assert report["allow_unsafe_serialization"] is False
    assert report["final_adapter"] == {"present": False, "path": None}
    assert report["export"] == {"ready": False, "source_path": None, "source_kind": None}
    assert report["base_model"] is None


def test_legacy_bin_adapter_is_accepted_with_allow_unsafe(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_legacy_bin_adapter(run_dir)

    report = build_run_status(str(run_dir), allow_unsafe=True)

    assert report["allow_unsafe_serialization"] is True
    assert report["final_adapter"] == {"present": True, "path": str(run_dir.resolve())}
    assert report["export"]["ready"] is True
    assert report["export"]["source_kind"] == "final_adapter"
    assert report["base_model"] == "Qwen/Qwen3.5-0.5B"

    result = runner.invoke(app, ["run-status", str(run_dir), "--allow-unsafe-serialization"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == report


# --------------------------------------------------------------------------
# 8. Malformed adapter_config.json
# --------------------------------------------------------------------------


def test_malformed_adapter_config_yields_null_base_model(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_text("final-weights")
    (run_dir / "adapter_config.json").write_text("{not json at all")

    report = build_run_status(str(run_dir))

    assert report["base_model"] is None
    assert report["base_revision"] is None
    assert report["final_adapter"]["present"] is True

    result = runner.invoke(app, ["run-status", str(run_dir)])
    assert result.exit_code == 0


def test_adapter_config_without_base_model_key_yields_null_base_model(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir, base_model=None)

    report = build_run_status(str(run_dir))

    assert report["base_model"] is None
    assert report["base_revision"] is None
    assert report["final_adapter"]["present"] is True

    result = runner.invoke(app, ["run-status", str(run_dir)])
    assert result.exit_code == 0


@pytest.mark.parametrize("payload", ["[]", '["a", "b"]', '"a string"', "3", "null"])
def test_non_object_adapter_config_yields_null_base_model(runner, tmp_path, payload):
    """Valid JSON that is not an object must degrade, not crash.

    The checkpoint helpers call `.get(...)` on whatever `json.load` returns, so a
    list or scalar config raises `AttributeError` rather than a decode error.
    """
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_text("final-weights")
    (run_dir / "adapter_config.json").write_text(payload)

    report = build_run_status(str(run_dir))

    assert report["base_model"] is None
    assert report["base_revision"] is None
    assert report["final_adapter"]["present"] is True

    result = runner.invoke(app, ["run-status", str(run_dir)])
    assert result.exit_code == 0


# --------------------------------------------------------------------------
# 9. CLI exit codes
# --------------------------------------------------------------------------


def _assert_clean_exit(result, code):
    assert result.exit_code == code
    # A crash would surface here as something other than the exception click
    # raises for a controlled exit.
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_cli_missing_run_dir_exits_one(runner, tmp_path):
    result = runner.invoke(app, ["run-status", str(tmp_path / "does_not_exist")])

    _assert_clean_exit(result, 1)


def test_cli_parent_traversal_exits_one(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)

    result = runner.invoke(app, ["run-status", f"{run_dir}/../{run_dir.name}"])

    _assert_clean_exit(result, 1)


def test_cli_file_instead_of_directory_exits_nonzero(runner, tmp_path):
    a_file = tmp_path / "not_a_dir.txt"
    a_file.write_text("hello")

    result = runner.invoke(app, ["run-status", str(a_file)])

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_cli_reports_inspection_failure_as_exit_one(runner, tmp_path, monkeypatch):
    """A permission/IO failure while walking the run dir is a controlled exit.

    Path resolution succeeds, so the failure surfaces from report construction;
    it must still be a logged error and exit 1 rather than a raw traceback.
    """
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)

    def _boom(*args, **kwargs):
        raise PermissionError(f"Permission denied: {run_dir}")

    monkeypatch.setattr("agoge_forger.cli.build_run_status", _boom)

    result = runner.invoke(app, ["run-status", str(run_dir)])

    _assert_clean_exit(result, 1)


# --------------------------------------------------------------------------
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

    assert report["base_model"] == "Qwen/Qwen3.5-0.5B"
    assert report["base_revision"] == "deadbeefcafe"


def test_base_revision_is_null_when_absent(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 40)

    report = build_run_status(str(run_dir))

    assert report["base_model"] == "Qwen/Qwen3.5-0.5B"
    assert report["base_revision"] is None
