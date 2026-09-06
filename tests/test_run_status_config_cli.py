"""Adapter-configuration and CLI failure-boundary tests for run-status."""

import json
import os

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.run_status import build_run_status, find_merged_model_dir
from tests.test_run_status import (
    SKIP_IF_ROOT,
    _deny_read_access_or_skip,
    _make_run_dir,
    _minimal_safetensors,
    _write_checkpoint,
    _write_final_adapter,
    _write_merged_model,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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
    # Source exists, but export-final-model will fail parsing this file.
    assert report["export"]["ready"] is False
    assert report["export"]["source_kind"] == "final_adapter"
    assert report["export"]["source_path"] == str(run_dir.resolve())

    result = runner.invoke(app, ["run-status", str(run_dir)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["schema_version"] == 1


def test_deeply_nested_adapter_config_yields_null_base_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(_minimal_safetensors())
    nested = '{"base_model_name_or_path":"org/model","nested":' + "[" * 100_000
    nested += "0" + "]" * 100_000 + "}"
    (run_dir / "adapter_config.json").write_text(nested)

    report = build_run_status(str(run_dir))

    assert report["base_model"] is None
    assert report["base_revision"] is None
    assert report["export"]["ready"] is False


@pytest.mark.parametrize("payload", [["org/model"], {"id": "org/model"}, True, 1])
def test_non_string_base_model_field_yields_null_base_model(tmp_path, payload):
    """A truthy non-string base_model_name_or_path must not leak into the report."""
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_text("final-weights")
    (run_dir / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": payload}))

    report = build_run_status(str(run_dir))

    assert report["base_model"] is None
    assert report["base_revision"] is None
    assert report["final_adapter"]["present"] is True
    dumped = json.loads(json.dumps(report))
    assert dumped["base_model"] is None


def test_adapter_config_without_base_model_key_yields_null_base_model(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir, base_model=None)

    report = build_run_status(str(run_dir))

    assert report["base_model"] is None
    assert report["base_revision"] is None
    assert report["final_adapter"]["present"] is True
    # Default export-final-model --run-dir requires a string base model.
    assert report["export"]["ready"] is False

    result = runner.invoke(app, ["run-status", str(run_dir)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["schema_version"] == 1


@pytest.mark.parametrize("peft_type", [None, "UNKNOWN_PEFT_TYPE"])
def test_missing_or_unknown_peft_type_is_not_export_ready(tmp_path, peft_type):
    run_dir = _make_run_dir(tmp_path)
    payload = {"base_model_name_or_path": "org/model"}
    if peft_type is not None:
        payload["peft_type"] = peft_type
    (run_dir / "adapter_config.json").write_text(json.dumps(payload))
    (run_dir / "adapter_model.safetensors").write_bytes(_minimal_safetensors())

    assert build_run_status(str(run_dir))["export"]["ready"] is False


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
    assert report["export"]["ready"] is False
    assert report["export"]["source_kind"] == "final_adapter"

    result = runner.invoke(app, ["run-status", str(run_dir)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["schema_version"] == 1


def test_find_merged_model_dir_valueerror_is_absent(tmp_path):
    """A '..' or not-a-dir merged_dir must return None, not raise."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    not_a_dir = tmp_path / "a_file.txt"
    not_a_dir.write_text("hello")

    assert find_merged_model_dir(run_dir, merged_dir=f"{tmp_path}/safe/../escape") is None
    assert find_merged_model_dir(run_dir, merged_dir=str(not_a_dir)) is None


def test_find_merged_model_dir_unresolvable_home_is_absent(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    assert (
        find_merged_model_dir(
            run_dir,
            merged_dir="~no-such-account-for-agoge-tests/merged/run",
        )
        is None
    )


def test_symlinked_adapter_run_finds_logical_merged_sibling(tmp_path):
    """A run dir that is a symlink must still find merged/<run_name>."""
    store = tmp_path / "store"
    (store / "adapters").mkdir(parents=True)
    real_run = tmp_path / "external" / "demo_run"
    real_run.mkdir(parents=True)
    _write_final_adapter(real_run)
    logical_run = store / "adapters" / "demo_run"
    logical_run.symlink_to(real_run)
    merged = _write_merged_model(store / "merged" / "demo_run")

    report = build_run_status(str(logical_run))

    assert report["merged_model"] == {"present": True, "path": str(merged.resolve())}
    # The resolved external tree has no sibling merged/ — that was the bug.
    assert not (tmp_path / "merged" / "demo_run").exists()


def test_cli_symlinked_run_dir_finds_logical_merged_sibling(runner, tmp_path):
    store = tmp_path / "store"
    (store / "adapters").mkdir(parents=True)
    real_run = tmp_path / "external" / "demo_run"
    real_run.mkdir(parents=True)
    _write_final_adapter(real_run)
    logical_run = store / "adapters" / "demo_run"
    logical_run.symlink_to(real_run)
    merged = _write_merged_model(store / "merged" / "demo_run")

    result = runner.invoke(app, ["run-status", str(logical_run)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {
        "present": True,
        "path": str(merged.resolve()),
    }


def test_logical_symlink_run_name_uses_logical_basename(tmp_path):
    """run_name must be the logical adapters/<name>, not the symlink target."""
    store = tmp_path / "store"
    (store / "adapters").mkdir(parents=True)
    real_run = tmp_path / "external" / "target-name"
    real_run.mkdir(parents=True)
    _write_final_adapter(real_run)
    logical_run = store / "adapters" / "logical-name"
    logical_run.symlink_to(real_run)
    merged = _write_merged_model(store / "merged" / "logical-name")

    report = build_run_status(str(logical_run))

    assert report["run_name"] == "logical-name"
    assert report["merged_model"] == {"present": True, "path": str(merged.resolve())}


def test_dot_run_dir_finds_conventional_merged_sibling(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path, name="demo_run")
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / "demo_run")
    monkeypatch.chdir(run_dir)

    report = build_run_status(".")

    assert report["run_name"] == "demo_run"
    assert report["merged_model"] == {"present": True, "path": str(merged.resolve())}


def test_short_relative_run_dir_finds_conventional_merged_sibling(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path, name="demo_run")
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / "demo_run")
    monkeypatch.chdir(run_dir.parent)

    report = build_run_status("demo_run")

    assert report["merged_model"] == {"present": True, "path": str(merged.resolve())}


@SKIP_IF_ROOT
def test_inaccessible_explicit_merged_dir_exits_one(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    protected = tmp_path / "protected"
    merged = _write_merged_model(protected / "merged")
    original_mode = protected.stat().st_mode
    _deny_read_access_or_skip(protected)
    try:
        result = runner.invoke(app, ["run-status", str(run_dir), "--merged-dir", str(merged)])
        with pytest.raises(PermissionError):
            build_run_status(str(run_dir), merged_dir=str(merged))
    finally:
        os.chmod(protected, original_mode)

    _assert_clean_exit(result, 1)


@SKIP_IF_ROOT
def test_inaccessible_conventional_merged_dir_exits_one(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged_parent = tmp_path / "merged"
    _write_merged_model(merged_parent / run_dir.name)
    original_mode = merged_parent.stat().st_mode
    _deny_read_access_or_skip(merged_parent)
    try:
        result = runner.invoke(app, ["run-status", str(run_dir)])
        with pytest.raises(PermissionError):
            build_run_status(str(run_dir))
    finally:
        os.chmod(merged_parent, original_mode)

    _assert_clean_exit(result, 1)


@SKIP_IF_ROOT
@pytest.mark.parametrize(
    "target_name",
    ["merged_config", "adapter_config", "trainer_state", "safetensors"],
)
def test_artifact_permission_error_exits_one(runner, tmp_path, target_name):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    checkpoint = _write_checkpoint(run_dir, 50)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    targets = {
        "merged_config": merged / "config.json",
        "adapter_config": run_dir / "adapter_config.json",
        "trainer_state": checkpoint / "trainer_state.json",
        "safetensors": run_dir / "adapter_model.safetensors",
    }
    target = targets[target_name]
    _deny_read_access_or_skip(target)
    try:
        result = runner.invoke(app, ["run-status", str(run_dir)])
        with pytest.raises(OSError):
            build_run_status(str(run_dir))
    finally:
        os.chmod(target, 0o644)

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


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


def test_cli_merged_dir_parent_traversal_exits_one(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)

    result = runner.invoke(
        app,
        ["run-status", str(run_dir), "--merged-dir", f"{tmp_path}/safe/../escape"],
    )

    _assert_clean_exit(result, 1)


def test_cli_existing_invalid_merged_dir_reports_absent(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    invalid = tmp_path / "not-a-merged-model"
    invalid.mkdir()

    result = runner.invoke(app, ["run-status", str(run_dir), "--merged-dir", str(invalid)])

    _assert_clean_exit(result, 0)
    assert json.loads(result.stdout)["merged_model"] == {"present": False, "path": None}


def test_cli_file_instead_of_directory_exits_nonzero(runner, tmp_path):
    a_file = tmp_path / "not_a_dir.txt"
    a_file.write_text("hello")

    result = runner.invoke(app, ["run-status", str(a_file)])

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_cli_reports_inspection_failure_as_exit_one(runner, tmp_path, monkeypatch, caplog):
    """A permission/IO failure while walking the run dir is a controlled exit.

    Path resolution succeeds, so the failure surfaces from report construction;
    it must still be a logged error and exit 1 rather than a raw traceback.
    """
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)

    def _boom(*args, **kwargs):
        raise PermissionError(f"Permission denied: {run_dir}")

    monkeypatch.setattr("agoge_forger.cli.build_run_status", _boom)

    with caplog.at_level("ERROR", logger="agoge"):
        result = runner.invoke(app, ["run-status", str(run_dir)])

    _assert_clean_exit(result, 1)
    assert f"Permission denied: {run_dir}" in caplog.messages


@pytest.mark.parametrize("flag", [None, "--merged-dir"])
def test_cli_unresolvable_home_directory_exits_one(runner, tmp_path, flag):
    """An unknown `~user` remains unresolved and fails as a missing path."""
    bad = "~no-such-account-for-agoge-tests/adapters/run"

    if flag is None:
        args = ["run-status", bad]
    else:
        run_dir = _make_run_dir(tmp_path)
        _write_final_adapter(run_dir)
        args = ["run-status", str(run_dir), flag, bad]

    result = runner.invoke(app, args)

    _assert_clean_exit(result, 1)


# --------------------------------------------------------------------------
