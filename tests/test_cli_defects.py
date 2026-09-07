"""Regression tests for the CLI defects filed as #127.

Each of these failed before the accompanying fix: a config-file dry run that
launched for real, two commands that answered an ordinary operator mistake with
a traceback, and an inspection command that crashed on every valid file.
"""

import json
import os

import pytest
import typer
from typer.testing import CliRunner

from agoge_forger._cli_serving import (
    _DEFAULT_SMOKE_BASE_URL,
    _fill_smoke_defaults,
    _merge_serving_config,
)
from agoge_forger.artifacts.safetensors_io import inspect_safetensors_file
from agoge_forger.cli import app
from tests.test_run_status import _make_run_dir, _minimal_safetensors, _write_final_adapter


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _assert_clean_exit(result, code):
    assert result.exit_code == code, result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


# --- serve-vllm: an unset --dry-run must defer to the config file ------------


def _serving_config(tmp_path, **fields):
    path = tmp_path / "serving.yaml"
    body = "\n".join(f"{key}: {value}" for key, value in fields.items())
    path.write_text(body + "\n")
    return str(path)


def _captured_serve(monkeypatch):
    """Capture the ServingConfig the command hands to the launcher."""
    seen = {}

    def fake_serve(cfg):
        seen["cfg"] = cfg
        return 0

    monkeypatch.setattr("agoge_forger._cli_serving._serve_vllm", fake_serve)
    return seen


def test_config_dry_run_is_honored_when_the_flag_is_unset(runner, tmp_path, monkeypatch):
    """The CLI default was False, not None, so it always beat the config file and
    `dry_run: true` silently launched vLLM for real. Driven through the command so
    the default is what is under test."""
    config = _serving_config(tmp_path, model="example/model", dry_run="true")
    seen = _captured_serve(monkeypatch)

    result = runner.invoke(app, ["serve-vllm", "--config", config])

    _assert_clean_exit(result, 0)
    assert seen["cfg"].dry_run is True


def test_explicit_no_dry_run_overrides_the_config(runner, tmp_path, monkeypatch):
    config = _serving_config(tmp_path, model="example/model", dry_run="true")
    seen = _captured_serve(monkeypatch)

    result = runner.invoke(app, ["serve-vllm", "--config", config, "--no-dry-run"])

    _assert_clean_exit(result, 0)
    assert seen["cfg"].dry_run is False


def test_explicit_dry_run_flag_still_applies(runner, tmp_path, monkeypatch):
    config = _serving_config(tmp_path, model="example/model")
    seen = _captured_serve(monkeypatch)

    result = runner.invoke(app, ["serve-vllm", "--config", config, "--dry-run"])

    _assert_clean_exit(result, 0)
    assert seen["cfg"].dry_run is True


def test_explicit_dry_run_without_a_config_still_applies():
    cfg = _merge_serving_config(None, {"model": "example/model", "dry_run": True})

    assert cfg.dry_run is True


def test_dry_run_defaults_off_without_a_config():
    cfg = _merge_serving_config(None, {"model": "example/model", "dry_run": None})

    assert cfg.dry_run is False


@pytest.mark.parametrize("body", ["- a\n- b\n", "just a string\n", "42\n"])
def test_non_mapping_serving_config_is_a_usage_error(tmp_path, body):
    """`load_serving_config` signals a non-mapping config with TypeError, which no
    boundary caught — so `serve-vllm --config` on a sequence gave a traceback.
    smoke-vllm's side of this was already closed; this is the serving path."""
    path = tmp_path / "serving.yaml"
    path.write_text(body)

    with pytest.raises(typer.BadParameter):
        _merge_serving_config(str(path), {"model": "example/model"})


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


def test_non_mapping_chat_config_is_a_usage_error(runner, tmp_path):
    """A YAML sequence reached `data[key]` and crashed with a raw TypeError."""
    config = tmp_path / "chat.yaml"
    config.write_text("- not\n- a mapping\n")

    result = runner.invoke(app, ["smoke-vllm", "--config", str(config)])

    assert result.exit_code != 0
    assert "YAML mapping" in result.output


@pytest.mark.parametrize("value", [0, False, []])
def test_invalid_falsey_base_url_is_not_defaulted(value):
    """`not data.get(...)` turned 0/False/[] into a silent call to localhost."""
    data = {"base_url": value}

    _fill_smoke_defaults(data)

    assert data["base_url"] == value


@pytest.mark.parametrize("value", [None, ""])
def test_absent_base_url_takes_the_default(value):
    data = {"base_url": value}

    _fill_smoke_defaults(data)

    assert data["base_url"] == _DEFAULT_SMOKE_BASE_URL


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
