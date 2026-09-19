"""CPU tests for F0 correlation markers, window requests, and profile summaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.config import ExperimentConfig, ProfileWindowConfig, TelemetryConfig, load_config
from agoge_forger.telemetry.markers import MarkerWriter
from agoge_forger.telemetry.session import open_training_session
from agoge_forger.telemetry.window import (
    overlay_cli_telemetry,
    overlay_telemetry_env,
    parse_profile_window,
    profile_window_id,
)

FIXTURES = Path("tests/fixtures/f0-correlation")


def _writer(tmp_path: Path, clock: list[int] | None = None) -> MarkerWriter:
    ticks = clock or [10, 20, 30, 40]

    def _clock() -> int:
        return ticks.pop(0)

    return MarkerWriter(
        path=tmp_path / "agoge-markers.jsonl",
        run_id="run_test",
        hostname="testhost",
        gpu={"uuid": "GPU-test", "pci_bus_id": None, "name": None, "compute_capability": None},
        collector_version="test",
        model_id="org/model",
        model_revision=None,
        dataset={"id": "tiny.jsonl", "split": "train", "config_digest": None},
        _clock=_clock,
    )


def test_parse_compact_and_kv_windows():
    compact = parse_profile_window("train:1-2")
    assert compact.enabled is True
    assert compact.phase == "train"
    assert compact.start_step == 1
    assert compact.end_step == 2
    kv = parse_profile_window("phase=train,start_step=2,end_step=3")
    assert kv.start_step == 2
    assert kv.end_step == 3


def test_parse_kv_window_honors_strict_disabled_value():
    disabled = parse_profile_window("enabled=false,phase=train,start_step=1,end_step=1")
    assert disabled.enabled is False
    with pytest.raises(ValueError, match="must be true or false"):
        parse_profile_window("enabled=maybe,phase=train,start_step=1,end_step=1")


def test_parse_window_rejects_zero_start_step():
    with pytest.raises(ValueError, match=">= 1"):
        parse_profile_window("train:0-1")


def test_parse_window_rejects_garbage():
    with pytest.raises(ValueError):
        parse_profile_window("not-a-window")


def test_parse_window_rejects_non_integer_steps():
    with pytest.raises(ValueError, match="must be an integer"):
        parse_profile_window("phase=train,start_step=abc,end_step=2")


def test_parse_window_rejects_unknown_and_duplicate_fields():
    with pytest.raises(ValueError, match="unknown profile window field"):
        parse_profile_window("phase=train,start_stpe=1,end_step=2")
    with pytest.raises(ValueError, match="duplicate profile window field"):
        parse_profile_window("phase=train,start_step=1,start_step=2")


def test_parse_window_rejects_eval_phase():
    with pytest.raises(ValueError, match="must be 'train'"):
        parse_profile_window("eval:1-2")


def test_cli_window_skips_malformed_environment(monkeypatch):
    telemetry = ExperimentConfig(model_id="m", dataset_path="x").telemetry
    monkeypatch.setenv("AGOGE_PROFILE_WINDOW", "not-a-window")
    updated = overlay_cli_telemetry(telemetry, None, "train:2-3")
    assert updated.profile_window.start_step == 2
    assert updated.profile_window.end_step == 3


def test_open_session_does_not_reapply_environment_over_cli_window(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGOGE_PROFILE_BACKEND", "nsight_systems")
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="cli")
    config.telemetry = overlay_cli_telemetry(config.telemetry, None, "train:1-1")

    session = open_training_session(config)

    assert session.window.backend == "torch"


def test_non_mapping_telemetry_is_usage_error(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        f'model_id: "org/model"\ndataset_path: "{dataset.name}"\ntelemetry: []\n'
    )
    with pytest.raises(TypeError, match="telemetry must be a mapping"):
        load_config(str(config_path))
    result = CliRunner().invoke(app, ["train-qlora", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "Traceback" not in (result.stdout + (result.stderr or ""))


def test_env_enables_window_yaml_left_off():
    telemetry = ExperimentConfig(model_id="m", dataset_path="x").telemetry
    updated = overlay_telemetry_env(
        telemetry,
        {"AGOGE_RUN_ID": "rid", "AGOGE_PROFILE_WINDOW": "train:1-1"},
    )
    assert updated.run_id == "rid"
    assert updated.profile_window.enabled is True
    assert profile_window_id("rid", updated.profile_window) == "rid:train:1-1"


def test_environment_overrides_nondefault_yaml_telemetry():
    telemetry = ExperimentConfig(model_id="m", dataset_path="x").telemetry
    telemetry.run_id = "yaml-run"
    telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=1, backend="torch"
    )

    updated = overlay_telemetry_env(
        telemetry,
        {
            "AGOGE_RUN_ID": "env-run",
            "AGOGE_PROFILE_WINDOW": "train:3-4",
            "AGOGE_PROFILE_BACKEND": "nsight_systems",
        },
    )

    assert updated.run_id == "env-run"
    assert updated.profile_window.start_step == 3
    assert updated.profile_window.end_step == 4
    assert updated.profile_window.backend == "nsight_systems"


def test_programmatic_session_resolves_environment_once(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGOGE_RUN_ID", "env-run")
    monkeypatch.setenv("AGOGE_PROFILE_WINDOW", "train:2-3")
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="api")

    session = open_training_session(config)

    assert session.run_id == "env-run"
    assert session.window.start_step == 2
    assert session.window.end_step == 3


def test_external_resolution_sentinel_cannot_bypass_environment(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGOGE_RUN_ID", "env-run")
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="api")
    config.telemetry = TelemetryConfig.model_validate(
        {"run_id": "yaml-run", "environment_resolved": True}
    )

    session = open_training_session(config)

    assert session.run_id == "env-run"


def test_yaml_telemetry_section(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        "model_id: org/model\n"
        "dataset_path: data.jsonl\n"
        "telemetry:\n"
        "  run_id: yaml-run\n"
        "  profile_window:\n"
        "    enabled: true\n"
        "    phase: train\n"
        "    start_step: 1\n"
        "    end_step: 1\n"
    )
    config = load_config(str(config_path))
    assert config.telemetry.run_id == "yaml-run"
    assert config.telemetry.profile_window.enabled is True
    assert config.telemetry.profile_window.end_step == 1
