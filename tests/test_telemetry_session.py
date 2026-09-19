"""CPU tests for F0 correlation markers, window requests, and profile summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.config import ExperimentConfig, ProfileWindowConfig
from agoge_forger.telemetry.markers import MarkerWriter
from agoge_forger.telemetry.session import open_training_session

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


def test_cli_profile_window_flag(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(f'model_id: "org/model"\ndataset_path: "{dataset.name}"\n')
    seen = {}

    def fake_train(cfg):
        seen["cfg"] = cfg

    monkeypatch.setattr("agoge_forger._cli_train._train_qlora", fake_train)
    result = CliRunner().invoke(
        app,
        [
            "train-qlora",
            "--config",
            str(config_path),
            "--run-id",
            "rid",
            "--profile-window",
            "train:2-3",
        ],
    )
    assert result.exit_code == 0, result.stdout
    window: ProfileWindowConfig = seen["cfg"].telemetry.profile_window
    assert seen["cfg"].telemetry.run_id == "rid"
    assert window.enabled is True
    assert window.start_step == 2
    assert window.end_step == 3


def test_cli_rejects_malformed_profile_window(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(f'model_id: "org/model"\ndataset_path: "{dataset.name}"\n')
    result = CliRunner().invoke(
        app,
        ["train-qlora", "--config", str(config_path), "--profile-window", "nope"],
    )
    assert result.exit_code == 1


def test_run_training_emits_failure_marker(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="failrun")
    monkeypatch.setattr(
        "agoge_forger.train.trainer.check_cuda_available",
        lambda required=True: (_ for _ in ()).throw(
            RuntimeError("CUDA is required but not available.")
        ),
    )
    from agoge_forger.train.trainer import run_training

    with pytest.raises(RuntimeError):
        run_training(config, producer_provenance=object())
    rows = [
        json.loads(line)
        for line in (tmp_path / "runs/failrun/telemetry/agoge-markers.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[-1]["event"] == "run_failed"
    assert rows[-1]["phase"] == "failed"


def test_disabled_window_does_not_publish_stale_summary(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="stale")
    session = open_training_session(config)
    session.summary_path.write_text('{"stale": true}\n', encoding="utf-8")
    entry = session.manifest_entry()
    assert entry["profile_window_id"] is None
    assert entry["profile_window"] is None


def test_reused_run_resets_marker_artifact_for_current_session(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="reused")
    config.telemetry.run_id = "same-run-id"

    first = open_training_session(config)
    first.emit("step_end", phase="train", global_step=1)
    second = open_training_session(config)

    rows = [json.loads(line) for line in second.markers_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["run_start"]
    assert rows[0]["agoge_run_id"] == "same-run-id"
    assert second.manifest_entry()["markers"] == str(second.markers_path)


def test_failed_marker_reset_does_not_publish_stale_artifact(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / "runs/reset-failure/telemetry/agoge-markers.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"old": true}\n')
    original_write_text = Path.write_text

    def fail_marker_reset(path, *args, **kwargs):
        if path.name == "agoge-markers.jsonl":
            raise OSError("reset failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_marker_reset)
    config = ExperimentConfig(
        model_id="org/model", dataset_path=str(dataset), run_name="reset-failure"
    )

    session = open_training_session(config)

    assert session.manifest_entry()["markers"] is None
    assert stale.read_text() == '{"old": true}\n'


def test_disabled_markers_do_not_publish_stale_marker_path(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="quiet")
    config.telemetry.emit_markers = False
    session = open_training_session(config)
    session.markers_path.write_text('{"stale": true}\n', encoding="utf-8")

    assert session.manifest_entry()["markers"] is None


def test_failed_current_marker_write_does_not_publish_stale_path(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / "runs/stale-marker/telemetry/agoge-markers.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"old": true}\n')
    monkeypatch.setattr(
        MarkerWriter,
        "_write",
        lambda self, event, phase, global_step, extras: (_ for _ in ()).throw(
            OSError("marker failed")
        ),
    )
    config = ExperimentConfig(
        model_id="org/model", dataset_path=str(dataset), run_name="stale-marker"
    )

    session = open_training_session(config)

    assert session.manifest_entry()["markers"] is None


def test_failed_current_request_write_does_not_publish_stale_path(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / "runs/stale-request/telemetry/profile-window-request.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"old": true}\n')
    original_write_text = Path.write_text

    def fail_request(path, *args, **kwargs):
        if path.name == "profile-window-request.json":
            raise OSError("request failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_request)
    config = ExperimentConfig(
        model_id="org/model", dataset_path=str(dataset), run_name="stale-request"
    )

    session = open_training_session(config)

    assert session.manifest_entry()["profile_window_request"] is None


def test_secondary_rank_does_not_write_markers(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RANK", "1")
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="rank1")
    session = open_training_session(config)
    session.emit("run_start", phase="setup", global_step=0)
    assert session.primary_rank is False
    assert not session.markers_path.exists()
    assert not session.request_path.exists()
