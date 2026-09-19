"""CPU tests for F0 correlation markers, window requests, and profile summaries."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agoge_forger.config import ExperimentConfig, ProfileWindowConfig
from agoge_forger.telemetry.callback import TrainingCorrelationCallback
from agoge_forger.telemetry.session import open_training_session


def _callback_session(tmp_path, monkeypatch, run_name, window):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name=run_name)
    config.telemetry.profile_window = window
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
    session.callback = callback
    return session, callback


def test_session_close_finalizes_active_window_after_training_error(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "failed",
        ProfileWindowConfig(
            enabled=True, phase="train", start_step=1, end_step=3, backend="nsight_systems"
        ),
    )
    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)

    session.close()

    summary = json.loads(session.summary_path.read_text())
    assert summary["complete"] is False
    assert summary["actual_end_step"] == 1
    events = [json.loads(line)["event"] for line in session.markers_path.read_text().splitlines()]
    assert "profile_window_end" in events


def test_failure_before_first_window_step_has_no_false_end_marker(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "midstep",
        ProfileWindowConfig(
            enabled=True, phase="train", start_step=1, end_step=3, backend="nsight_systems"
        ),
    )
    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)

    session.record_failure()
    session.close()

    summary = json.loads(session.summary_path.read_text())
    assert summary["complete"] is False
    assert summary["actual_end_step"] is None
    events = [json.loads(line)["event"] for line in session.markers_path.read_text().splitlines()]
    assert "profile_window_end" not in events
    assert events[-1] == "run_failed"


def test_failure_marker_preserves_last_completed_step(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "failed-after-step",
        ProfileWindowConfig(enabled=False),
    )
    callback.on_step_end(None, SimpleNamespace(global_step=3), None)

    session.record_failure()

    rows = [json.loads(line) for line in session.markers_path.read_text().splitlines()]
    assert rows[-1]["event"] == "run_failed"
    assert rows[-1]["global_step"] == 3
