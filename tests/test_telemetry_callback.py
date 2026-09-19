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


def test_unstarted_window_does_not_emit_end(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "missed",
        ProfileWindowConfig(enabled=True, phase="train", start_step=5, end_step=5),
    )
    state = SimpleNamespace(global_step=1)
    callback.on_step_end(None, state, None)
    callback.on_train_end(None, state, None)
    events = [json.loads(line)["event"] for line in session.markers_path.read_text().splitlines()]
    assert "profile_window_start" not in events
    assert "profile_window_end" not in events


def test_unsupported_backend_closes_at_end_step(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "nsight",
        ProfileWindowConfig(
            enabled=True, phase="train", start_step=1, end_step=1, backend="nsight_systems"
        ),
    )
    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)
    events = [json.loads(line)["event"] for line in session.markers_path.read_text().splitlines()]
    assert "step_begin" in events
    assert "profile_window_start" in events
    assert "profile_window_end" in events
    summary = json.loads(session.summary_path.read_text())
    assert summary["overhead"]["status"] == "unavailable"
    assert summary["monotonic_ns_start"] is not None
    assert summary["monotonic_ns_end"] is not None


def test_partial_window_records_actual_end_step(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "partial",
        ProfileWindowConfig(
            enabled=True, phase="train", start_step=1, end_step=3, backend="nsight_systems"
        ),
    )
    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)
    callback.on_train_end(None, SimpleNamespace(global_step=1), None)

    summary = json.loads(session.summary_path.read_text())
    assert summary["start_step"] == 1
    assert summary["end_step"] == 3
    assert summary["actual_end_step"] == 1
    assert summary["complete"] is False
    markers = [json.loads(line) for line in session.markers_path.read_text().splitlines()]
    end = next(row for row in markers if row["event"] == "profile_window_end")
    assert end["global_step"] == 1


def test_profiler_advance_failure_is_nonfatal_and_recorded(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "advance",
        ProfileWindowConfig(enabled=True, phase="train", start_step=1, end_step=2),
    )

    class BrokenProfiler:
        def step(self):
            raise RuntimeError("step failed")

    callback._profiler = BrokenProfiler()
    callback._profiler_active = True
    callback._window_started = True
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)

    assert session.profiler_error == "step failed"


def test_profiler_synchronization_failure_is_nonfatal_and_recorded(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "sync",
        ProfileWindowConfig(enabled=True, phase="train", start_step=1, end_step=1),
    )

    class Profiler:
        def step(self):
            return None

        def stop(self):
            return None

        def events(self):
            return []

        def key_averages(self):
            return []

    callback._profiler = Profiler()
    callback._profiler_active = True
    callback._window_started = True
    callback._window_start_ns = 1
    monkeypatch.setattr("agoge_forger.telemetry.callback.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "agoge_forger.telemetry.callback.torch.cuda.synchronize",
        lambda: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    callback.on_step_end(None, SimpleNamespace(global_step=1), None)

    assert session.profiler_error == "sync failed"


def test_profiled_step_duration_includes_profiler_lifecycle(tmp_path, monkeypatch):
    _, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "timing",
        ProfileWindowConfig(enabled=True, phase="train", start_step=1, end_step=2),
    )
    clock = {"now": 1.0}
    monkeypatch.setattr("agoge_forger.telemetry.callback.time.perf_counter", lambda: clock["now"])

    class AdvancingProfiler:
        def step(self):
            clock["now"] = 3.0

    callback._profiler = AdvancingProfiler()
    callback._profiler_active = True
    callback._window_started = True
    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)

    assert callback._profiled_s == [2.0]


def test_profiled_step_duration_includes_profiler_startup(tmp_path, monkeypatch):
    _, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "startup",
        ProfileWindowConfig(enabled=True, phase="train", start_step=1, end_step=2),
    )
    clock = {"now": 1.0}
    monkeypatch.setattr("agoge_forger.telemetry.callback.time.perf_counter", lambda: clock["now"])

    class StartingProfiler:
        def start(self):
            clock["now"] = 3.0

        def step(self):
            return None

    monkeypatch.setattr(
        "agoge_forger.telemetry.callback.profile",
        lambda **kwargs: StartingProfiler(),
    )

    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)

    assert callback._profiled_s == [2.0]


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


def test_profiler_start_failure_attempts_cleanup(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "startfail",
        ProfileWindowConfig(enabled=True, phase="train", start_step=1, end_step=1),
    )
    cleaned = {"value": False}

    class FailingProfiler:
        def start(self):
            raise RuntimeError("start failed")

        def stop(self):
            cleaned["value"] = True

    monkeypatch.setattr(
        "agoge_forger.telemetry.callback.profile",
        lambda **kwargs: FailingProfiler(),
    )

    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)

    assert cleaned["value"] is True
    assert session.profiler_error == "start failed"
