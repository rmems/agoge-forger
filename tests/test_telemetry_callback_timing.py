"""CPU tests for F0 correlation markers, window requests, and profile summaries."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agoge_forger.config import ExperimentConfig, ProfileWindowConfig
from agoge_forger.telemetry.callback import TrainingCorrelationCallback
from agoge_forger.telemetry.session import open_training_session


def _callback_session(tmp_path, monkeypatch, run_name, window):
    for key in ("AGOGE_PROFILE_WINDOW", "AGOGE_PROFILE_BACKEND", "AGOGE_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name=run_name)
    config.telemetry.profile_window = window
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
    session.callback = callback
    return session, callback


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


def test_unprofiled_baseline_waits_for_cuda_before_recording(tmp_path, monkeypatch):
    _, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "baseline",
        ProfileWindowConfig(enabled=True, phase="train", start_step=2, end_step=2),
    )
    clock = {"now": 1.0}
    monkeypatch.setattr("agoge_forger.telemetry.callback.time.perf_counter", lambda: clock["now"])
    monkeypatch.setattr("agoge_forger.telemetry.callback.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "agoge_forger.telemetry.callback.torch.cuda.synchronize",
        lambda: clock.update(now=3.0),
    )

    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)

    assert callback._unprofiled_s == [2.0]


def test_summary_waits_for_later_unprofiled_baseline(tmp_path, monkeypatch):
    session, callback = _callback_session(
        tmp_path,
        monkeypatch,
        "late-baseline",
        ProfileWindowConfig(enabled=True, phase="train", start_step=1, end_step=1),
    )
    clock = iter([1.0, 3.0, 4.0, 5.0])
    monkeypatch.setattr("agoge_forger.telemetry.callback.time.perf_counter", lambda: next(clock))

    class Profiler:
        def start(self):
            return None

        def step(self):
            return None

        def stop(self):
            return None

        def events(self):
            return []

        def key_averages(self):
            return []

    monkeypatch.setattr("agoge_forger.telemetry.callback.profile", lambda **kwargs: Profiler())

    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)
    assert not session.summary_path.exists()

    callback.on_step_begin(None, SimpleNamespace(global_step=1), None)
    callback.on_step_end(None, SimpleNamespace(global_step=2), None)
    callback.on_train_end(None, SimpleNamespace(global_step=2), None)

    summary = json.loads(session.summary_path.read_text())
    assert summary["overhead"]["status"] == "ok"
    assert summary["overhead"]["profiled_step_mean_s"] == pytest.approx(2.0)
    assert summary["overhead"]["unprofiled_step_mean_s"] == pytest.approx(1.0)
