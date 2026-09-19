"""CPU tests for F0 correlation markers, window requests, and profile summaries."""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.config import ExperimentConfig, ProfileWindowConfig, TelemetryConfig, load_config
from agoge_forger.telemetry.backends import probe_profiler_backends, requested_backend_status
from agoge_forger.telemetry.callback import TrainingCorrelationCallback
from agoge_forger.telemetry.join import join_profile_window, join_samples, load_jsonl
from agoge_forger.telemetry.markers import MarkerWriter
from agoge_forger.telemetry.overhead import measure_torch_profiler_overhead
from agoge_forger.telemetry.schema import SCHEMA_VERSION
from agoge_forger.telemetry.session import open_training_session
from agoge_forger.telemetry.summarize import (
    classify_name,
    overhead_from_step_times,
    summarize_profiler,
)
from agoge_forger.telemetry.summarize_families import duration_block
from agoge_forger.telemetry.summarize_names import transfer_direction
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


def test_marker_monotonic_and_unpinned_revision(tmp_path):
    writer = _writer(tmp_path)
    writer.emit("run_start", phase="setup", global_step=0)
    writer.emit(
        "step_end",
        phase="train",
        global_step=1,
        extras={"loss": {"value": 1.0, "unit": "1", "status": "ok"}},
    )
    rows = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert rows[0]["schema_version"] == SCHEMA_VERSION
    assert rows[0]["record_kind"] == "agoge_marker"
    assert rows[0]["model_revision"] == "unpinned"
    assert rows[0]["monotonic_ns"] < rows[1]["monotonic_ns"]
    assert rows[1]["loss"]["status"] == "ok"


def test_marker_clock_ties_still_increase(tmp_path):
    writer = _writer(tmp_path, clock=[5, 5, 5])
    writer.emit("a", phase="train", global_step=0)
    writer.emit("b", phase="train", global_step=1)
    rows = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert rows[1]["monotonic_ns"] == rows[0]["monotonic_ns"] + 1


def test_open_session_records_unsupported_nsight(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="sess")
    session = open_training_session(config)
    request = json.loads(session.request_path.read_text())
    assert request["enabled"] is False
    assert request["backends"]["nsight_systems"]["status"] == "unsupported"
    assert request["backends"]["cupti"]["status"] == "unsupported"
    assert request["backends"]["torch"]["status"] == "ok"
    markers = [json.loads(line) for line in session.markers_path.read_text().splitlines()]
    assert markers[0]["event"] == "run_start"


def test_join_markers_to_gpu_samples_and_profile_window():
    markers = load_jsonl(FIXTURES / "agoge-markers.jsonl")
    samples = load_jsonl(FIXTURES / "bkl-gpu-samples.jsonl")
    summary = json.loads((FIXTURES / "profile-window.json").read_text())
    pairs = join_samples(markers, samples)
    assert len(pairs) == 2
    idle_marker, idle_sample = pairs[0]
    assert int(idle_sample["utilization_gpu"]["value"]) == 0
    assert idle_marker["global_step"] == 0
    assert idle_sample["profile_window_ref"] is None
    window_marker, busy_sample = pairs[1]
    assert window_marker["global_step"] == 1
    assert busy_sample["profile_window_ref"] == summary["profile_window_id"]
    matched = join_profile_window(summary, samples)
    assert len(matched) == 1
    assert matched[0]["profile_window_ref"] == summary["profile_window_id"]
    other_host = dict(samples[1])
    other_host["host"] = {"hostname": "other-host"}
    other_gpu = dict(samples[1])
    other_gpu["gpu"] = {**samples[1]["gpu"], "uuid": "GPU-other"}
    assert join_profile_window(summary, [other_host, other_gpu]) == []


def test_join_profile_window_rejects_missing_interval_bounds():
    summary = {
        "agoge_run_id": "run_test",
        "host": {"hostname": "testhost"},
        "gpu": {"uuid": "GPU-test"},
        "profile_window_id": "run_test:train:5-5",
        "monotonic_ns_start": None,
        "monotonic_ns_end": None,
    }
    sample = {
        "agoge_run_id": "run_test",
        "host": {"hostname": "testhost"},
        "gpu": {"uuid": "GPU-test"},
        "profile_window_ref": None,
        "monotonic_ns": 25,
    }
    assert join_profile_window(summary, [sample]) == []


def test_unknown_backend_is_unsupported_not_guessed():
    probes = probe_profiler_backends()
    status = requested_backend_status("made_up", probes)
    assert status["status"] == "unsupported"
    assert "not guessed" in status["reason"]


def test_classify_kernel_families():
    assert classify_name("ampere_sgemm") == "gemm"
    assert classify_name("flash_attn_fwd") == "attention"
    assert classify_name("MemcpyHtoD") == "transfer"
    assert classify_name("cudaDeviceSynchronize") == "sync"


def test_summarize_records_cuda_unavailable_without_inventing_kernels():
    fake = SimpleNamespace(
        events=lambda: [
            SimpleNamespace(
                name="aten::mm", duration=12.0, device_type=SimpleNamespace(name="CPU")
            ),
        ],
        key_averages=list,
    )
    summary = summarize_profiler(fake, cuda_kernel_status="unavailable")
    assert summary["kernel_duration"]["status"] == "unavailable"
    assert summary["kernel_duration"]["value"] is None
    assert summary["cpu_ops"][0]["name"] == "aten::mm"


def test_key_average_summaries_weight_each_invocation():
    fake = SimpleNamespace(
        events=list,
        key_averages=lambda: [
            SimpleNamespace(
                key="MemcpyHtoD",
                count=3,
                device_time_total=30.0,
                cpu_time_total=0.0,
            ),
            SimpleNamespace(
                key="MemcpyDtoH",
                count=1,
                device_time_total=20.0,
                cpu_time_total=0.0,
            ),
            SimpleNamespace(
                key="aten::add",
                count=4,
                device_time_total=0.0,
                cpu_time_total=8.0,
            ),
        ],
    )

    summary = summarize_profiler(fake, cuda_kernel_status="ok")

    assert summary["kernel_duration"]["count"] == 4
    assert summary["kernel_duration"]["sum"] == 50.0
    assert summary["kernel_duration"]["p50"] == 10.0
    assert summary["kernel_duration"]["p95"] == 20.0
    assert summary["transfers"]["h2d_us"]["count"] == 3
    assert summary["transfers"]["h2d_us"]["sum"] == 30.0
    assert summary["transfers"]["d2h_us"]["count"] == 1
    assert summary["cpu_ops"][0]["count"] == 4
    assert summary["cpu_ops"][0]["duration_us"]["sum"] == 8.0


def test_weighted_summary_memory_is_bounded_by_rows_not_invocations():
    tracemalloc.start()
    try:
        result = duration_block(
            [{"device": "cuda", "duration_us": 2.0, "count": 1_000_000}],
            "cuda",
            "ok",
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result["count"] == 1_000_000
    assert result["sum"] == 2_000_000.0
    assert peak < 1_000_000


def test_transfer_evidence_requires_cuda_and_explicit_direction():
    fake = SimpleNamespace(
        events=lambda: [
            SimpleNamespace(
                name="MemcpyDtoD",
                duration=7.0,
                device_type=SimpleNamespace(name="CUDA"),
            ),
            SimpleNamespace(
                name="copy_kernel",
                duration=11.0,
                device_type=SimpleNamespace(name="CPU"),
            ),
        ],
        key_averages=list,
    )

    summary = summarize_profiler(fake, cuda_kernel_status="ok")

    assert transfer_direction("MemcpyDtoD") is None
    assert transfer_direction("copy_kernel") is None
    assert summary["transfers"] == {
        "status": "unavailable",
        "h2d_us": None,
        "d2h_us": None,
    }


def test_overhead_is_unavailable_for_nonpositive_baseline():
    result = overhead_from_step_times([2.0], [0.0])
    assert result["status"] == "unavailable"
    assert result["reason"] == "unprofiled optimizer-step mean is not positive"
    assert result["ratio"] is None


def test_overhead_microbenchmark_is_finite():
    result = measure_torch_profiler_overhead(repeats=4, size=32)
    assert result["unprofiled_s"]["status"] == "ok"
    assert result["profiled_s"]["status"] == "ok"
    assert result["profiled_s"]["value"] > 0
    assert result["ratio"]["status"] == "ok"


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


def test_unstarted_window_does_not_emit_end(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="missed")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=5, end_step=5
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
    state = SimpleNamespace(global_step=1)
    callback.on_step_end(None, state, None)
    callback.on_train_end(None, state, None)
    events = [json.loads(line)["event"] for line in session.markers_path.read_text().splitlines()]
    assert "profile_window_start" not in events
    assert "profile_window_end" not in events


def test_unsupported_backend_closes_at_end_step(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="nsight")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=1, backend="nsight_systems"
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
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
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="partial")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=3, backend="nsight_systems"
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
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
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="advance")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=2
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)

    class BrokenProfiler:
        def step(self):
            raise RuntimeError("step failed")

    callback._profiler = BrokenProfiler()
    callback._profiler_active = True
    callback._window_started = True
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)

    assert session.profiler_error == "step failed"


def test_profiler_synchronization_failure_is_nonfatal_and_recorded(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="sync")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=1
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)

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
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="timing")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=2
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
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
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="startup")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=2
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
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
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="failed")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=3, backend="nsight_systems"
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
    session.callback = callback
    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)

    session.close()

    summary = json.loads(session.summary_path.read_text())
    assert summary["complete"] is False
    assert summary["actual_end_step"] == 1
    events = [json.loads(line)["event"] for line in session.markers_path.read_text().splitlines()]
    assert "profile_window_end" in events


def test_failure_before_first_window_step_has_no_false_end_marker(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="midstep")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=3, backend="nsight_systems"
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
    session.callback = callback
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
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(model_id="org/model", dataset_path=str(dataset), run_name="startfail")
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True, phase="train", start_step=1, end_step=1
    )
    session = open_training_session(config)
    callback = TrainingCorrelationCallback(session)
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


def test_summarize_prefers_device_time_and_memory_counters():
    fake = SimpleNamespace(
        events=lambda: [
            SimpleNamespace(
                name="aten::empty",
                duration=9.0,
                cpu_time_total=9.0,
                device_time_total=40.0,
                device_type=SimpleNamespace(name="CUDA"),
                cpu_memory_usage=128.0,
                device_memory_usage=256.0,
            ),
        ],
        key_averages=list,
    )
    summary = summarize_profiler(fake, cuda_kernel_status="ok")
    assert summary["kernel_duration"]["status"] == "ok"
    assert int(summary["kernels"][0]["duration_us"]["sum"]) == 40
    assert summary["memory"]["status"] == "ok"
    assert int(summary["memory"]["device_bytes"]["sum"]) == 256


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
