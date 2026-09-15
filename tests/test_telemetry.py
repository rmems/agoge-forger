"""CPU tests for F0 correlation markers, window requests, and profile summaries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.config import ExperimentConfig, ProfileWindowConfig, load_config
from agoge_forger.telemetry.backends import probe_profiler_backends, requested_backend_status
from agoge_forger.telemetry.join import join_profile_window, join_samples, load_jsonl
from agoge_forger.telemetry.markers import MarkerWriter
from agoge_forger.telemetry.overhead import measure_torch_profiler_overhead
from agoge_forger.telemetry.schema import SCHEMA_VERSION
from agoge_forger.telemetry.session import open_training_session
from agoge_forger.telemetry.summarize import classify_name, summarize_profiler
from agoge_forger.telemetry.window import (
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


def test_parse_window_rejects_garbage():
    with pytest.raises(ValueError):
        parse_profile_window("not-a-window")


def test_env_enables_window_yaml_left_off():
    telemetry = ExperimentConfig(model_id="m", dataset_path="x").telemetry
    updated = overlay_telemetry_env(
        telemetry,
        {"AGOGE_RUN_ID": "rid", "AGOGE_PROFILE_WINDOW": "train:1-1"},
    )
    assert updated.run_id == "rid"
    assert updated.profile_window.enabled is True
    assert profile_window_id("rid", updated.profile_window) == "rid:train:1-1"


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
    assert idle_sample["utilization_gpu"]["value"] == 0.0
    assert idle_marker["global_step"] == 0
    assert idle_sample["profile_window_ref"] is None
    window_marker, busy_sample = pairs[1]
    assert window_marker["global_step"] == 1
    assert busy_sample["profile_window_ref"] == summary["profile_window_id"]
    matched = join_profile_window(summary, samples)
    assert len(matched) == 1
    assert matched[0]["profile_window_ref"] == summary["profile_window_id"]


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
