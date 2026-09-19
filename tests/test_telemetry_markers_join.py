"""CPU tests for F0 correlation markers, window requests, and profile summaries."""

from __future__ import annotations

import json
from pathlib import Path

from agoge_forger.config import ExperimentConfig
from agoge_forger.telemetry.backends import probe_profiler_backends, requested_backend_status
from agoge_forger.telemetry.join import join_profile_window, join_samples, load_jsonl
from agoge_forger.telemetry.markers import MarkerWriter
from agoge_forger.telemetry.schema import SCHEMA_VERSION
from agoge_forger.telemetry.session import open_training_session
from agoge_forger.telemetry.summarize import (
    classify_name,
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
