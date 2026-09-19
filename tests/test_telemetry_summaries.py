"""CPU tests for F0 correlation markers, window requests, and profile summaries."""

from __future__ import annotations

import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

from agoge_forger.telemetry.markers import MarkerWriter
from agoge_forger.telemetry.overhead import measure_torch_profiler_overhead
from agoge_forger.telemetry.summarize import (
    overhead_from_step_times,
    summarize_profiler,
)
from agoge_forger.telemetry.summarize_families import duration_block
from agoge_forger.telemetry.summarize_names import transfer_direction

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
    assert summary["kernel_duration"]["sum"] == pytest.approx(50.0)
    assert summary["kernel_duration"]["p50"] == pytest.approx(10.0)
    assert summary["kernel_duration"]["p95"] == pytest.approx(20.0)
    assert summary["transfers"]["h2d_us"]["count"] == 3
    assert summary["transfers"]["h2d_us"]["sum"] == pytest.approx(30.0)
    assert summary["transfers"]["d2h_us"]["count"] == 1
    assert summary["cpu_ops"][0]["count"] == 4
    assert summary["cpu_ops"][0]["duration_us"]["sum"] == pytest.approx(8.0)


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
    assert result["sum"] == pytest.approx(2_000_000.0)
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
