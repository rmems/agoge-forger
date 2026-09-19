"""Open/close a per-run telemetry session without owning trainer orchestration."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import ExperimentConfig, ProfileWindowConfig
from ..logging import logger
from ..split_schema import sha256_file
from .backends import probe_profiler_backends, requested_backend_status
from .callback import TrainingCorrelationCallback
from .gpu import gpu_identity
from .markers import MarkerWriter
from .schema import (
    RECORD_PROFILE_REQUEST,
    RECORD_PROFILE_WINDOW,
    TORCH_BACKEND,
    UNPINNED_REVISION,
    EnvelopeIdentity,
    envelope,
)
from .window import overlay_telemetry_env, profile_window_id


@dataclass
class TelemetrySession:
    run_id: str
    run_dir: Path
    telemetry_dir: Path
    markers_path: Path
    request_path: Path
    summary_path: Path
    writer: MarkerWriter
    window: ProfileWindowConfig
    backend_probes: dict[str, dict[str, Any]]
    window_id: str | None
    emit_markers: bool = True
    primary_rank: bool = True
    profiler_error: str | None = None
    _closed: bool = field(default=False, init=False)
    _summary_written: bool = field(default=False, init=False)
    _request_written: bool = field(default=False, init=False)
    callback: Any = field(default=None, init=False, repr=False)

    def emit(
        self,
        event: str,
        *,
        phase: str,
        global_step: int,
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.emit_markers:
            return
        self.writer.emit(event, phase=phase, global_step=global_step, extras=extras)

    def record_failure(self) -> None:
        if self.callback is not None:
            self.callback.finalize()
        self.emit("run_failed", phase="failed", global_step=0)

    def write_profile_summary(
        self,
        *,
        summary: Mapping[str, Any],
    ) -> None:
        if not self._can_write_summary():
            return
        try:
            self._write_summary(summary)
            self._summary_written = True
        except OSError as error:
            logger.warning(f"profile window summary write failed: {error}")

    def close(self) -> None:
        if self._closed:
            return
        if self.callback is not None:
            self.callback.finalize()
        self._closed = True
        if self.window.enabled and not self._summary_written:
            self.write_profile_summary(
                summary={
                    "events": None,
                    "overhead": {
                        "status": "unavailable",
                        "reason": "profile window start step was not reached",
                    },
                    "start_ns": None,
                    "end_ns": None,
                    "actual_end_step": None,
                    "complete": False,
                },
            )

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "agoge_run_id": self.run_id,
            "markers": self._published_markers_path(),
            "profile_window_request": self._published_request_path(),
            "profile_window": self._published_summary_path(),
            "profile_window_id": self.window_id,
        }

    def _write_summary(self, summary: Mapping[str, Any]) -> None:
        events = summary["events"]
        start_ns = summary["start_ns"]
        end_ns = summary["end_ns"]
        stamp_utc, monotonic_ns = self.writer.stamp()
        backend = requested_backend_status(self.window.backend, self.backend_probes)
        record = envelope(
            record_kind=RECORD_PROFILE_WINDOW,
            identity=self.envelope_identity(),
            monotonic_ns=end_ns if end_ns is not None else monotonic_ns,
        )
        record.update(
            {
                "timestamp_utc": stamp_utc,
                "profile_window_id": self.window_id,
                "phase": self.window.phase,
                "start_step": self.window.start_step,
                "end_step": self.window.end_step,
                "actual_end_step": summary["actual_end_step"],
                "complete": summary["complete"],
                "monotonic_ns_start": start_ns,
                "monotonic_ns_end": end_ns,
                "backend": self.window.backend,
                "backend_status": backend,
                "profiler_error": self.profiler_error,
                "overhead": dict(summary["overhead"]),
                **_unavailable_profile_metrics(),
            }
        )
        if events:
            record.update(events)
        if self.window.backend != TORCH_BACKEND:
            record["kernel_duration"] = {
                "status": "unsupported",
                "value": None,
                "unit": "us",
                "count": 0,
            }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def cuda_kernel_status(self) -> str:
        torch_probe = self.backend_probes.get(TORCH_BACKEND, {})
        return str(torch_probe.get("cuda_kernels", "unavailable"))

    def _can_write_summary(self) -> bool:
        if self._summary_written:
            return False
        if not self.window.enabled:
            return False
        return self.primary_rank

    def _published_summary_path(self) -> str | None:
        if not self.window.enabled or not self._summary_written:
            return None
        if not self.summary_path.exists():
            return None
        return str(self.summary_path)

    def _published_markers_path(self) -> str | None:
        if not self.emit_markers:
            return None
        if not self.writer.published:
            return None
        if not self.markers_path.exists():
            return None
        return str(self.markers_path)

    def _published_request_path(self) -> str | None:
        if not self._request_written or not self.request_path.exists():
            return None
        return str(self.request_path)

    def envelope_identity(self) -> EnvelopeIdentity:
        return EnvelopeIdentity(
            run_id=self.run_id,
            hostname=self.writer.hostname,
            gpu=self.writer.gpu,
            collector_version=self.writer.collector_version,
        )


def open_training_session(config: ExperimentConfig) -> TelemetrySession:
    telemetry = config.telemetry
    if not telemetry.environment_resolved:
        telemetry = overlay_telemetry_env(telemetry, os.environ)
    run_id = telemetry.run_id or config.run_name
    run_dir = Path("runs") / config.run_name
    telemetry_dir = run_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    probes = probe_profiler_backends()
    window = telemetry.profile_window
    window_id = profile_window_id(run_id, window) if window.enabled else None
    writer = MarkerWriter(
        path=telemetry_dir / "agoge-markers.jsonl",
        run_id=run_id,
        hostname=socket.gethostname(),
        gpu=gpu_identity(),
        collector_version=__version__,
        model_id=config.model_id,
        model_revision=config.revision or UNPINNED_REVISION,
        dataset=_dataset_ref(config),
    )
    primary = _is_primary_rank()
    session = TelemetrySession(
        run_id=run_id,
        run_dir=run_dir,
        telemetry_dir=telemetry_dir,
        markers_path=telemetry_dir / "agoge-markers.jsonl",
        request_path=telemetry_dir / "profile-window-request.json",
        summary_path=telemetry_dir / "profile-window.json",
        writer=writer,
        window=window,
        backend_probes=probes,
        window_id=window_id,
        emit_markers=telemetry.emit_markers and primary,
        primary_rank=primary,
    )
    if primary:
        _write_request(session)
        session.emit("run_start", phase="setup", global_step=0)
    return session


def _unavailable_profile_metrics() -> dict[str, Any]:
    return {
        "kernels": [],
        "kernel_duration": {
            "status": "unavailable",
            "value": None,
            "unit": "us",
            "count": 0,
        },
        "sync": {"status": "unavailable", "count": 0, "duration_us": None},
        "transfers": {"status": "unavailable", "h2d_us": None, "d2h_us": None},
    }


def attach_telemetry(trainer: Any, session: TelemetrySession) -> None:
    callback = TrainingCorrelationCallback(session)
    session.callback = callback
    trainer.add_callback(callback)


def _dataset_ref(config: ExperimentConfig) -> dict[str, Any]:
    digest = _dataset_digest(config.dataset_path)
    return {
        "id": Path(config.dataset_path).name,
        "split": "train",
        "config_digest": digest,
    }


def _dataset_digest(path: str) -> str | None:
    try:
        return "sha256:" + sha256_file(Path(path))
    except OSError:
        return None


def _write_request(session: TelemetrySession) -> None:
    stamp_utc, monotonic_ns = session.writer.stamp()
    record = envelope(
        record_kind=RECORD_PROFILE_REQUEST,
        identity=session.envelope_identity(),
        monotonic_ns=monotonic_ns,
    )
    record.update(
        {
            "timestamp_utc": stamp_utc,
            "enabled": session.window.enabled,
            "phase": session.window.phase,
            "start_step": session.window.start_step,
            "end_step": session.window.end_step,
            "profile_window_id": session.window_id,
            "backend": session.window.backend,
            "backends": session.backend_probes,
            "reason": None
            if session.window.enabled
            else "profiler is opt-in; ordinary Agoge runs do not capture CUDA windows",
        }
    )
    try:
        session.request_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        session._request_written = True
    except OSError as error:
        logger.warning(f"profile window request write failed: {error}")


def _is_primary_rank() -> bool:
    for key in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value is not None and value != "":
            return value == "0"
    return True
