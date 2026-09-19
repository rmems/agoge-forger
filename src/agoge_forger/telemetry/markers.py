"""Append-only ``agoge_marker`` JSONL. Failures never abort training."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging import logger
from .schema import RECORD_MARKER, UNPINNED_REVISION, EnvelopeIdentity, envelope, utc_z


@dataclass
class MarkerWriter:
    path: Path
    run_id: str
    hostname: str
    gpu: dict[str, Any]
    collector_version: str
    model_id: str
    model_revision: str
    dataset: dict[str, Any]
    _last_monotonic_ns: int = field(default=-1, init=False)
    _clock: Any = field(default=None, repr=False)
    published: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self._clock is None:
            self._clock = time.monotonic_ns

    def update_gpu(self, gpu: Mapping[str, Any]) -> None:
        self.gpu = dict(gpu)

    def emit(
        self,
        event: str,
        *,
        phase: str,
        global_step: int,
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            self._write(event, phase, global_step, extras or {})
            self.published = True
        except OSError as error:
            logger.warning(f"telemetry marker write failed ({event}): {error}")

    def _write(
        self,
        event: str,
        phase: str,
        global_step: int,
        extras: Mapping[str, Any],
    ) -> None:
        monotonic_ns = self._next_monotonic()
        record = envelope(
            record_kind=RECORD_MARKER,
            identity=EnvelopeIdentity(
                run_id=self.run_id,
                hostname=self.hostname,
                gpu=self.gpu,
                collector_version=self.collector_version,
            ),
            monotonic_ns=monotonic_ns,
        )
        record.update(
            {
                "event": event,
                "model_id": self.model_id,
                "model_revision": self.model_revision or UNPINNED_REVISION,
                "dataset": dict(self.dataset),
                "phase": phase,
                "global_step": global_step,
                "microstep": extras.get("microstep"),
                "tokens_accepted": extras.get("tokens_accepted"),
                "examples_accepted": extras.get("examples_accepted"),
            }
        )
        if "loss" in extras and extras["loss"] is not None:
            record["loss"] = extras["loss"]
        if extras.get("profile_window_ref"):
            record["profile_window_ref"] = extras["profile_window_ref"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()

    def next_monotonic(self) -> int:
        return self._next_monotonic()

    def _next_monotonic(self) -> int:
        now_ns = int(self._clock())
        if now_ns <= self._last_monotonic_ns:
            now_ns = self._last_monotonic_ns + 1
        self._last_monotonic_ns = now_ns
        return now_ns

    def stamp(self) -> tuple[str, int]:
        return utc_z(), self._next_monotonic()
