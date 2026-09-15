"""HuggingFace callback: phase/step markers and a bounded torch.profiler window."""

from __future__ import annotations

import time
from typing import Any

from torch.profiler import profile
from transformers import TrainerCallback

from ..logging import logger
from .backends import requested_backend_status, torch_profiler_activities
from .schema import TORCH_BACKEND, loss_measurement
from .summarize import overhead_from_step_times, summarize_profiler
from .window import upcoming_step_starts_window, window_covers_step


class TrainingCorrelationCallback(TrainerCallback):
    def __init__(self, session: Any) -> None:
        self._session = session
        self._profiler: Any = None
        self._step_t0: float | None = None
        self._profiled_s: list[float] = []
        self._unprofiled_s: list[float] = []
        self._window_start_ns: int | None = None
        self._window_end_ns: int | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._session.emit("train_start", phase="train", global_step=int(state.global_step))

    def on_step_begin(self, args, state, control, **kwargs):
        upcoming = int(state.global_step) + 1
        if upcoming_step_starts_window(self._session.window, upcoming):
            self._start_profiler()
        self._step_t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        self._record_step_duration(step)
        if self._profiler is not None:
            self._profiler.step()
        if step == self._session.window.end_step and self._profiler is not None:
            self._stop_profiler()

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = int(state.global_step)
        extras = {
            "loss": loss_measurement(logs),
            "tokens_accepted": getattr(state, "num_input_tokens_seen", None),
        }
        if window_covers_step(self._session.window, step):
            extras["profile_window_ref"] = self._session.window_id
        phase = "eval" if logs and any(key.startswith("eval_") for key in logs) else "train"
        event = "eval_log" if phase == "eval" else "step_end"
        self._session.emit(event, phase=phase, global_step=step, extras=extras)

    def on_save(self, args, state, control, **kwargs):
        self._session.emit(
            "checkpoint_complete", phase="checkpoint", global_step=int(state.global_step)
        )

    def on_train_end(self, args, state, control, **kwargs):
        self._stop_profiler()
        self._session.emit("train_end", phase="train", global_step=int(state.global_step))
        self._session.close()

    def _record_step_duration(self, step: int) -> None:
        if self._step_t0 is None:
            return
        elapsed = time.perf_counter() - self._step_t0
        self._step_t0 = None
        if window_covers_step(self._session.window, step):
            self._profiled_s.append(elapsed)
        else:
            self._unprofiled_s.append(elapsed)

    def _start_profiler(self) -> None:
        if self._profiler is not None:
            return
        probes = self._session.backend_probes
        status = requested_backend_status(self._session.window.backend, probes)
        if self._session.window.backend != TORCH_BACKEND or status.get("status") != "ok":
            logger.info(
                "profile window requested backend "
                f"{self._session.window.backend} status={status.get('status')}; not starting torch.profiler"
            )
            self._window_start_ns = self._session.writer.next_monotonic()
            return
        try:
            self._profiler = profile(
                activities=torch_profiler_activities(),
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
                acc_events=True,
            )
            self._profiler.start()
            self._window_start_ns = self._session.writer.next_monotonic()
            self._session.emit(
                "profile_window_start",
                phase=self._session.window.phase,
                global_step=self._session.window.start_step,
                extras={"profile_window_ref": self._session.window_id},
            )
        except (RuntimeError, OSError, ValueError, AttributeError) as error:
            logger.warning(f"torch.profiler failed to start: {error}")
            self._profiler = None
            self._session.profiler_error = str(error)

    def _stop_profiler(self) -> None:
        if self._window_end_ns is not None:
            return
        self._window_end_ns = self._session.writer.next_monotonic()
        summary_events = None
        if self._profiler is not None:
            try:
                self._profiler.stop()
                summary_events = summarize_profiler(
                    self._profiler,
                    cuda_kernel_status=self._session.cuda_kernel_status(),
                )
            except (RuntimeError, OSError, ValueError, AttributeError) as error:
                logger.warning(f"torch.profiler stop/summarize failed: {error}")
                self._session.profiler_error = str(error)
            self._profiler = None
        if self._session.window.enabled:
            self._session.write_profile_summary(
                events=summary_events,
                overhead=overhead_from_step_times(self._profiled_s, self._unprofiled_s),
                start_ns=self._window_start_ns,
                end_ns=self._window_end_ns,
            )
            self._session.emit(
                "profile_window_end",
                phase=self._session.window.phase,
                global_step=self._session.window.end_step,
                extras={"profile_window_ref": self._session.window_id},
            )
