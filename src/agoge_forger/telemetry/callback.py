"""HuggingFace callback: phase/step markers and a bounded torch.profiler window."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.profiler import profile
from transformers import TrainerCallback

from ..logging import logger
from .backends import requested_backend_status, torch_profiler_activities
from .schema import TORCH_BACKEND, TRAIN_PHASE, loss_measurement
from .summarize import overhead_from_step_times, summarize_profiler
from .window import upcoming_step_starts_window, window_covers_step


class TrainingCorrelationCallback(TrainerCallback):
    def __init__(self, session: Any) -> None:
        self._session = session
        self._profiler: Any = None
        self._profiler_active = False
        self._window_started = False
        self._step_t0: float | None = None
        self._profiled_s: list[float] = []
        self._unprofiled_s: list[float] = []
        self._window_start_ns: int | None = None
        self._window_end_ns: int | None = None
        self._summary_events: dict[str, Any] | None = None
        self._summary_actual_step: int | None = None
        self._summary_complete = False
        self._last_step = 0
        self._finalized = False

    def on_train_begin(self, args, state, control, **kwargs):
        self._session.emit("train_start", phase=TRAIN_PHASE, global_step=int(state.global_step))

    def on_step_begin(self, args, state, control, **kwargs):
        upcoming = int(state.global_step) + 1
        self._step_t0 = time.perf_counter()
        self._maybe_start_window(upcoming)
        self._emit_step_begin(upcoming)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        self._last_step = step
        was_profiled = self._profiler_active and window_covers_step(self._session.window, step)
        self._advance_profiler()
        if not self._maybe_stop_window(step, profiled=was_profiled):
            self._synchronize_timed_step()
            self._record_step_duration(profiled=was_profiled)

    def on_log(self, args, state, control, **kwargs):
        self._emit_log(args, state, kwargs.get("logs"))

    def on_save(self, args, state, control, **kwargs):
        self._session.emit(
            "checkpoint_complete", phase="checkpoint", global_step=int(state.global_step)
        )

    def on_train_end(self, args, state, control, **kwargs):
        actual_step = int(state.global_step)
        self._last_step = actual_step
        self.finalize()
        self._session.emit("train_end", phase=TRAIN_PHASE, global_step=actual_step)
        self._session.close()

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        actual_step = self._completed_window_step()
        self._stop_profiler(
            actual_step=actual_step,
            complete=(actual_step is not None and actual_step >= self._session.window.end_step),
        )
        self._write_final_summary()

    @property
    def last_completed_step(self) -> int:
        return self._last_step

    def _completed_window_step(self) -> int | None:
        if self._last_step < self._session.window.start_step:
            return None
        return self._last_step

    def _emit_step_begin(self, upcoming: int) -> None:
        extras = {}
        if self._window_started and window_covers_step(self._session.window, upcoming):
            extras["profile_window_ref"] = self._session.window_id
        self._session.emit("step_begin", phase=TRAIN_PHASE, global_step=upcoming, extras=extras)

    def _emit_log(self, args: Any, state: Any, logs: Any) -> None:
        step = int(state.global_step)
        if _is_evaluation_log(logs):
            phase = "eval"
        elif _is_training_step_log(logs):
            phase = TRAIN_PHASE
        else:
            return
        tokens_accepted = None
        if getattr(args, "include_num_input_tokens_seen", False):
            tokens_accepted = getattr(state, "num_input_tokens_seen", None)
        extras = {
            "loss": loss_measurement(logs),
            "tokens_accepted": tokens_accepted,
        }
        if self._window_started and window_covers_step(self._session.window, step):
            extras["profile_window_ref"] = self._session.window_id
        event = "eval_log" if phase == "eval" else "step_end"
        self._session.emit(event, phase=phase, global_step=step, extras=extras)

    def _record_step_duration(self, *, profiled: bool) -> None:
        if self._step_t0 is None:
            return
        elapsed = time.perf_counter() - self._step_t0
        self._step_t0 = None
        if profiled:
            self._profiled_s.append(elapsed)
            return
        self._unprofiled_s.append(elapsed)

    def _maybe_start_window(self, upcoming: int) -> None:
        window = self._session.window
        if window.phase != TRAIN_PHASE:
            return
        if upcoming_step_starts_window(window, upcoming):
            self._start_profiler()

    def _maybe_stop_window(self, step: int, *, profiled: bool) -> bool:
        if self._window_started and step == self._session.window.end_step:
            self._stop_profiler(
                actual_step=step,
                complete=True,
                record_profiled=profiled,
            )
            return True
        return False

    def _advance_profiler(self) -> None:
        if self._profiler is None:
            return
        try:
            self._profiler.step()
        except (RuntimeError, OSError, ValueError, AttributeError) as error:
            logger.warning(f"torch.profiler advancement failed: {error}")
            self._record_profiler_error(error)
            self._best_effort_stop(self._profiler)
            self._profiler = None
            self._profiler_active = False

    def _start_profiler(self) -> None:
        if self._window_started:
            return
        self._window_started = True
        self._window_start_ns = self._session.writer.next_monotonic()
        self._session.emit(
            "profile_window_start",
            phase=self._session.window.phase,
            global_step=self._session.window.start_step,
            extras={"profile_window_ref": self._session.window_id},
        )
        if not self._torch_backend_ready():
            return
        self._launch_torch_profiler()

    def _torch_backend_ready(self) -> bool:
        if not self._session.primary_rank:
            logger.info("not starting torch.profiler on a non-primary rank")
            return False
        probes = self._session.backend_probes
        status = requested_backend_status(self._session.window.backend, probes)
        if self._session.window.backend == TORCH_BACKEND and status.get("status") == "ok":
            return True
        logger.info(
            "profile window requested backend "
            f"{self._session.window.backend} status={status.get('status')}; not starting torch.profiler"
        )
        return False

    def _launch_torch_profiler(self) -> None:
        try:
            self._profiler = profile(
                activities=torch_profiler_activities(),
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
                acc_events=True,
            )
            self._profiler.start()
            self._profiler_active = True
        except (RuntimeError, OSError, ValueError, AttributeError) as error:
            logger.warning(f"torch.profiler failed to start: {error}")
            self._best_effort_stop(self._profiler)
            self._profiler = None
            self._profiler_active = False
            self._session.profiler_error = str(error)

    def _stop_profiler(
        self,
        *,
        actual_step: int | None,
        complete: bool,
        record_profiled: bool | None = None,
    ) -> None:
        if self._window_end_ns is not None or not self._window_started:
            return
        if self._profiler_active:
            self._synchronize_profiled_device()
        self._window_end_ns = self._session.writer.next_monotonic()
        self._summary_events = self._collect_summary()
        self._summary_actual_step = actual_step
        self._summary_complete = complete
        if record_profiled is not None:
            self._record_step_duration(profiled=record_profiled)
        if self._session.window.enabled and actual_step is not None:
            self._session.emit(
                "profile_window_end",
                phase=self._session.window.phase,
                global_step=actual_step,
                extras={"profile_window_ref": self._session.window_id},
            )

    def _write_final_summary(self) -> None:
        if not self._session.window.enabled or self._window_end_ns is None:
            return
        self._session.write_profile_summary(
            summary={
                "events": self._summary_events,
                "overhead": overhead_from_step_times(self._profiled_s, self._unprofiled_s),
                "start_ns": self._window_start_ns,
                "end_ns": self._window_end_ns,
                "actual_end_step": self._summary_actual_step,
                "complete": self._summary_complete,
            },
        )

    def _collect_summary(self) -> dict[str, Any] | None:
        if self._profiler is None:
            return None
        try:
            self._profiler.stop()
            return summarize_profiler(
                self._profiler,
                cuda_kernel_status=self._session.cuda_kernel_status(),
            )
        except (RuntimeError, OSError, ValueError, AttributeError) as error:
            logger.warning(f"torch.profiler stop/summarize failed: {error}")
            self._session.profiler_error = str(error)
            return None
        finally:
            self._profiler = None
            self._profiler_active = False

    def _synchronize_profiled_device(self) -> None:
        try:
            _synchronize_profiled_device()
        except (RuntimeError, OSError, ValueError, AttributeError) as error:
            logger.warning(f"torch.profiler synchronization failed: {error}")
            self._record_profiler_error(error)

    def _synchronize_timed_step(self) -> None:
        window = self._session.window
        if window.enabled and window.backend == TORCH_BACKEND:
            self._synchronize_profiled_device()

    def _record_profiler_error(self, error: Exception) -> None:
        self._session.profiler_error = str(error)

    @staticmethod
    def _best_effort_stop(profiler: Any) -> None:
        try:
            profiler.stop()
        except (RuntimeError, OSError, ValueError, AttributeError) as error:
            logger.warning(f"torch.profiler cleanup after advancement failure failed: {error}")


def _synchronize_profiled_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _is_evaluation_log(logs: Any) -> bool:
    return bool(logs and any(key.startswith("eval_") for key in logs))


def _is_training_step_log(logs: Any) -> bool:
    return bool(logs and "loss" in logs)
