"""Opt-in bounded CUDA/torch.profiler window. Off for ordinary training."""

from pydantic import BaseModel, model_validator


class ProfileWindowConfig(BaseModel):
    enabled: bool = False
    phase: str = "train"
    start_step: int = 1
    end_step: int = 2
    backend: str = "torch"

    @model_validator(mode="after")
    def range_ok(self) -> "ProfileWindowConfig":
        _require_window_steps(self.start_step, self.end_step)
        self.phase = _require_train_phase(self.phase)
        return self


def _require_window_steps(start_step: int, end_step: int) -> None:
    if not _positive(start_step) or not _positive(end_step):
        raise ValueError("profile window steps must be >= 1")
    if end_step < start_step:
        raise ValueError("profile window end_step must be >= start_step")


def _require_train_phase(phase: str) -> str:
    stripped = phase.strip()
    if not stripped:
        raise ValueError("profile window phase must be non-empty")
    if stripped != "train":
        raise ValueError(
            "profile window phase must be 'train'; the training callback cannot honor other phases"
        )
    return stripped


def _positive(step: int) -> bool:
    return step >= 1
