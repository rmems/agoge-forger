"""Pydantic configuration models for vLLM serving."""

from __future__ import annotations

import yaml
from pydantic import BaseModel, Field

from ..path_safety import resolve_existing_path


class ServingConfig(BaseModel):
    """Configuration for `agoge serve-vllm`."""

    model: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    max_model_len: int | None = None
    dtype: str | None = None
    gpu_memory_utilization: float | None = None
    extra_args: list[str] = Field(default_factory=list)
    dry_run: bool = False


class PromptSet(BaseModel):
    """A vLLM smoke-test prompt set and optional system message."""

    system: str = ""
    prompts: list[str] = Field(default_factory=list)


def load_serving_config(path: str) -> ServingConfig:
    """Load a YAML serving config."""
    config_path = resolve_existing_path(path, must_be_file=True)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Serving config must be a YAML mapping: {path}")
    return ServingConfig.model_validate(data)


def load_prompt_set(path: str) -> PromptSet:
    """Load a YAML prompt set for vLLM compatibility smoke testing."""
    prompt_path = resolve_existing_path(path, must_be_file=True)
    raw = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return PromptSet(prompts=raw)
    if isinstance(raw, dict):
        prompts = raw.get("prompts", [])
        if not isinstance(prompts, list):
            raise TypeError(f"Prompt set 'prompts' must be a list: {path}")
        return PromptSet(system=raw.get("system", ""), prompts=prompts)
    raise TypeError(f"Prompt set must be a list of strings or a mapping: {path}")
