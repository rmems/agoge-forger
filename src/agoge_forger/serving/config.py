from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..path_safety import resolve_existing_path


class Frontend(str, Enum):
    python = "python"
    rust = "rust"


class ServingConfig(BaseModel):
    model: str = ""
    frontend: Frontend = Frontend.python
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int | None = None
    dtype: str | None = None
    gpu_memory_utilization: float | None = None
    extra_args: list[str] = Field(default_factory=list)
    dry_run: bool = False


class PromptSet(BaseModel):
    system: str = ""
    prompts: list[str] = Field(default_factory=list)


class BenchmarkConfig(BaseModel):
    model: str = ""
    frontend: Frontend | None = None  # None means benchmark both frontends.
    host: str = "0.0.0.0"
    port: int = 8000
    prompt_set: str = ""
    out_dir: str = ""
    run_name: str = "vllm_bench"
    stream: bool = False
    max_tokens: int = 512
    temperature: float = 0.7
    concurrency: int = 1
    dry_run: bool = False
    max_model_len: int | None = None
    dtype: str | None = None
    gpu_memory_utilization: float | None = None
    extra_args: list[str] = Field(default_factory=list)


def load_serving_config(path: str) -> ServingConfig:
    config_path = resolve_existing_path(path, must_be_file=True)
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Serving config must be a YAML mapping: {path}")
    return ServingConfig.model_validate(data)


def load_prompt_set(path: str) -> PromptSet:
    prompt_path = resolve_existing_path(path, must_be_file=True)
    raw = yaml.safe_load(prompt_path.read_text())
    if isinstance(raw, list):
        return PromptSet(prompts=raw)
    if isinstance(raw, dict):
        prompts = raw.get("prompts", [])
        if not isinstance(prompts, list):
            raise TypeError(f"Prompt set 'prompts' must be a list: {path}")
        return PromptSet(system=raw.get("system", ""), prompts=prompts)
    raise TypeError(f"Prompt set must be a list of strings or a mapping: {path}")


def load_benchmark_config(path: str) -> BenchmarkConfig:
    config_path = resolve_existing_path(path, must_be_file=True)
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Benchmark config must be a YAML mapping: {path}")
    cfg = BenchmarkConfig.model_validate(data)
    if cfg.prompt_set and not Path(cfg.prompt_set).is_absolute():
        resolved = (config_path.parent / cfg.prompt_set).resolve()
        cfg.prompt_set = str(resolved)
    return cfg


def to_serving_config(benchmark_cfg: BenchmarkConfig, frontend: Frontend) -> ServingConfig:
    """Convert a benchmark config into a serving config for one frontend."""
    return ServingConfig(
        model=benchmark_cfg.model,
        frontend=frontend,
        host=benchmark_cfg.host,
        port=benchmark_cfg.port,
        max_model_len=benchmark_cfg.max_model_len,
        dtype=benchmark_cfg.dtype,
        gpu_memory_utilization=benchmark_cfg.gpu_memory_utilization,
        extra_args=benchmark_cfg.extra_args,
        dry_run=benchmark_cfg.dry_run,
    )
