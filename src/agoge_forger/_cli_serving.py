"""vLLM serving and chat-completions smoke commands."""

import os
from dataclasses import dataclass
from typing import Annotated, Any

import typer
import yaml

from ._cli_app import app
from .path_safety import resolve_existing_path
from .providers.chat_completions import ChatCompletionsConfig
from .serving.config import ServingConfig, load_serving_config
from .serving.serve import serve_vllm as _serve_vllm
from .serving.smoke import run_vllm_smoke

# Sourced from the model rather than repeated, so the CLI fallback cannot drift
# away from the field default it is standing in for.
_DEFAULT_SMOKE_BASE_URL: str = ChatCompletionsConfig.model_fields["base_url"].default


def _merge_serving_config(config_path: str | None, overrides: dict[str, Any]) -> ServingConfig:
    cfg = load_serving_config(config_path) if config_path else ServingConfig()
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if not cfg.model:
        raise typer.BadParameter("Model ID is required (--model or config.model)")
    return cfg


# pylint: disable=too-many-arguments,too-many-positional-arguments
@app.command()
def serve_vllm(
    config: str | None = typer.Option(None, "--config", help="Path to YAML serving config"),
    model: str | None = typer.Option(None, help="Model ID"),
    host: str | None = typer.Option(None, help="Host to bind"),
    port: int | None = typer.Option(None, help="Port"),
    max_model_len: int | None = typer.Option(None, help="Maximum model context length"),
    dtype: str | None = typer.Option(None, help="Model dtype"),
    gpu_memory_utilization: float | None = typer.Option(None, help="GPU memory utilization"),
    dry_run: bool | None = typer.Option(
        None,
        "--dry-run/--no-dry-run",
        help="Print command and exit; unset defers to the config file",
    ),
    extra_arg: Annotated[
        list[str] | None, typer.Option("--extra-arg", help="Extra vllm serve argument")
    ] = None,
):
    """Serve a model with vLLM."""
    overrides: dict[str, Any] = {
        "model": model,
        "host": host,
        "port": port,
        "max_model_len": max_model_len,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dry_run": dry_run,
    }
    if extra_arg:
        overrides["extra_args"] = extra_arg
    cfg = _merge_serving_config(config, overrides)
    raise typer.Exit(_serve_vllm(cfg))


# pylint: enable=too-many-arguments,too-many-positional-arguments


def _first_non_empty(value: str | None, *env_names: str) -> str | None:
    """Return ``value`` if provided, otherwise the first non-empty environment variable."""
    if value is not None:
        return value
    for name in env_names:
        env_value = os.environ.get(name)
        if env_value:
            return env_value
    return None


@dataclass(frozen=True)
class _SmokeEnvInputs:
    base_url: str | None
    model: str | None
    api_key: str | None
    prompt: str | None
    system: str | None
    stream: bool | None
    config_path: str | None


def _smoke_env_defaults(
    inputs: _SmokeEnvInputs,
) -> tuple[str | None, str | None, str | None, str | None, str | None, bool | None]:
    """Apply environment fallbacks and determine the effective streaming flag."""
    return (
        _first_non_empty(inputs.base_url, "AGOGE_SMOKE_BASE_URL"),
        _first_non_empty(inputs.model, "AGOGE_SMOKE_MODEL"),
        _first_non_empty(inputs.api_key, "OPENAI_API_KEY", "VLLM_API_KEY"),
        _first_non_empty(inputs.prompt, "AGOGE_SMOKE_PROMPT"),
        _first_non_empty(inputs.system, "AGOGE_SMOKE_SYSTEM"),
        _effective_smoke_stream(inputs.stream, inputs.config_path),
    )


def _effective_smoke_stream(stream: bool | None, config_path: str | None) -> bool | None:
    if stream is not None:
        return stream
    if config_path is None:
        return False
    return None


def _smoke_config_base(config_path: str | None) -> dict[str, Any]:
    """Start from the YAML config when one is given, else the model's defaults."""
    if config_path:
        path = resolve_existing_path(config_path, must_be_file=True)
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ChatCompletionsConfig().model_dump()


def _fill_smoke_defaults(data: dict[str, Any]) -> None:
    """Supply the values the endpoint needs but an operator may leave unset."""
    if not data.get("base_url"):
        data["base_url"] = _DEFAULT_SMOKE_BASE_URL
    if data.get("api_key") is None:
        data["api_key"] = ""


def _merge_smoke_chat_config(
    config_path: str | None, overrides: dict[str, Any]
) -> ChatCompletionsConfig:
    """Load a ChatCompletionsConfig from YAML and apply CLI overrides.

    Overrides are merged into a dict and then re-validated so that Pydantic
    field validators (e.g. stripping trailing slashes from ``base_url``) run.
    """
    data = _smoke_config_base(config_path)
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    _fill_smoke_defaults(data)
    if not data.get("model"):
        raise typer.BadParameter("Model is required (--model or config.model)")
    return ChatCompletionsConfig.model_validate(data)


# pylint: disable=too-many-arguments,too-many-positional-arguments
@app.command()
def smoke_vllm(
    config: str | None = typer.Option(
        None, "--config", help="Path to YAML chat-completions config"
    ),
    base_url: str | None = typer.Option(None, help="vLLM endpoint base URL"),
    model: str | None = typer.Option(None, help="Model name or path"),
    api_key: str | None = typer.Option(None, help="API key"),
    stream: bool | None = typer.Option(None, "--stream/--no-stream", help="Use streaming"),
    max_tokens: int | None = typer.Option(None, help="Maximum tokens in the response"),
    temperature: float | None = typer.Option(None, help="Sampling temperature"),
    timeout_s: float | None = typer.Option(None, help="HTTP timeout in seconds"),
    prompt: str | None = typer.Option(None, help="Single prompt"),
    system: str | None = typer.Option(None, help="System message"),
    prompt_set: str | None = typer.Option(None, "--prompt-set", help="YAML prompt set"),
    run_name: str | None = typer.Option(None, help="Run name for output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip the HTTP call"),
):
    """Run a vLLM/OpenAI-compatible chat-completion smoke test."""
    base_url, model, api_key, prompt, system, stream = _smoke_env_defaults(
        _SmokeEnvInputs(
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prompt,
            system=system,
            stream=stream,
            config_path=config,
        )
    )

    overrides: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout_s": timeout_s,
    }
    cfg = _merge_smoke_chat_config(config, overrides)

    _, results = run_vllm_smoke(
        cfg,
        run_name=run_name or "vllm_smoke",
        prompt_set=prompt_set,
        prompt=prompt,
        system=system,
        dry_run=dry_run,
    )
    errors = sum(1 for r in results if r.status == "error")
    raise typer.Exit(code=1 if errors else 0)


# pylint: enable=too-many-arguments,too-many-positional-arguments


if __name__ == "__main__":
    app()
