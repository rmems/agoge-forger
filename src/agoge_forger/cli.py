"""Agoge Forger Typer CLI entry point."""

import json
import os
from typing import Annotated, Any

import typer
import yaml

from .artifacts.safetensors_io import assert_no_unsafe_weight_bins, inspect_safetensors_file
from .backends.torch_backend import check_torch_env
from .config import load_config
from .datasets import dataset_stats as _dataset_stats
from .eval.smoke_eval import run_smoke_eval
from .export.merge_adapter import export_final_model as _export_final_model
from .export.merge_adapter import merge_adapter as _merge_adapter
from .logging import logger
from .models.inspect import inspect_model as _inspect_model
from .models.lora_targets import inspect_lora_targets as _inspect_lora_targets
from .models.metadata import get_model_config_metadata
from .path_safety import resolve_existing_path, resolve_output_directory
from .providers.chat_completions import ChatCompletionsConfig
from .serving.config import ServingConfig, load_serving_config
from .serving.serve import serve_vllm as _serve_vllm
from .serving.smoke import run_vllm_smoke
from .split_contract import (
    SPLIT_NAMES,
    CanonicalIdentityPolicy,
    SplitMaterializationSpec,
    SplitPolicy,
    materialize_split,
)
from .train.checkpoints import infer_base_model_from_adapter, is_adapter_artifact
from .train.lora import train_lora as _train_lora
from .train.qlora import train_qlora as _train_qlora

app = typer.Typer(help="Agoge Forger CLI")


@app.command()
def check_torch():
    """Check PyTorch/CUDA environment."""
    check_torch_env()


@app.command()
def inspect_model(
    model_id: str = typer.Option(..., help="Hugging Face model ID"),
    trust_remote_code: bool = typer.Option(False, help="Trust remote code from the model repo"),
):
    """Inspect model architecture (loads weights)."""
    _inspect_model(model_id, trust_remote_code)


@app.command()
def model_metadata(
    model_id: str = typer.Option(..., help="Hugging Face model ID"),
    trust_remote_code: bool = typer.Option(False, help="Trust remote code from the model repo"),
):
    """Inspect model metadata without loading weights."""
    meta = get_model_config_metadata(model_id, trust_remote_code)
    logger.info(json.dumps(meta, indent=2))


@app.command()
def inspect_lora_targets(
    model_id: str = typer.Option(..., help="Hugging Face model ID"),
    out: str = typer.Option(None, help="Output JSON path"),
    trust_remote_code: bool = typer.Option(False, help="Trust remote code from the model repo"),
):
    """Inspect model for potential LoRA targets."""
    _inspect_lora_targets(model_id, trust_remote_code, out)


@app.command()
def train_qlora(config: str = typer.Option(..., help="Path to YAML config")):
    """Run QLoRA training."""
    cfg = load_config(config)
    _train_qlora(cfg)


@app.command()
def train_lora(config: str = typer.Option(..., help="Path to YAML config")):
    """Run LoRA training."""
    cfg = load_config(config)
    _train_lora(cfg)


@app.command()
def smoke_eval(
    adapter_path: str = typer.Option(..., help="Path to PEFT adapter"),
    trust_remote_code: bool = typer.Option(False, help="Trust remote code from the model repo"),
    allow_unsafe_serialization: bool = typer.Option(
        False, help="Allow .bin weight files in the adapter"
    ),
):
    """Run a smoke evaluation on an adapter."""
    safe_adapter_path = resolve_existing_path(adapter_path, must_be_dir=True)
    # Enforce the safetensors-only policy on the adapter path the same
    # way resume/export do, so PeftModel.from_pretrained cannot be
    # tricked into deserializing a pickle-based adapter_model.bin.
    if not allow_unsafe_serialization:
        try:
            assert_no_unsafe_weight_bins(str(safe_adapter_path), recursive=True)
        except RuntimeError as e:
            logger.error(str(e))
            raise typer.Exit(code=1)
    try:
        base_model = infer_base_model_from_adapter(str(safe_adapter_path))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
        logger.error("Could not infer base model: %s", e)
        raise typer.Exit(code=1)

    run_smoke_eval(base_model, str(safe_adapter_path), trust_remote_code=trust_remote_code)


@app.command()
def merge_adapter(
    base_model: str = typer.Option(..., help="Base model ID"),
    adapter_path: str = typer.Option(..., help="Path to PEFT adapter"),
    out_dir: str = typer.Option(..., help="Output directory"),
    trust_remote_code: bool = typer.Option(False, help="Trust remote code from the model repo"),
    allow_unsafe_serialization: bool = typer.Option(
        False, help="Allow .bin weight files in the adapter"
    ),
):
    """Merge PEFT adapter into base model."""
    safe_adapter_path = str(resolve_existing_path(adapter_path, must_be_dir=True))
    # Enforce the safetensors-only policy before PeftModel.from_pretrained,
    # mirroring smoke_eval and resolve_export_source so every adapter entry
    # point applies the same safety check.
    if not allow_unsafe_serialization:
        if not is_adapter_artifact(safe_adapter_path):
            raise typer.BadParameter(
                f"Adapter path is not a valid safetensors adapter artifact: {safe_adapter_path}"
            )
        try:
            assert_no_unsafe_weight_bins(safe_adapter_path, recursive=True)
        except RuntimeError as e:
            logger.error(str(e))
            raise typer.Exit(code=1)
    safe_out_dir = str(resolve_output_directory(out_dir))
    _merge_adapter(
        base_model,
        safe_adapter_path,
        safe_out_dir,
        trust_remote_code=trust_remote_code,
        allow_unsafe=allow_unsafe_serialization,
    )


@app.command()
def export_final_model(
    out_dir: str = typer.Option(..., help="Output directory for the merged model"),
    run_dir: str | None = typer.Option(
        None, help="Run directory containing checkpoints or a final adapter"
    ),
    adapter_path: str | None = typer.Option(
        None, help="Specific adapter or checkpoint directory to export"
    ),
    base_model: str | None = typer.Option(None, help="Base model ID override"),
    save_safetensors: bool = typer.Option(
        True,
        help=(
            "Require safetensors export (Transformers 5 always writes safetensors for "
            "merged models; False is rejected)"
        ),
    ),
    allow_unsafe_serialization: bool = typer.Option(
        False, help="Allow unsafe .bin input adapters / skip post-save .bin assert"
    ),
    max_shard_size: str = typer.Option("4GB", help="Maximum shard size for merged weights"),
    trust_remote_code: bool = typer.Option(False, help="Trust remote code from the model repo"),
):
    """Export one final merged model from the latest valid checkpoint or adapter."""
    safe_out_dir = str(resolve_output_directory(out_dir))
    safe_run_dir = str(resolve_existing_path(run_dir, must_be_dir=True)) if run_dir else None
    safe_adapter_path = (
        str(resolve_existing_path(adapter_path, must_be_dir=True)) if adapter_path else None
    )
    _export_final_model(
        out_dir=safe_out_dir,
        run_dir=safe_run_dir,
        adapter_path=safe_adapter_path,
        base_model_id=base_model,
        save_safetensors=save_safetensors,
        allow_unsafe=allow_unsafe_serialization,
        max_shard_size=max_shard_size,
        trust_remote_code=trust_remote_code,
    )


@app.command()
def inspect_safetensors(path: str = typer.Option(..., help="Path to safetensors file")):
    """Inspect a safetensors file."""
    safe_path = str(resolve_existing_path(path, must_be_file=True))
    info = inspect_safetensors_file(safe_path)
    logger.info(json.dumps(info, indent=2))


@app.command()
def dataset_stats(
    path: str = typer.Option(..., help="Path to JSONL dataset"),
    model_id: str = typer.Option(..., help="Model ID for tokenizer"),
    trust_remote_code: bool = typer.Option(False, help="Trust remote code from the model repo"),
):
    """Get dataset token statistics."""
    safe_path = str(resolve_existing_path(path, must_be_file=True))
    _dataset_stats(safe_path, model_id, trust_remote_code=trust_remote_code)


@app.command("freeze-split")
def freeze_split(
    source: str = typer.Option(..., help="Versioned curated source JSONL"),
    source_path: str = typer.Option(
        ...,
        help="Canonical repository-relative path of the source at --source-revision",
    ),
    output_dir: str = typer.Option(..., help="New immutable snapshot directory"),
    source_repository: str = typer.Option(...),
    source_revision: str = typer.Option(...),
    dataset_version: str = typer.Option(...),
    seed: int = typer.Option(...),
    salt: str = typer.Option(...),
    train_weight: int = typer.Option(80),
    validation_weight: int = typer.Option(10),
    held_out_weight: int = typer.Option(10),
    canonical_id_field: str = typer.Option("canonical_id"),
    lineage_id_field: str = typer.Option("lineage_id"),
    group_id_field: str = typer.Option("group_id"),
):
    """Materialize an immutable three-way SFT split from a pinned local source."""
    safe_source = resolve_existing_path(source, must_be_file=True)
    safe_output = resolve_output_directory(output_dir)
    spec = SplitMaterializationSpec(
        source_repository=source_repository,
        source_revision=source_revision,
        dataset_version=dataset_version,
        source_path=source_path,
        split_policy=SplitPolicy(
            seed=seed,
            salt=salt,
            weights={
                "train": train_weight,
                "validation": validation_weight,
                "held_out": held_out_weight,
            },
        ),
        canonical_identity=CanonicalIdentityPolicy(
            canonical_id_field=canonical_id_field,
            lineage_id_field=lineage_id_field,
            group_id_field=group_id_field,
            content_hash_policy="normalized-training-payload-v1",
        ),
    )
    manifest = materialize_split(safe_source, safe_output, spec)
    logger.info(
        "source: %s@%s:%s",
        manifest.source.repository,
        manifest.source.revision,
        manifest.source.path,
    )
    logger.info("wrote %s/split_manifest.json", safe_output)
    logger.info("wrote %s/split_report.md", safe_output)
    counts = ", ".join(f"{name}={manifest.splits[name].record_count}" for name in SPLIT_NAMES)
    logger.info("counts: %s", counts)


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
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command and exit"),
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


def _smoke_env_defaults(
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    prompt: str | None,
    system: str | None,
    stream: bool | None,
    config_path: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None, bool | None]:
    """Apply environment fallbacks and determine the effective streaming flag."""
    return (
        _first_non_empty(base_url, "AGOGE_SMOKE_BASE_URL"),
        _first_non_empty(model, "AGOGE_SMOKE_MODEL"),
        _first_non_empty(api_key, "OPENAI_API_KEY", "VLLM_API_KEY"),
        _first_non_empty(prompt, "AGOGE_SMOKE_PROMPT"),
        _first_non_empty(system, "AGOGE_SMOKE_SYSTEM"),
        stream if stream is not None else (False if config_path is None else None),
    )


def _merge_smoke_chat_config(
    config_path: str | None, overrides: dict[str, Any]
) -> ChatCompletionsConfig:
    """Load a ChatCompletionsConfig from YAML and apply CLI overrides.

    Overrides are merged into a dict and then re-validated so that Pydantic
    field validators (e.g. stripping trailing slashes from ``base_url``) run.
    """
    if config_path:
        path = resolve_existing_path(config_path, must_be_file=True)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = ChatCompletionsConfig().model_dump()

    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    if not data.get("base_url"):
        data["base_url"] = "http://localhost:8000/v1"
    if not data.get("model"):
        raise typer.BadParameter("Model is required (--model or config.model)")
    if data.get("api_key") is None:
        data["api_key"] = ""

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
        base_url, model, api_key, prompt, system, stream, config
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
