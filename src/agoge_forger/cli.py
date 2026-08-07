import json
from typing import Annotated

import typer

from .artifacts.safetensors_io import assert_no_unsafe_weight_bins, inspect_safetensors_file
from .backends.jax_backend import check_jax_env
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
from .serving.benchmark import benchmark_vllm_frontends
from .serving.config import (
    BenchmarkConfig,
    Frontend,
    ServingConfig,
    load_benchmark_config,
    load_serving_config,
)
from .serving.serve import serve_vllm as _serve_vllm
from .train.checkpoints import infer_base_model_from_adapter, is_adapter_artifact
from .train.lora import train_lora as _train_lora
from .train.qlora import train_qlora as _train_qlora

app = typer.Typer(help="Agoge Forger CLI")


@app.command()
def check_env():
    """Run all environment checks."""
    check_torch_env()
    check_jax_env()


@app.command()
def check_torch():
    """Check PyTorch/CUDA environment."""
    check_torch_env()


@app.command()
def check_jax():
    """Check JAX environment."""
    check_jax_env()


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
        logger.error(f"Could not infer base model: {e}")
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
    save_safetensors: bool = typer.Option(True, help="Save using safetensors"),
    allow_unsafe_serialization: bool = typer.Option(False, help="Allow .bin weight files"),
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


def _merge_serving_config(
    config_path: str | None,
    model: str | None,
    frontend: Frontend | None,
    host: str | None,
    port: int | None,
    max_model_len: int | None,
    dtype: str | None,
    gpu_memory_utilization: float | None,
    dry_run: bool,
    extra_args: list[str] | None,
) -> ServingConfig:
    cfg = load_serving_config(config_path) if config_path else ServingConfig()
    if model is not None:
        cfg.model = model
    if frontend is not None:
        cfg.frontend = frontend
    if host is not None:
        cfg.host = host
    if port is not None:
        cfg.port = port
    if max_model_len is not None:
        cfg.max_model_len = max_model_len
    if dtype is not None:
        cfg.dtype = dtype
    if gpu_memory_utilization is not None:
        cfg.gpu_memory_utilization = gpu_memory_utilization
    if dry_run:
        cfg.dry_run = True
    if extra_args:
        cfg.extra_args = extra_args
    if not cfg.model:
        raise typer.BadParameter("Model ID is required (--model or config.model)")
    return cfg


@app.command()
def serve_vllm(
    config: str | None = typer.Option(None, "--config", help="Path to YAML serving config"),
    model: str | None = typer.Option(None, help="Model ID"),
    frontend: Annotated[Frontend | None, typer.Option(help="python or rust")] = None,
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
    """Serve a model with vLLM, optionally using the Rust frontend."""
    cfg = _merge_serving_config(
        config,
        model,
        frontend,
        host,
        port,
        max_model_len,
        dtype,
        gpu_memory_utilization,
        dry_run,
        extra_arg,
    )
    raise typer.Exit(_serve_vllm(cfg))


def _merge_benchmark_config(
    config_path: str | None,
    model: str | None,
    frontend: Frontend | None,
    prompt_set: str | None,
    out_dir: str | None,
    stream: bool | None,
    max_tokens: int | None,
    temperature: float | None,
    concurrency: int | None,
    dry_run: bool,
) -> BenchmarkConfig:
    if config_path:
        cfg = load_benchmark_config(config_path)
        if cfg.prompt_set:
            cfg.prompt_set = str(resolve_existing_path(cfg.prompt_set, must_be_file=True))
    else:
        cfg = BenchmarkConfig()
    if model is not None:
        cfg.model = model
    if frontend is not None:
        cfg.frontend = frontend
    if prompt_set is not None:
        cfg.prompt_set = str(resolve_existing_path(prompt_set, must_be_file=True))
    if out_dir is not None:
        cfg.out_dir = out_dir
    if stream is not None:
        cfg.stream = stream
    if max_tokens is not None:
        cfg.max_tokens = max_tokens
    if temperature is not None:
        cfg.temperature = temperature
    if concurrency is not None:
        cfg.concurrency = concurrency
    if dry_run:
        cfg.dry_run = True
    if not cfg.model:
        raise typer.BadParameter("Model ID is required (--model or config.model)")
    if not cfg.prompt_set:
        raise typer.BadParameter("Prompt set is required (--prompt-set or config.prompt_set)")
    return cfg


@app.command()
def bench_vllm_frontend(
    config: str | None = typer.Option(None, "--config", help="Path to YAML benchmark config"),
    model: str | None = typer.Option(None, help="Model ID"),
    frontend: Annotated[
        Frontend | None, typer.Option(help="python, rust, or omit for both")
    ] = None,
    prompt_set: str | None = typer.Option(None, "--prompt-set", help="Path to prompt set YAML"),
    out_dir: str | None = typer.Option(None, help="Output directory"),
    stream: bool | None = typer.Option(None, "--stream/--no-stream", help="Use streaming requests"),
    max_tokens: int | None = typer.Option(None, help="Max tokens per request"),
    temperature: float | None = typer.Option(None, help="Sampling temperature"),
    concurrency: int | None = typer.Option(None, help="Request concurrency"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate benchmark"),
):
    """Benchmark vLLM Python vs Rust frontends."""
    cfg = _merge_benchmark_config(
        config,
        model,
        frontend,
        prompt_set,
        out_dir,
        stream,
        max_tokens,
        temperature,
        concurrency,
        dry_run,
    )
    raise typer.Exit(benchmark_vllm_frontends(cfg))


if __name__ == "__main__":
    app()
