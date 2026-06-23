import json

import typer

from .artifacts.safetensors_io import inspect_safetensors_file
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
from .train.checkpoints import infer_base_model_from_adapter
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
):
    """Run a smoke evaluation on an adapter."""
    safe_adapter_path = resolve_existing_path(adapter_path, must_be_dir=True)
    try:
        base_model = infer_base_model_from_adapter(str(safe_adapter_path))
    except Exception as e:
        logger.error(f"Could not infer base model: {e}")
        raise typer.Exit(code=1)

    run_smoke_eval(base_model, str(safe_adapter_path), trust_remote_code=trust_remote_code)

@app.command()
def merge_adapter(
    base_model: str = typer.Option(..., help="Base model ID"),
    adapter_path: str = typer.Option(..., help="Path to PEFT adapter"),
    out_dir: str = typer.Option(..., help="Output directory"),
    trust_remote_code: bool = typer.Option(False, help="Trust remote code from the model repo"),
):
    """Merge PEFT adapter into base model."""
    safe_adapter_path = str(resolve_existing_path(adapter_path, must_be_dir=True))
    safe_out_dir = str(resolve_output_directory(out_dir))
    _merge_adapter(base_model, safe_adapter_path, safe_out_dir, trust_remote_code=trust_remote_code)

@app.command()
def export_final_model(
    out_dir: str = typer.Option(..., help="Output directory for the merged model"),
    run_dir: str | None = typer.Option(None, help="Run directory containing checkpoints or a final adapter"),
    adapter_path: str | None = typer.Option(None, help="Specific adapter or checkpoint directory to export"),
    base_model: str = typer.Option(None, help="Base model ID override"),
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

if __name__ == "__main__":
    app()
