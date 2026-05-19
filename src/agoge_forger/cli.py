import typer
from .config import load_config
from .backends.torch_backend import check_torch_env
from .backends.jax_backend import check_jax_env
from .models.inspect import inspect_model as _inspect_model
from .models.lora_targets import inspect_lora_targets as _inspect_lora_targets
from .models.metadata import get_model_config_metadata
from .train.qlora import train_qlora as _train_qlora
from .train.lora import train_lora as _train_lora
from .eval.smoke_eval import run_smoke_eval
from .export.merge_adapter import merge_adapter as _merge_adapter
from .artifacts.safetensors_io import inspect_safetensors_file
from .datasets import dataset_stats as _dataset_stats
from .logging import logger
import json

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
def inspect_model(model_id: str = typer.Option(..., help="Hugging Face model ID")):
    """Inspect model architecture (loads weights)."""
    _inspect_model(model_id)

@app.command()
def model_metadata(
    model_id: str = typer.Option(..., help="Hugging Face model ID"),
    trust_remote_code: bool = typer.Option(True, help="Trust remote code")
):
    """Inspect model metadata without loading weights."""
    meta = get_model_config_metadata(model_id, trust_remote_code)
    logger.info(json.dumps(meta, indent=2))

@app.command()
def inspect_lora_targets(
    model_id: str = typer.Option(..., help="Hugging Face model ID"),
    out: str = typer.Option(None, help="Output JSON path"),
    trust_remote_code: bool = typer.Option(True, help="Trust remote code")
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
def smoke_eval(adapter_path: str = typer.Option(..., help="Path to PEFT adapter")):
    """Run a smoke evaluation on an adapter."""
    import os
    try:
        with open(os.path.join(adapter_path, "adapter_config.json")) as f:
            base_model = json.load(f).get("base_model_name_or_path")
        if not base_model:
            raise ValueError("base_model_name_or_path not found in adapter_config.json")
    except Exception as e:
        logger.error(f"Could not infer base model: {e}")
        raise typer.Exit(code=1)
        
    run_smoke_eval(base_model, adapter_path)

@app.command()
def merge_adapter(
    base_model: str = typer.Option(..., help="Base model ID"),
    adapter_path: str = typer.Option(..., help="Path to PEFT adapter"),
    out_dir: str = typer.Option(..., help="Output directory")
):
    """Merge PEFT adapter into base model."""
    _merge_adapter(base_model, adapter_path, out_dir)

@app.command()
def inspect_safetensors(path: str = typer.Option(..., help="Path to safetensors file")):
    """Inspect a safetensors file."""
    info = inspect_safetensors_file(path)
    logger.info(json.dumps(info, indent=2))

@app.command()
def dataset_stats(
    path: str = typer.Option(..., help="Path to JSONL dataset"),
    model_id: str = typer.Option(..., help="Model ID for tokenizer")
):
    """Get dataset token statistics."""
    _dataset_stats(path, model_id)

if __name__ == "__main__":
    app()
