import typer
from .config import load_config
from .backends.torch_backend import check_torch_env
from .backends.jax_backend import check_jax_env
from .models.inspect import inspect_model as _inspect_model
from .models.lora_targets import inspect_lora_targets as _inspect_lora_targets
from .train.qlora import train_qlora as _train_qlora
from .train.lora import train_lora as _train_lora
from .eval.smoke_eval import run_smoke_eval
from .export.merge_adapter import merge_adapter as _merge_adapter
from .logging import logger

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
    """Inspect model architecture."""
    _inspect_model(model_id)

@app.command()
def inspect_lora_targets(model_id: str = typer.Option(..., help="Hugging Face model ID")):
    """Inspect model for potential LoRA targets."""
    _inspect_lora_targets(model_id)

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
    # Try to infer base model from adapter config
    import json
    import os
    try:
        with open(os.path.join(adapter_path, "adapter_config.json")) as f:
            base_model = json.load(f).get("base_model_name_or_path")
    except Exception:
        logger.error("Could not find adapter_config.json to infer base model.")
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

if __name__ == "__main__":
    app()
