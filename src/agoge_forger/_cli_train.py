"""Training entry points."""

import typer

from ._cli_app import app
from .config import load_config
from .train.lora import train_lora as _train_lora
from .train.qlora import train_qlora as _train_qlora


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
