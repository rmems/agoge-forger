"""Training entry points."""

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from .config import load_config
from .train.lora import train_lora as _train_lora
from .train.qlora import train_qlora as _train_qlora


def _load(config: str):
    """Read the training config, reporting a bad path or bad YAML as exit 1.

    `load_config` resolves the path, requires `model_id` and `dataset_path`, and
    stats the dataset — all ordinary operator mistakes that should not surface as
    a traceback.
    """
    try:
        return load_config(config)
    except CLI_PATH_ERRORS as e:
        exit_on_error(e)


@app.command()
def train_qlora(config: str = typer.Option(..., help="Path to YAML config")):
    """Run QLoRA training."""
    cfg = _load(config)
    _train_qlora(cfg)


@app.command()
def train_lora(config: str = typer.Option(..., help="Path to YAML config")):
    """Run LoRA training."""
    cfg = _load(config)
    _train_lora(cfg)
