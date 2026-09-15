"""Training entry points."""

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from .config import ExperimentConfig, load_config
from .telemetry.window import overlay_cli_telemetry
from .train.lora import train_lora as _train_lora
from .train.qlora import train_qlora as _train_qlora

_RUN_ID_HELP = "Stable agoge_run_id for GPU/CUDA correlation (default: config run_name)"
_PROFILE_WINDOW_HELP = (
    "Opt-in bounded CUDA profile window, e.g. train:1-2. Off unless set. "
    "Not required for ordinary training."
)


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


def _apply_telemetry_cli(
    cfg: ExperimentConfig,
    run_id: str | None,
    profile_window: str | None,
) -> ExperimentConfig:
    try:
        cfg.telemetry = overlay_cli_telemetry(cfg.telemetry, run_id, profile_window)
    except ValueError as error:
        exit_on_error(error)
    return cfg


@app.command()
def train_qlora(
    config: str = typer.Option(..., help="Path to YAML config"),
    run_id: str | None = typer.Option(None, help=_RUN_ID_HELP),
    profile_window: str | None = typer.Option(None, help=_PROFILE_WINDOW_HELP),
):
    """Run QLoRA training."""
    cfg = _apply_telemetry_cli(_load(config), run_id, profile_window)
    _train_qlora(cfg)


@app.command()
def train_lora(
    config: str = typer.Option(..., help="Path to YAML config"),
    run_id: str | None = typer.Option(None, help=_RUN_ID_HELP),
    profile_window: str | None = typer.Option(None, help=_PROFILE_WINDOW_HELP),
):
    """Run LoRA training."""
    cfg = _apply_telemetry_cli(_load(config), run_id, profile_window)
    _train_lora(cfg)
