"""Environment and artifact inspection commands."""

import json

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from .artifacts.safetensors_io import inspect_safetensors_file
from .backends.torch_backend import check_torch_env
from .logging import logger
from .models.inspect import inspect_model as _inspect_model
from .models.lora_targets import inspect_lora_targets as _inspect_lora_targets
from .models.metadata import get_model_config_metadata
from .path_safety import resolve_existing_path


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
def inspect_safetensors(path: str = typer.Option(..., help="Path to safetensors file")):
    """Inspect a safetensors file."""
    try:
        safe_path = str(resolve_existing_path(path, must_be_file=True))
        info = inspect_safetensors_file(safe_path)
    except CLI_PATH_ERRORS as e:
        exit_on_error(e)
    except Exception as e:  # noqa: BLE001
        # safetensors raises its own SafetensorError for a malformed header, which
        # is not an OSError and so slips past inspect_safetensors_file's own
        # handling. A corrupt file is an operator-facing error, not a traceback.
        exit_on_error(e)
    logger.info(json.dumps(info, indent=2))
