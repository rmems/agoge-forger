"""Evaluation, adapter merge, and final-model export commands."""

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from .artifacts.producer_provenance import producer_provenance_from_adapter
from .artifacts.safetensors_io import assert_no_unsafe_weight_bins
from .eval.smoke_eval import run_smoke_eval
from .export.merge_adapter import export_final_model as _export_final_model
from .export.merge_adapter import merge_adapter as _merge_adapter
from .logging import logger
from .path_safety import resolve_existing_path, resolve_output_directory
from .train.checkpoints import (
    infer_base_model_from_adapter,
    is_adapter_artifact,
    resolve_export_source,
)

_TRUST_REMOTE_CODE_HELP = "Trust remote code from the model repo"


@app.command()
def smoke_eval(
    adapter_path: str = typer.Option(..., help="Path to PEFT adapter"),
    trust_remote_code: bool = typer.Option(False, help=_TRUST_REMOTE_CODE_HELP),
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
    trust_remote_code: bool = typer.Option(False, help=_TRUST_REMOTE_CODE_HELP),
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
        producer_provenance=producer_provenance_from_adapter(safe_adapter_path),
    )


def _run_export(**kwargs) -> None:
    """Run the export, reporting its input rejections through the boundary.

    `_export_final_model` still validates on its own — `save_safetensors=False`
    is refused outright under Transformers 5 — and those rejections are operator
    input errors, not crashes.
    """
    try:
        _export_final_model(**kwargs)
    except CLI_PATH_ERRORS as e:
        exit_on_error(e)


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
    trust_remote_code: bool = typer.Option(False, help=_TRUST_REMOTE_CODE_HELP),
):
    """Export one final merged model from the latest valid checkpoint or adapter."""
    safe_out_dir = str(resolve_output_directory(out_dir))
    # Selecting the source and reading its sealed provenance both happen before a
    # single weight is loaded, and both fail on ordinary operator mistakes: a run
    # with no exportable artifact, or an adapter never finalized with an
    # artifact_index.json. Report those the way the rest of the boundary does.
    try:
        safe_run_dir = str(resolve_existing_path(run_dir, must_be_dir=True)) if run_dir else None
        safe_adapter_path = (
            str(resolve_existing_path(adapter_path, must_be_dir=True)) if adapter_path else None
        )
        source_adapter = resolve_export_source(
            run_dir=safe_run_dir,
            adapter_path=safe_adapter_path,
            allow_unsafe=allow_unsafe_serialization,
        )
        provenance = producer_provenance_from_adapter(source_adapter)
    except (*CLI_PATH_ERRORS, TypeError) as e:
        # TypeError joins the tuple because _load_artifact_index_payload raises it
        # -- not ValueError -- for an index that parses as valid JSON but is not
        # an object (`[]`, `"text"`, `3`).
        exit_on_error(e)
    _run_export(
        out_dir=safe_out_dir,
        run_dir=safe_run_dir,
        adapter_path=safe_adapter_path,
        base_model_id=base_model,
        save_safetensors=save_safetensors,
        allow_unsafe=allow_unsafe_serialization,
        max_shard_size=max_shard_size,
        trust_remote_code=trust_remote_code,
        producer_provenance=provenance,
    )
