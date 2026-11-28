"""Offline preparation of explicit completion-only rows."""

from typing import Annotated

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from .compact_completion import CompletionFormat
from .completion_admission import CompletionAdmissionPolicy, admit_completions
from .logging import logger


@app.command("admit-completions")
def admit_completions_command(
    source: str = typer.Option(..., help="JSONL with exact text and completion_start_char"),
    tokenizer_dir: str = typer.Option(..., help="Local saved fast tokenizer directory"),
    output_dir: str = typer.Option(..., help="New local admission directory"),
    max_seq_length: int = typer.Option(..., min=1, help="Inclusive end-to-end token budget"),
    completion_format: Annotated[
        CompletionFormat, typer.Option(help="Input rendering contract before token admission")
    ] = CompletionFormat.FROZEN_TEXT,
):
    """Filter completion rows before freezing, without truncation or model loading."""
    try:
        report = admit_completions(
            source,
            tokenizer_dir,
            output_dir,
            CompletionAdmissionPolicy(max_seq_length, completion_format),
        )
    except CLI_PATH_ERRORS as exc:
        exit_on_error(exc)
    else:
        logger.info("Admitted %s/%s rows", report["admitted_rows"], report["source_rows"])
