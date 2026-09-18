"""Prometheus consumer-contract command: parse derivatives without training."""

from typing import Annotated

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from .consumer_contract import ConsumerContractError, run_consumer_contract
from .logging import logger


@app.command("consumer-contract")
def consumer_contract(
    input_path: Annotated[
        list[str],
        typer.Option("--input", help="JSONL with messages or instruction/input/output rows"),
    ],
    out_dir: Annotated[str, typer.Option(help="Directory for provenance sidecars")],
):
    """Load sample derivatives through the current dataset parser."""
    try:
        sidecars = run_consumer_contract(input_path, out_dir)
    except (*CLI_PATH_ERRORS, ConsumerContractError) as exc:
        exit_on_error(exc)
    logger.info("Wrote %s consumer-contract sidecar(s) to %s", len(sidecars), out_dir)
