"""Offline reproducibility-bundle verification command."""

from __future__ import annotations

import json
from typing import Annotated, NoReturn

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from ._cli_runs import _quiet_logger
from .path_safety import resolve_existing_path
from .release.report import (
    BundleVerificationReport,
    BundleVerifyFormat,
    format_verification_table,
)
from .release.verify import verify_reproducibility_bundle


def _emit_verification(report: BundleVerificationReport, output_format: BundleVerifyFormat) -> None:
    if output_format == BundleVerifyFormat.table:
        typer.echo(format_verification_table(report))
        return
    typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def _fail_verify(exc: BaseException, *, quiet: bool) -> NoReturn:
    if quiet:
        typer.echo(json.dumps({"error": str(exc), "verdict": "fail"}, indent=2, sort_keys=True))
        raise typer.Exit(code=1)
    exit_on_error(exc)


@app.command("verify-bundle")
def verify_bundle(
    bundle_dir: str = typer.Argument(..., help="Reproducibility bundle directory"),
    output_format: Annotated[
        BundleVerifyFormat, typer.Option("--format", help="Report format")
    ] = BundleVerifyFormat.json,
):
    """Verify a reproducibility bundle offline without loading model weights."""

    quiet = output_format == BundleVerifyFormat.json
    try:
        with _quiet_logger(quiet):
            resolved = resolve_existing_path(bundle_dir, must_be_dir=True)
            report = verify_reproducibility_bundle(resolved)
    except CLI_PATH_ERRORS as exc:
        _fail_verify(exc, quiet=quiet)
    _emit_verification(report, output_format)
    if report.verdict != "pass":
        raise typer.Exit(code=1)
