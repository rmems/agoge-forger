"""Run-lifecycle commands: report readiness, then reclaim disk."""

import json
from typing import Annotated, Any

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from .path_safety import resolve_existing_path
from .run_status import (
    RunStatusFormat,
    build_run_status,
    format_run_status_table,
)


def _resolve_run_status_run_dir(run_dir: str) -> str:
    # Keep RuntimeError in the controlled boundary for platforms whose user-home
    # expansion can raise it; POSIX normally leaves an unknown `~user` unresolved,
    # which becomes FileNotFoundError during strict resolution.
    try:
        return str(resolve_existing_path(run_dir, must_be_dir=True))
    except CLI_PATH_ERRORS as e:
        exit_on_error(e)


def _resolve_optional_merged_dir(merged_dir: str | None) -> str | None:
    """Resolve --merged-dir, or keep the raw path when it is not exported yet."""
    if merged_dir is None:
        return None
    try:
        return str(resolve_existing_path(merged_dir, must_be_dir=True))
    except FileNotFoundError:
        # A merged model that has not been exported yet is a legitimate
        # "not ready" answer, so report it as absent instead of failing.
        return merged_dir
    except CLI_PATH_ERRORS as e:
        exit_on_error(e)


def _emit_run_status(report: dict[str, Any], output_format: RunStatusFormat) -> None:
    # stdout, not the logger: the JSON report is meant to be piped into jq.
    if output_format == RunStatusFormat.table:
        typer.echo(format_run_status_table(report))
        return
    typer.echo(json.dumps(report, indent=2))


@app.command()
def run_status(
    run_dir: str = typer.Argument(..., help="Run directory to inspect (adapters/<run_name>)"),
    merged_dir: str | None = typer.Option(
        None, "--merged-dir", help="Merged model directory (defaults to merged/<run_name>)"
    ),
    output_format: Annotated[
        RunStatusFormat, typer.Option("--format", help="Report format")
    ] = RunStatusFormat.json,
    allow_unsafe_serialization: bool = typer.Option(
        False, help="Accept legacy .bin adapter artifacts"
    ),
):
    """Report resume/export readiness for a training run directory."""
    # Validate first so bad paths still exit 1, then pass the original
    # argument so build_run_status can keep the logical adapters/<run>
    # path for conventional merged/<run_name> discovery.
    _resolve_run_status_run_dir(run_dir)
    safe_merged_dir = _resolve_optional_merged_dir(merged_dir)
    try:
        report = build_run_status(
            run_dir,
            merged_dir=safe_merged_dir,
            allow_unsafe=allow_unsafe_serialization,
        )
    except (ValueError, OSError) as e:
        # Inspection walks the run directory, so a permission or I/O failure can
        # surface here rather than at path resolution. Report it the same way as
        # a bad path — a logged error and exit 1 — instead of a raw traceback.
        exit_on_error(e)
    _emit_run_status(report, output_format)
