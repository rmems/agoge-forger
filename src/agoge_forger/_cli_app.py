"""The shared Typer application and the CLI's common failure boundary.

`cli.py` stayed a single 661-line file holding sixteen unrelated commands until
it breached the file-size gate. The commands now live in `_cli_*` siblings
grouped by responsibility, and they all decorate this one `app`, so the console
entry point and every command name are unchanged.
"""

from typing import NoReturn

import typer

from .logging import logger

app = typer.Typer(help="Agoge Forger CLI")

# What a path resolution or a filesystem inspection can raise. `resolve_existing_path`
# raises ValueError for empty/traversal/wrong-type paths and propagates whatever
# `Path.resolve(strict=True)` raises, RuntimeError on a symlink loop included.
CLI_PATH_ERRORS = (FileNotFoundError, ValueError, NotADirectoryError, OSError, RuntimeError)


def exit_on_error(exc: BaseException) -> NoReturn:
    """Log a failure and stop the CLI with exit 1 rather than a raw traceback."""
    logger.error(str(exc))
    raise typer.Exit(code=1)
