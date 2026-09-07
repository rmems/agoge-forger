"""Agoge Forger Typer CLI entry point.

The commands live in `_cli_*` modules grouped by responsibility; importing them
here is what registers each one against the shared `app`. The console script
stays `agoge_forger.cli:app`, so no command name or invocation changes.
"""

# ruff: noqa: I001
# Import order is registration order, which is the order `agoge --help` lists
# commands in. It is pinned to the pre-split listing rather than sorted, so the
# help output a reader already knows does not get reshuffled by a refactor.
from ._cli_app import app
from . import _cli_env  # noqa: F401
from . import _cli_train  # noqa: F401
from . import _cli_export  # noqa: F401
from . import _cli_runs  # noqa: F401
from . import _cli_data  # noqa: F401
from . import _cli_serving  # noqa: F401

__all__ = ["app"]


if __name__ == "__main__":
    app()
