"""Agoge Forger Typer CLI entry point.

The commands live in `_cli_*` modules grouped by responsibility. Importing each
module is what registers its commands against the shared `app`, so the console
script stays `agoge_forger.cli:app` and no command name or invocation changes.

Nothing here references the imported modules, so they read as unused: the
`pylint: disable` below is scoped to just that block, matching how this file
already scopes `too-many-arguments` around the wide serving commands.
"""

# ruff: noqa: I001
# Import order is registration order, which is the order `agoge --help` lists
# commands in. It is pinned to the pre-split listing rather than sorted, so the
# help output a reader already knows does not get reshuffled by a refactor.
from ._cli_app import app

# pylint: disable=unused-import
from . import _cli_env  # noqa: F401
from . import _cli_train  # noqa: F401
from . import _cli_export  # noqa: F401
from . import _cli_runs  # noqa: F401
from . import _cli_data  # noqa: F401
from . import _cli_serving  # noqa: F401

# pylint: enable=unused-import

__all__ = ["app"]


if __name__ == "__main__":
    app()
