"""Agoge Forger Typer CLI entry point.

The commands live in `_cli_*` modules grouped by responsibility. Importing each
module is what registers its commands against the shared `app`, so the console
script stays `agoge_forger.cli:app` and no command name or invocation changes.

The modules are imported by name rather than with `import` statements because
nothing here references them: as plain imports they read as unused to every
linter, and the suppressions needed to quiet that would also hide a genuinely
dead one. Listing them makes the registration explicit, and the order is the
order `agoge --help` prints, pinned to the pre-split listing.
"""

from importlib import import_module

from ._cli_app import app

COMMAND_MODULES = (
    "_cli_env",
    "_cli_train",
    "_cli_export",
    "_cli_runs",
    "_cli_data",
    "_cli_serving",
)

for _module_name in COMMAND_MODULES:
    import_module(f".{_module_name}", __package__)

__all__ = ["COMMAND_MODULES", "app"]


if __name__ == "__main__":
    app()
