"""The shared Typer application every command module registers against.

`cli.py` stayed a single 661-line file holding sixteen unrelated commands until
it breached the file-size gate. The commands now live in `_cli_*` siblings
grouped by responsibility, and they all decorate this one `app`, so the console
entry point and every command name are unchanged.
"""

import typer

app = typer.Typer(help="Agoge Forger CLI")
