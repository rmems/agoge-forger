"""Freeze and run the first Granite 4.1 base-versus-SFT experiment."""

from pathlib import Path

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from .experiment.execute import execute_experiment
from .experiment.freeze import freeze_experiment_contract
from .experiment.local import local_phases
from .logging import logger
from .manifests import get_git_info


def _require_clean_commit() -> tuple[str, bool]:
    info = get_git_info()
    commit = info.get("commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError("Agoge git commit is unavailable")
    return commit, not bool(info.get("dirty", True))


@app.command("freeze-granite-first-sft")
def freeze_granite_first_sft(
    experiment_id: str = typer.Option(..., help="Directory name under granite-4.1-first-sft/"),
    output_dir: str = typer.Option(..., help="New reports/granite-4.1-first-sft/<experiment_id>"),
    split_manifest: str = typer.Option(..., help="Frozen split_manifest.json"),
    qualification_report: str = typer.Option(
        ..., help="Passing agoge.model-compatibility.v1 report for Granite 4.1"
    ),
) -> None:
    """Freeze the pre-registered contract before any measured G0 inspection."""

    try:
        commit, clean = _require_clean_commit()
        if not clean:
            raise ValueError("freeze requires a clean Agoge worktree")
        contract = freeze_experiment_contract(
            experiment_id=experiment_id,
            output_dir=output_dir,
            split_manifest=split_manifest,
            qualification_report=qualification_report,
            agoge_commit=commit,
        )
    except (*CLI_PATH_ERRORS, TypeError) as exc:
        exit_on_error(exc)
    logger.info("froze Granite experiment %s at commit %s", contract.experiment_id, commit)


@app.command("run-granite-first-sft")
def run_granite_first_sft(
    contract: str = typer.Option(..., help="Frozen experiment-contract.json"),
) -> None:
    """Evaluate G0, train G1, clean-reload, then score G1 on one local GPU."""

    try:
        commit, clean = _require_clean_commit()
        outcome = execute_experiment(
            contract,
            agoge_commit=commit,
            worktree_clean=clean,
            phases=local_phases(Path(contract).expanduser().resolve()),
        )
    except (*CLI_PATH_ERRORS, TypeError) as exc:
        exit_on_error(exc)
    logger.info("Granite experiment %s", outcome.status)
    if outcome.status != "scored":
        raise typer.Exit(code=1)
