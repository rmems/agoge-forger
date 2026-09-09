"""Dataset statistics and immutable split materialization."""

from dataclasses import dataclass

import typer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from .datasets import dataset_stats as _dataset_stats
from .logging import logger
from .path_safety import resolve_absent_output_directory, resolve_existing_path
from .split_contract import (
    SPLIT_NAMES,
    CanonicalIdentityPolicy,
    SplitManifest,
    SplitMaterializationSpec,
    SplitPolicy,
    materialize_split,
)


@app.command()
def dataset_stats(
    path: str = typer.Option(..., help="Path to JSONL dataset"),
    model_id: str = typer.Option(..., help="Model ID for tokenizer"),
    trust_remote_code: bool = typer.Option(False, help="Trust remote code from the model repo"),
):
    """Get dataset token statistics."""
    safe_path = str(resolve_existing_path(path, must_be_file=True))
    _dataset_stats(safe_path, model_id, trust_remote_code=trust_remote_code)


@dataclass(frozen=True)
class _FreezeSplitOptions:
    source_repository: str
    source_revision: str
    dataset_version: str
    source_path: str
    seed: int
    salt: str
    train_weight: int
    validation_weight: int
    held_out_weight: int
    canonical_id_field: str
    lineage_id_field: str
    group_id_field: str


@app.command("freeze-split")
def freeze_split(
    source: str = typer.Option(..., help="Versioned curated source JSONL"),
    source_path: str = typer.Option(
        ...,
        help="Canonical repository-relative path of the source at --source-revision",
    ),
    output_dir: str = typer.Option(..., help="New immutable snapshot directory"),
    source_repository: str = typer.Option(...),
    source_revision: str = typer.Option(...),
    dataset_version: str = typer.Option(...),
    seed: int = typer.Option(...),
    salt: str = typer.Option(...),
    train_weight: int = typer.Option(80),
    validation_weight: int = typer.Option(10),
    held_out_weight: int = typer.Option(10),
    canonical_id_field: str = typer.Option("canonical_id"),
    lineage_id_field: str = typer.Option("lineage_id"),
    group_id_field: str = typer.Option("group_id"),
):
    """Materialize an immutable three-way SFT split from a pinned local source."""
    try:
        safe_source = resolve_existing_path(source, must_be_file=True)
        safe_output = resolve_absent_output_directory(output_dir)
        spec = _freeze_split_spec(
            _FreezeSplitOptions(
                source_repository=source_repository,
                source_revision=source_revision,
                dataset_version=dataset_version,
                source_path=source_path,
                seed=seed,
                salt=salt,
                train_weight=train_weight,
                validation_weight=validation_weight,
                held_out_weight=held_out_weight,
                canonical_id_field=canonical_id_field,
                lineage_id_field=lineage_id_field,
                group_id_field=group_id_field,
            )
        )
        manifest = materialize_split(safe_source, safe_output, spec)
    except CLI_PATH_ERRORS as e:
        exit_on_error(e)
    else:
        _log_freeze_split(manifest, str(safe_output))


def _freeze_split_spec(options: _FreezeSplitOptions) -> SplitMaterializationSpec:
    return SplitMaterializationSpec(
        source_repository=options.source_repository,
        source_revision=options.source_revision,
        dataset_version=options.dataset_version,
        source_path=options.source_path,
        split_policy=SplitPolicy(
            seed=options.seed,
            salt=options.salt,
            weights={
                "train": options.train_weight,
                "validation": options.validation_weight,
                "held_out": options.held_out_weight,
            },
        ),
        canonical_identity=CanonicalIdentityPolicy(
            canonical_id_field=options.canonical_id_field,
            lineage_id_field=options.lineage_id_field,
            group_id_field=options.group_id_field,
            content_hash_policy="normalized-training-payload-v1",
        ),
    )


def _log_freeze_split(manifest: SplitManifest, output_dir: str) -> None:
    logger.info(
        "source: %s@%s:%s",
        manifest.source.repository,
        manifest.source.revision,
        manifest.source.path,
    )
    logger.info("wrote %s/split_manifest.json", output_dir)
    logger.info("wrote %s/split_report.md", output_dir)
    counts = ", ".join(f"{name}={manifest.splits[name].record_count}" for name in SPLIT_NAMES)
    logger.info("counts: %s", counts)
