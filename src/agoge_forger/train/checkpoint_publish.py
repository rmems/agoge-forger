"""Atomic checkpoint directory publication with rollback into quarantine."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .._atomic_directory import rename_noreplace, require_rename_noreplace_support
from .._atomic_file import _fsync_directory
from .._run_status_safetensors import safetensors_usable
from .checkpoints import (
    CHECKPOINT_STAGING_PREFIX,
    incomplete_checkpoint_reason,
    quarantine_incomplete_checkpoints,
    quarantine_tree,
)

Rename = Callable[[Path, Path], None]
CheckpointWriter = Callable[[Path], None]


class IncompleteCheckpointError(ValueError):
    """Staging tree is not a complete checkpoint and must not be published."""

    def __init__(self, reason: str, path: Path) -> None:
        super().__init__(f"incomplete checkpoint ({reason}): {path}")
        self.reason = reason
        self.path = path


@dataclass(frozen=True)
class PublishOptions:
    rename: Rename | None = None
    allow_unsafe: bool = False


@dataclass(frozen=True)
class _FailedPublish:
    staged: Path
    destination: Path
    publish: PublishOptions
    reason: str | None


def publish_directory_noreplace(
    staged: Path,
    destination: Path,
    *,
    rename: Rename | None = None,
) -> Path:
    """Publish a fully written directory with a no-replace atomic rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_rename_noreplace_support(destination.parent)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing checkpoint path: {destination}")
    publisher = rename or rename_noreplace
    publisher(staged, destination)
    _fsync_directory(destination.parent)
    return destination


def publish_checkpoint_tree(
    run_dir: Path,
    step: int,
    write_tree: CheckpointWriter,
    options: PublishOptions | None = None,
) -> Path:
    """Write a checkpoint in staging and rename it onto `checkpoint-{step}`."""
    publish = options or PublishOptions()
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / f"checkpoint-{step}"
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing checkpoint path: {destination}")
    staging_root = Path(tempfile.mkdtemp(prefix=CHECKPOINT_STAGING_PREFIX, dir=run_dir))
    staged = staging_root / destination.name
    try:
        return _stage_and_publish(staged, destination, write_tree, publish)
    except BaseException as exc:
        rollback_reason = exc.reason if isinstance(exc, IncompleteCheckpointError) else None
        _rollback_failed_publish(
            run_dir,
            _FailedPublish(staged, destination, publish, rollback_reason),
        )
        raise
    finally:
        _reclaim_staging_root(run_dir, staging_root)


def rollback_run_dir(run_dir: Path, *, allow_unsafe: bool = False) -> list[Path]:
    """Quarantine leftover staging and incomplete checkpoint trees after a fault."""
    return quarantine_incomplete_checkpoints(run_dir, allow_unsafe=allow_unsafe)


def _stage_and_publish(
    staged: Path,
    destination: Path,
    write_tree: CheckpointWriter,
    publish: PublishOptions,
) -> Path:
    staged.mkdir()
    write_tree(staged)
    reason = _publish_refusal_reason(staged, allow_unsafe=publish.allow_unsafe)
    if reason is not None:
        raise IncompleteCheckpointError(reason, staged)
    return publish_directory_noreplace(staged, destination, rename=publish.rename)


def _publish_refusal_reason(staged: Path, *, allow_unsafe: bool) -> str | None:
    reason = incomplete_checkpoint_reason(staged, allow_unsafe=allow_unsafe)
    if reason is not None:
        return reason
    weights = staged / "adapter_model.safetensors"
    if weights.is_file() and not safetensors_usable(weights):
        return "short_write"
    return None


def _rollback_failed_publish(run_dir: Path, failed: _FailedPublish) -> None:
    if os.path.lexists(failed.destination):
        destination_reason = _publish_refusal_reason(
            failed.destination, allow_unsafe=failed.publish.allow_unsafe
        )
        if destination_reason is not None:
            quarantine_tree(failed.destination, reason=destination_reason, run_dir=run_dir)
    if os.path.lexists(failed.staged):
        staged_reason = failed.reason or _publish_refusal_reason(
            failed.staged, allow_unsafe=failed.publish.allow_unsafe
        )
        quarantine_tree(
            failed.staged,
            reason=staged_reason or "interrupted_save",
            run_dir=run_dir,
        )


def _reclaim_staging_root(run_dir: Path, staging_root: Path) -> None:
    if not staging_root.exists():
        return
    try:
        leftover = any(staging_root.iterdir())
    except OSError:
        leftover = True
    if leftover:
        quarantine_tree(staging_root, reason="leftover_staging", run_dir=run_dir)
        return
    staging_root.rmdir()
