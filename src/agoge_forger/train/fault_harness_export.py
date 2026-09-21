"""Final adapter export for the checkpoint fault harness."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .._atomic_directory import rename_noreplace, require_rename_noreplace_support
from .checkpoints import EXPORT_STAGING_PREFIX, is_adapter_artifact, quarantine_tree
from .fault_harness_faults import FaultSpec, raise_if_fault
from .fault_harness_payloads import ADAPTER_FILENAMES, write_adapter_files


def export_final_adapter(run_dir: Path, *, max_steps: int, fault: FaultSpec | None) -> None:
    if is_adapter_artifact(run_dir):
        return
    require_rename_noreplace_support(run_dir)
    staging_root = Path(tempfile.mkdtemp(prefix=EXPORT_STAGING_PREFIX, dir=run_dir))
    staged = staging_root / "adapter"
    published: list[Path] = []
    try:
        published.extend(
            _publish_run_root_adapter(run_dir, staged, max_steps=max_steps, fault=fault)
        )
    except BaseException:
        _quarantine_failed_export(run_dir, staged, published)
        raise
    finally:
        _reclaim_export_staging(run_dir, staging_root)


def _publish_run_root_adapter(
    run_dir: Path,
    staged: Path,
    *,
    max_steps: int,
    fault: FaultSpec | None,
) -> list[Path]:
    staged.mkdir()
    write_adapter_files(staged)
    raise_if_fault(fault, max_steps, "during_export")
    for filename in ADAPTER_FILENAMES:
        destination = run_dir / filename
        if os.path.lexists(destination):
            raise FileExistsError(f"refusing to overwrite existing adapter file: {destination}")
    published: list[Path] = []
    for filename in ADAPTER_FILENAMES:
        destination = run_dir / filename
        _publish_adapter_file(staged / filename, destination)
        published.append(destination)
    _rmdir_if_empty(staged)
    return published


def _publish_adapter_file(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing adapter file: {destination}")
    rename_noreplace(source, destination)


def _quarantine_failed_export(run_dir: Path, staged: Path, published: list[Path]) -> None:
    for leaked in published:
        if os.path.lexists(leaked):
            quarantine_tree(leaked, reason="export_failure", run_dir=run_dir)
    if staged.exists():
        quarantine_tree(staged, reason="export_failure", run_dir=run_dir)


def _reclaim_export_staging(run_dir: Path, staging_root: Path) -> None:
    if not staging_root.exists() and not staging_root.is_symlink():
        return
    try:
        leftover = staging_root.is_symlink() or any(staging_root.iterdir())
    except OSError:
        leftover = True
    if leftover:
        quarantine_tree(staging_root, reason="leftover_staging", run_dir=run_dir)
        return
    try:
        staging_root.rmdir()
    except OSError:
        quarantine_tree(staging_root, reason="leftover_staging", run_dir=run_dir)


def _rmdir_if_empty(path: Path) -> None:
    try:
        leftover = any(path.iterdir())
    except OSError:
        leftover = True
    if leftover:
        return
    try:
        path.rmdir()
    except OSError:
        return
