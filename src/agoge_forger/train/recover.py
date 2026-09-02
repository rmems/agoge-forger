"""Recovery of the final immutable marker for completed frozen training runs."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from safetensors import SafetensorError, safe_open

from .._atomic_directory import rename_noreplace, require_rename_noreplace_support
from ..artifacts.safetensors_io import write_artifact_index_noreplace
from ..eval._artifact_schema import ArtifactIndex
from ..path_safety import resolve_existing_path
from .checkpoints import (
    infer_base_model_from_adapter,
    infer_base_revision_from_adapter,
    is_adapter_artifact,
)
from .trainer import _bind_frozen_source, _frozen_producer_provenance, _reject_frozen_resume


def recover_frozen_artifact_index(config) -> Path:
    """Publish a missing marker without loading a model or resuming training."""

    if config.runtime.allow_unsafe_serialization:
        raise ValueError("frozen artifact-index recovery requires unsafe serialization disabled")
    _reject_frozen_resume(config)
    binding = _bind_frozen_source(config)
    if binding is None:
        raise ValueError("artifact-index recovery requires a frozen split configuration")
    run_dir = resolve_existing_path(
        str(Path(config.output_dir).expanduser() / config.run_name),
        must_be_dir=True,
    )
    _require_completed_adapter(run_dir, config)
    index_path = run_dir / "artifact_index.json"
    _quarantine_invalid_index(index_path, run_dir.parent)
    provenance = _frozen_producer_provenance(config, binding).model_dump(mode="json")
    write_artifact_index_noreplace(str(run_dir), producer_provenance=provenance)
    return index_path


def _require_completed_adapter(run_dir: Path, config) -> None:
    if run_dir.is_symlink() or not is_adapter_artifact(run_dir, allow_unsafe=False):
        raise ValueError(f"frozen recovery requires a safetensors-only root adapter: {run_dir}")
    if infer_base_model_from_adapter(run_dir) != config.model_id:
        raise ValueError("adapter base model does not match frozen training config")
    if infer_base_revision_from_adapter(run_dir) != config.revision:
        raise ValueError("adapter revision does not match frozen training config")
    _require_nonempty_safetensors(run_dir / "adapter_model.safetensors")


def _require_nonempty_safetensors(path: Path) -> None:
    try:
        with safe_open(path, framework="pt") as handle:
            if not list(handle.keys()):
                raise ValueError("adapter safetensors contains no tensors")
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"frozen recovery requires valid adapter safetensors: {path}") from exc


def _quarantine_invalid_index(index_path: Path, quarantine_parent: Path) -> None:
    if not os.path.lexists(index_path):
        return
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError(f"artifact index recovery refuses a non-regular marker: {index_path}")
    try:
        ArtifactIndex.model_validate_json(index_path.read_bytes())
    except (OSError, ValueError):
        _quarantine_index(index_path, quarantine_parent)
        return
    raise FileExistsError(f"refusing to replace a schema-valid artifact index: {index_path}")


def _quarantine_index(index_path: Path, quarantine_parent: Path) -> None:
    require_rename_noreplace_support(quarantine_parent)
    quarantine = quarantine_parent / (
        f".{index_path.parent.name}.artifact_index.invalid.{secrets.token_hex(8)}"
    )
    rename_noreplace(index_path, quarantine)
