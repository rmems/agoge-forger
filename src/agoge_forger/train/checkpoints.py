import json
import re
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._atomic_file import publish_bytes_replace
from .._run_status_safetensors import safetensors_usable
from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins
from ..config import normalize_revision
from ..logging import logger
from ..path_safety import contains_mount, resolve_existing_path

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors",)
LEGACY_ADAPTER_WEIGHT_FILES = ("adapter_model.bin",)
QUARANTINE_DIRNAME = ".agoge-quarantine"
QUARANTINE_REASON_FILENAME = "quarantine_reason.json"
CHECKPOINT_STAGING_PREFIX = ".agoge-ckpt-staging-"
EXPORT_STAGING_PREFIX = ".agoge-export-staging-"
_STAGING_PREFIXES = (CHECKPOINT_STAGING_PREFIX, EXPORT_STAGING_PREFIX)

PathLike = str | Path


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return -1
    return int(match.group(1))


def _has_unsafe_weight_bins(adapter_dir: Path) -> bool:
    """Return True if the adapter directory contains any unsafe weight files."""
    try:
        assert_no_unsafe_weight_bins(str(adapter_dir), recursive=True)
        return False
    except RuntimeError:
        return True


def is_adapter_artifact(path: PathLike, *, allow_unsafe: bool = False) -> bool:
    """Return True when ``path`` looks like a PEFT adapter directory.

    By default only safetensors-only adapters pass. With ``allow_unsafe=True``,
    legacy ``adapter_model.bin``-only trees are accepted for explicit opt-in flows.
    """
    adapter_dir = Path(path)
    if not adapter_dir.is_dir():
        return False
    if not (adapter_dir / "adapter_config.json").is_file():
        return False

    weight_files: tuple[str, ...] = ADAPTER_WEIGHT_FILES
    if allow_unsafe:
        weight_files = ADAPTER_WEIGHT_FILES + LEGACY_ADAPTER_WEIGHT_FILES

    if not any((adapter_dir / weight_file).is_file() for weight_file in weight_files):
        return False
    return not (not allow_unsafe and _has_unsafe_weight_bins(adapter_dir))


def is_valid_checkpoint(path: PathLike, *, allow_unsafe: bool = False) -> bool:
    """A checkpoint is valid iff it is a `checkpoint-N` directory with the
    required trainer state, AND it is a valid adapter artifact. Unsafe-bin
    filtering happens here so callers selecting among checkpoints never have
    to re-validate unless ``allow_unsafe`` is explicitly enabled.
    """
    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir():
        return False
    if checkpoint_step(checkpoint_dir) < 0:
        return False
    if not (checkpoint_dir / "trainer_state.json").is_file():
        return False
    return is_adapter_artifact(checkpoint_dir, allow_unsafe=allow_unsafe)


def list_valid_checkpoints(run_dir: PathLike, *, allow_unsafe: bool = False) -> list[Path]:
    root = Path(run_dir)
    if not root.is_dir():
        return []
    checkpoints = [
        path for path in root.iterdir() if is_valid_checkpoint(path, allow_unsafe=allow_unsafe)
    ]
    checkpoints.sort(key=checkpoint_step)
    return checkpoints


def find_latest_valid_checkpoint(run_dir: PathLike, *, allow_unsafe: bool = False) -> Path | None:
    checkpoints = list_valid_checkpoints(run_dir, allow_unsafe=allow_unsafe)
    if not checkpoints:
        return None
    return checkpoints[-1]


def _load_adapter_config(adapter_path: PathLike) -> dict:
    adapter_dir = Path(adapter_path)
    config_path = adapter_dir / "adapter_config.json"
    with config_path.open() as handle:
        return json.load(handle)


def infer_base_model_from_adapter(adapter_path: PathLike) -> str:
    adapter_config = _load_adapter_config(adapter_path)
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"base_model_name_or_path not found in {Path(adapter_path) / 'adapter_config.json'}"
        )
    return base_model


def infer_base_revision_from_adapter(adapter_path: PathLike) -> str | None:
    """Return the normalized Hub revision persisted on a PEFT adapter."""
    revision = _load_adapter_config(adapter_path).get("revision")
    return normalize_revision(revision)


def incomplete_checkpoint_reason(path: PathLike, *, allow_unsafe: bool = False) -> str | None:
    """Return why a `checkpoint-N` tree is not resume-selectable, if it isn't."""
    checkpoint_dir = Path(path)
    if not _is_named_checkpoint_dir(checkpoint_dir):
        return None
    if not is_valid_checkpoint(checkpoint_dir, allow_unsafe=allow_unsafe):
        return _checkpoint_gap_reason(checkpoint_dir, allow_unsafe=allow_unsafe)
    return _unreadable_weight_reason(checkpoint_dir)


def _is_named_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink() and checkpoint_step(path) >= 0


def _checkpoint_gap_reason(checkpoint_dir: Path, *, allow_unsafe: bool) -> str:
    if not (checkpoint_dir / "trainer_state.json").is_file():
        return "missing_trainer_state"
    if not (checkpoint_dir / "adapter_config.json").is_file():
        return "missing_adapter_config"
    if not _has_adapter_weights(checkpoint_dir, allow_unsafe=allow_unsafe):
        return "missing_adapter_weights"
    if not allow_unsafe and _has_unsafe_weight_bins(checkpoint_dir):
        return "unsafe_weight_bins"
    return "incomplete_checkpoint"


def _has_adapter_weights(adapter_dir: Path, *, allow_unsafe: bool) -> bool:
    weight_files: tuple[str, ...] = ADAPTER_WEIGHT_FILES
    if allow_unsafe:
        weight_files = ADAPTER_WEIGHT_FILES + LEGACY_ADAPTER_WEIGHT_FILES
    return any((adapter_dir / weight_file).is_file() for weight_file in weight_files)


def _unreadable_weight_reason(checkpoint_dir: Path) -> str | None:
    for name in ADAPTER_WEIGHT_FILES:
        path = checkpoint_dir / name
        if path.is_file() and not safetensors_usable(path):
            return "short_write"
    return None


def quarantine_root(run_dir: PathLike) -> Path:
    return Path(run_dir) / QUARANTINE_DIRNAME


def list_quarantined_checkpoints(run_dir: PathLike) -> list[Path]:
    root = quarantine_root(run_dir)
    if not root.is_dir():
        return []
    quarantined = [path for path in root.iterdir() if path.is_dir()]
    quarantined.sort(key=lambda path: path.name)
    return quarantined


def read_quarantine_reason(path: PathLike) -> dict[str, Any]:
    reason_path = Path(path) / QUARANTINE_REASON_FILENAME
    with reason_path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("reason"), str):
        raise TypeError(f"quarantine reason document is not usable: {reason_path}")
    return payload


def quarantine_tree(
    path: PathLike,
    *,
    reason: str,
    run_dir: PathLike | None = None,
    details: Mapping[str, Any] | None = None,
) -> Path:
    """Move an incomplete tree out of the resume scan with an explicit reason."""
    source = Path(path)
    root = Path(run_dir) if run_dir is not None else source.parent
    destination = _unique_quarantine_destination(root, source.name, reason)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "reason": reason,
        "original_path": str(source),
        "quarantined_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "details": dict(details or {}),
    }
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    _refuse_quarantine_mount(source)
    if source.is_symlink() or source.is_file():
        destination.mkdir()
        publish_bytes_replace(destination / QUARANTINE_REASON_FILENAME, encoded)
        if source.exists() or source.is_symlink():
            shutil.move(str(source), str(destination / source.name))
        return destination
    if source.is_dir():
        publish_bytes_replace(source / QUARANTINE_REASON_FILENAME, encoded)
        shutil.move(str(source), str(destination))
        return destination
    destination.mkdir()
    publish_bytes_replace(destination / QUARANTINE_REASON_FILENAME, encoded)
    return destination


def _refuse_quarantine_mount(source: Path) -> None:
    if source.is_symlink():
        return
    if contains_mount(source):
        raise ValueError(f"Refusing to quarantine a mount point: {source}")


def _unique_quarantine_destination(run_dir: Path, original_name: str, reason: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = quarantine_root(run_dir)
    for suffix in range(1000):
        marker = stamp if suffix == 0 else f"{stamp}-{suffix}"
        destination = root / f"{original_name}.{reason}.{marker}"
        if not destination.exists() and not destination.is_symlink():
            return destination
    raise RuntimeError(f"unable to allocate a unique quarantine path under {root}")


def quarantine_incomplete_checkpoints(
    run_dir: PathLike, *, allow_unsafe: bool = False
) -> list[Path]:
    """Move incomplete `checkpoint-*` trees and leftover staging out of the run."""
    root = Path(run_dir)
    if not root.is_dir():
        return []
    quarantined: list[Path] = []
    for entry in list(root.iterdir()):
        quarantined.extend(_quarantine_run_entry(root, entry, allow_unsafe=allow_unsafe))
    return quarantined


def _quarantine_run_entry(run_dir: Path, entry: Path, *, allow_unsafe: bool) -> list[Path]:
    if entry.is_symlink() and entry.name.startswith(_STAGING_PREFIXES):
        return [quarantine_tree(entry, reason="leftover_staging", run_dir=run_dir)]
    if _is_staging_dir(entry):
        return _quarantine_staging_tree(run_dir, entry, allow_unsafe=allow_unsafe)
    reason = incomplete_checkpoint_reason(entry, allow_unsafe=allow_unsafe)
    if reason is None:
        return []
    if contains_mount(entry):
        return []
    return [quarantine_tree(entry, reason=reason, run_dir=run_dir)]


def _is_staging_dir(path: Path) -> bool:
    return not path.is_symlink() and path.is_dir() and path.name.startswith(_STAGING_PREFIXES)


def _quarantine_staging_tree(run_dir: Path, staging: Path, *, allow_unsafe: bool) -> list[Path]:
    moved: list[Path] = []
    for child in list(staging.iterdir()) if staging.is_dir() else []:
        child_reason = incomplete_checkpoint_reason(child, allow_unsafe=allow_unsafe)
        if child_reason is None and checkpoint_step(child) < 0:
            continue
        moved.append(
            quarantine_tree(
                child,
                reason=child_reason or "interrupted_save",
                run_dir=run_dir,
            )
        )
    if staging.exists():
        moved.append(quarantine_tree(staging, reason="leftover_staging", run_dir=run_dir))
    return moved


def resolve_resume_checkpoint(run_dir: str, config) -> str | None:
    allow_unsafe = config.runtime.allow_unsafe_serialization
    quarantine_incomplete_checkpoints(run_dir, allow_unsafe=allow_unsafe)

    if config.training.resume_checkpoint_path:
        checkpoint_path = str(
            resolve_existing_path(config.training.resume_checkpoint_path, must_be_dir=True)
        )
        if not is_valid_checkpoint(checkpoint_path, allow_unsafe=allow_unsafe):
            raise ValueError(f"Configured resume checkpoint is not valid: {checkpoint_path}")
        logger.info(f"Resuming from explicit checkpoint {checkpoint_path}")
        return checkpoint_path

    if not config.training.resume_from_latest_checkpoint:
        return None

    latest = find_latest_valid_checkpoint(run_dir, allow_unsafe=allow_unsafe)
    if latest is not None:
        logger.info(f"Resuming from latest valid checkpoint {latest}")
        return str(latest)
    logger.info(f"No valid checkpoints found under {run_dir}; starting a fresh run.")
    return None


def resolve_export_source(
    run_dir: str | None = None,
    adapter_path: str | None = None,
    *,
    allow_unsafe: bool = False,
) -> str:
    if adapter_path:
        safe_adapter_path = str(resolve_existing_path(adapter_path, must_be_dir=True))
        if not is_adapter_artifact(safe_adapter_path, allow_unsafe=allow_unsafe):
            raise ValueError(f"Adapter path is not a valid adapter artifact: {safe_adapter_path}")
        return safe_adapter_path

    if not run_dir:
        raise ValueError("Either run_dir or adapter_path must be provided.")

    safe_run_dir = str(resolve_existing_path(run_dir, must_be_dir=True))
    run_adapter_present = is_adapter_artifact(safe_run_dir, allow_unsafe=allow_unsafe)
    checkpoints: Sequence[Path] = ()
    if not run_adapter_present:
        checkpoints = list_valid_checkpoints(safe_run_dir, allow_unsafe=allow_unsafe)
    return resolve_export_source_from_snapshot(
        safe_run_dir,
        checkpoints,
        run_adapter_present=run_adapter_present,
    )


def resolve_export_source_from_snapshot(
    run_dir: str,
    checkpoints: Sequence[Path],
    *,
    run_adapter_present: bool,
) -> str:
    """Apply export precedence to one already-collected run snapshot."""
    if run_adapter_present:
        return run_dir
    if checkpoints:
        return str(max(checkpoints, key=checkpoint_step))

    raise ValueError(f"No exportable adapter artifact found under {run_dir}")
