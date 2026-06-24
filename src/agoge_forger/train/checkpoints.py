import json
import re
from pathlib import Path
from typing import Optional, Union

from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins
from ..logging import logger
from ..path_safety import resolve_existing_path

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors",)
LEGACY_ADAPTER_WEIGHT_FILES = ("adapter_model.bin",)

PathLike = Union[str, Path]


def _checkpoint_step(path: Path) -> int:
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

    weight_files = ADAPTER_WEIGHT_FILES
    if allow_unsafe:
        weight_files = ADAPTER_WEIGHT_FILES + LEGACY_ADAPTER_WEIGHT_FILES

    if not any((adapter_dir / weight_file).is_file() for weight_file in weight_files):
        return False
    if not allow_unsafe and _has_unsafe_weight_bins(adapter_dir):
        return False
    return True


def is_valid_checkpoint(path: PathLike, *, allow_unsafe: bool = False) -> bool:
    """A checkpoint is valid iff it is a `checkpoint-N` directory with the
    required trainer state, AND it is a valid adapter artifact. Unsafe-bin
    filtering happens here so callers selecting among checkpoints never have
    to re-validate unless ``allow_unsafe`` is explicitly enabled.
    """
    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir():
        return False
    if _checkpoint_step(checkpoint_dir) < 0:
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
    checkpoints.sort(key=_checkpoint_step)
    return checkpoints


def find_latest_valid_checkpoint(run_dir: PathLike, *, allow_unsafe: bool = False) -> Optional[Path]:
    checkpoints = list_valid_checkpoints(run_dir, allow_unsafe=allow_unsafe)
    if not checkpoints:
        return None
    return checkpoints[-1]


def infer_base_model_from_adapter(adapter_path: PathLike) -> str:
    adapter_dir = Path(adapter_path)
    config_path = adapter_dir / "adapter_config.json"
    with config_path.open() as handle:
        adapter_config = json.load(handle)

    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"base_model_name_or_path not found in {config_path}")
    return base_model


def resolve_resume_checkpoint(run_dir: str, config) -> Optional[str]:
    if config.training.resume_checkpoint_path:
        checkpoint_path = str(
            resolve_existing_path(config.training.resume_checkpoint_path, must_be_dir=True)
        )
        if not is_valid_checkpoint(checkpoint_path):
            raise ValueError(f"Configured resume checkpoint is not valid: {checkpoint_path}")
        # is_valid_checkpoint already enforces safetensors-only.
        logger.info(f"Resuming from explicit checkpoint {checkpoint_path}")
        return checkpoint_path

    if not config.training.resume_from_latest_checkpoint:
        return None

    # find_latest_valid_checkpoint only returns safetensors-clean
    # checkpoints, so no further unsafe-bin scan is needed.
    checkpoint_path = find_latest_valid_checkpoint(run_dir)
    if checkpoint_path:
        logger.info(f"Resuming from latest valid checkpoint {checkpoint_path}")
    else:
        logger.info(f"No valid checkpoints found under {run_dir}; starting a fresh run.")
    return str(checkpoint_path) if checkpoint_path else None


def resolve_export_source(
    run_dir: Optional[str] = None,
    adapter_path: Optional[str] = None,
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
    if is_adapter_artifact(safe_run_dir, allow_unsafe=allow_unsafe):
        return safe_run_dir

    checkpoint_path = find_latest_valid_checkpoint(safe_run_dir, allow_unsafe=allow_unsafe)
    if checkpoint_path:
        return str(checkpoint_path)

    raise ValueError(f"No exportable adapter artifact found under {safe_run_dir}")
