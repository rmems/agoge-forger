import json
import re
from pathlib import Path
from typing import Optional

from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins
from ..logging import logger
from ..path_safety import resolve_existing_path

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors",)


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return -1
    return int(match.group(1))


def is_adapter_artifact(path: str) -> bool:
    adapter_dir = resolve_existing_path(path, must_be_dir=True)
    return (adapter_dir / "adapter_config.json").exists() and any(
        (adapter_dir / weight_file).exists() for weight_file in ADAPTER_WEIGHT_FILES
    )


def is_valid_checkpoint(path: str) -> bool:
    checkpoint_dir = resolve_existing_path(path, must_be_dir=True)
    if _checkpoint_step(checkpoint_dir) < 0:
        return False
    if not (checkpoint_dir / "trainer_state.json").exists():
        return False
    return is_adapter_artifact(str(checkpoint_dir))


def list_valid_checkpoints(run_dir: str) -> list[str]:
    root = resolve_existing_path(run_dir, must_be_dir=True)

    checkpoints = [
        path for path in root.iterdir() if path.is_dir() and is_valid_checkpoint(str(path))
    ]
    checkpoints.sort(key=_checkpoint_step)
    return [str(path) for path in checkpoints]


def find_latest_valid_checkpoint(run_dir: str) -> Optional[str]:
    checkpoints = list_valid_checkpoints(run_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]


def infer_base_model_from_adapter(adapter_path: str) -> str:
    adapter_dir = resolve_existing_path(adapter_path, must_be_dir=True)
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
        assert_no_unsafe_weight_bins(checkpoint_path)
        logger.info(f"Resuming from explicit checkpoint {checkpoint_path}")
        return checkpoint_path

    if not config.training.resume_from_latest_checkpoint:
        return None

    checkpoint_path = find_latest_valid_checkpoint(run_dir)
    if checkpoint_path:
        assert_no_unsafe_weight_bins(checkpoint_path)
        logger.info(f"Resuming from latest valid checkpoint {checkpoint_path}")
    else:
        logger.info(f"No valid checkpoints found under {run_dir}; starting a fresh run.")
    return checkpoint_path


def resolve_export_source(run_dir: Optional[str] = None, adapter_path: Optional[str] = None) -> str:
    if adapter_path:
        safe_adapter_path = str(resolve_existing_path(adapter_path, must_be_dir=True))
        if not is_adapter_artifact(safe_adapter_path):
            raise ValueError(f"Adapter path is not a valid adapter artifact: {safe_adapter_path}")
        assert_no_unsafe_weight_bins(safe_adapter_path)
        return safe_adapter_path

    if not run_dir:
        raise ValueError("Either run_dir or adapter_path must be provided.")

    safe_run_dir = str(resolve_existing_path(run_dir, must_be_dir=True))
    if is_adapter_artifact(safe_run_dir):
        assert_no_unsafe_weight_bins(safe_run_dir)
        return safe_run_dir

    checkpoint_path = find_latest_valid_checkpoint(safe_run_dir)
    if checkpoint_path:
        assert_no_unsafe_weight_bins(checkpoint_path)
        return checkpoint_path

    raise ValueError(f"No exportable adapter artifact found under {safe_run_dir}")
