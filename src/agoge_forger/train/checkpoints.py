import json
import os
import re
from pathlib import Path
from typing import Optional

from ..logging import logger

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return -1
    return int(match.group(1))


def is_adapter_artifact(path: str) -> bool:
    adapter_dir = Path(path)
    if not adapter_dir.is_dir():
        return False
    return (adapter_dir / "adapter_config.json").exists() and any(
        (adapter_dir / weight_file).exists() for weight_file in ADAPTER_WEIGHT_FILES
    )


def is_valid_checkpoint(path: str) -> bool:
    checkpoint_dir = Path(path)
    if _checkpoint_step(checkpoint_dir) < 0:
        return False
    if not (checkpoint_dir / "trainer_state.json").exists():
        return False
    return is_adapter_artifact(str(checkpoint_dir))


def list_valid_checkpoints(run_dir: str) -> list[str]:
    root = Path(run_dir)
    if not root.exists():
        return []

    checkpoints = [path for path in root.iterdir() if is_valid_checkpoint(str(path))]
    checkpoints.sort(key=_checkpoint_step)
    return [str(path) for path in checkpoints]


def find_latest_valid_checkpoint(run_dir: str) -> Optional[str]:
    checkpoints = list_valid_checkpoints(run_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]


def infer_base_model_from_adapter(adapter_path: str) -> str:
    config_path = Path(adapter_path) / "adapter_config.json"
    with config_path.open() as handle:
        adapter_config = json.load(handle)

    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"base_model_name_or_path not found in {config_path}")
    return base_model


def resolve_resume_checkpoint(run_dir: str, config) -> Optional[str]:
    if config.training.resume_checkpoint_path:
        checkpoint_path = config.training.resume_checkpoint_path
        if not is_valid_checkpoint(checkpoint_path):
            raise ValueError(f"Configured resume checkpoint is not valid: {checkpoint_path}")
        logger.info(f"Resuming from explicit checkpoint {checkpoint_path}")
        return checkpoint_path

    if not config.training.resume_from_latest_checkpoint:
        return None

    checkpoint_path = find_latest_valid_checkpoint(run_dir)
    if checkpoint_path:
        logger.info(f"Resuming from latest valid checkpoint {checkpoint_path}")
    else:
        logger.info(f"No valid checkpoints found under {run_dir}; starting a fresh run.")
    return checkpoint_path


def resolve_export_source(run_dir: Optional[str] = None, adapter_path: Optional[str] = None) -> str:
    if adapter_path:
        if not is_adapter_artifact(adapter_path):
            raise ValueError(f"Adapter path is not a valid adapter artifact: {adapter_path}")
        return adapter_path

    if not run_dir:
        raise ValueError("Either run_dir or adapter_path must be provided.")

    checkpoint_path = find_latest_valid_checkpoint(run_dir)
    if checkpoint_path:
        return checkpoint_path

    if is_adapter_artifact(run_dir):
        return run_dir

    raise ValueError(f"No exportable adapter artifact found under {run_dir}")
