"""Non-executing Trainer-state integrity checks used by run-status."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ._run_status_torch_archive import torch_zip_metadata

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def _json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _trainer_state_matches_checkpoint(checkpoint: Path) -> bool:
    payload = _json_object(checkpoint / "trainer_state.json")
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if payload is None or match is None:
        return False
    global_step = payload.get("global_step")
    train_batch_size = payload.get("train_batch_size")
    return bool(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and global_step == int(match.group(1))
        and isinstance(train_batch_size, int)
        and not isinstance(train_batch_size, bool)
        and train_batch_size > 0
    )


def _rng_state_usable(checkpoint: Path) -> bool:
    single = checkpoint / "rng_state.pth"
    if single.is_file():
        return _torch_state_usable(single, {"python", "numpy", "cpu"}, require_data_record=True)
    # Transformers does not persist the original world size in safe JSON.
    # Rank filenames cannot prove that trailing states are present, so report
    # distributed checkpoints as not ready instead of returning a false positive.
    return False


def _torch_state_usable(
    path: Path,
    required_fields: set[str],
    *,
    require_data_record: bool = False,
) -> bool:
    fields = torch_zip_metadata(path, require_data_record=require_data_record)
    return fields is not None and required_fields.issubset(fields)


def trainer_state_usable(checkpoint: str | Path | None) -> bool:
    if checkpoint is None:
        return False
    checkpoint_dir = Path(checkpoint)
    return bool(
        _trainer_state_matches_checkpoint(checkpoint_dir)
        and _torch_state_usable(checkpoint_dir / "optimizer.pt", {"state", "param_groups"})
        and _torch_state_usable(checkpoint_dir / "scheduler.pt", {"last_epoch", "_step_count"})
        and _rng_state_usable(checkpoint_dir)
    )
