"""Restricted, memory-mapped validation for current PyTorch ZIP state."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from pickle import UnpicklingError  # nosec B403 - exception type only; no pickle loading
from typing import Any

import torch
from transformers.trainer_pt_utils import safe_globals


def torch_mapping(path: Path, *, allow_numpy: bool = False) -> dict[str, Any] | None:
    """Read safe metadata without executing globals or paging in tensor data."""
    if path.is_symlink() or not path.is_file():
        return None
    try:
        context = safe_globals() if allow_numpy else nullcontext()
        with context:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
    except (UnpicklingError, EOFError, IndexError, KeyError, RuntimeError, ValueError):
        return None
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        return None
    return payload
