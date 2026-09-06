"""Restricted RNG checkpoint validation used by run-status."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._run_status_torch_archive import torch_mapping


def _cuda_tensor_usable(value: Any) -> bool:
    # Transformers 5.12 Trainer._save_rng_state uses
    # torch.cuda.random.get_rng_state() for a single-process checkpoint. Current
    # PyTorch packs its CUDA Philox seed and offset into 16 bytes; 8 bytes is the
    # supported legacy seed-only state accepted by Generator.set_state().
    if not isinstance(value, torch.Tensor):
        return False
    return all(
        (
            value.layout == torch.strided,
            value.device.type == "cpu",
            value.dtype == torch.uint8,
            value.is_contiguous(),
            value.numel() in {8, 16},
        )
    )


def _cuda_rng_state_usable(value: Any) -> bool:
    if not _cuda_tensor_usable(value):
        return False
    if value.numel() == 8:
        return True
    offset = int.from_bytes(bytes(value[8:16].tolist()), byteorder=sys.byteorder, signed=True)
    return offset >= 0 and offset % 4 == 0


def _rng_payload_usable(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    try:
        random.Random().setstate(payload["python"])  # nosec B311 - validates saved RNG state
        np.random.RandomState().set_state(payload["numpy"])
        torch.Generator(device="cpu").set_state(payload["cpu"])
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    return _cuda_rng_state_usable(payload.get("cuda"))


def _rng_state_usable(checkpoint: Path) -> bool:
    """Return whether a single-process Transformers RNG checkpoint is usable."""
    single = checkpoint / "rng_state.pth"
    if single.is_file():
        return _rng_payload_usable(
            torch_mapping(single, allow_numpy=True, require_data_record=True)
        )
    # Transformers does not persist the original world size in safe JSON.
    # Rank filenames cannot prove that trailing states are present, so report
    # distributed checkpoints as not ready instead of returning a false positive.
    return False
