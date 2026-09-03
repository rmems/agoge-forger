"""Bounded local base-model inventory checks for run-status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from ._run_status_architecture import architecture_resources_bounded
from ._run_status_safetensors import has_complete_merged_weights


def causal_lm_shapes(config: Any) -> dict[str, tuple[int, ...]] | None:
    """Build the expected checkpoint inventory without allocating model tensors."""
    if not architecture_resources_bounded(config):
        return None
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        ignored = set(getattr(model, "_keys_to_ignore_on_save", None) or ())
        tied = set(model.all_tied_weights_keys or {})
        return {
            name: tuple(value.shape)
            for name, value in model.state_dict().items()
            if name not in ignored and name not in tied
        }
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def local_base_weights_usable(base_model: str, config: Any) -> bool:
    """Require exact safe weights for a path that export would load locally."""
    candidate = Path(base_model)
    if not candidate.is_dir():
        return True
    expected_shapes = causal_lm_shapes(config)
    return expected_shapes is not None and has_complete_merged_weights(candidate, expected_shapes)
