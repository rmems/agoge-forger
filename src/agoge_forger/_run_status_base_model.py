"""Bounded local base-model inventory checks for run-status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM

from ._run_status_architecture import architecture_resources_bounded
from ._run_status_hub_cache import cached_snapshot, cached_weights_usable
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


def local_base_weights_usable(
    base_model: str,
    config: Any,
    revision: str | None = None,
) -> bool:
    """Require exact safe weights for a local path or cached Hub snapshot."""
    candidate = Path(base_model)
    expected_shapes = causal_lm_shapes(config)
    if expected_shapes is None:
        return False
    if candidate.is_dir():
        return has_complete_merged_weights(candidate, expected_shapes)
    snapshot = cached_snapshot(base_model, revision, HF_HUB_CACHE)
    return snapshot is not None and cached_weights_usable(snapshot, expected_shapes)
