"""Lightweight file validation for operator-facing run readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.pytorch_utils import Conv1D

from ._run_status_architecture import architecture_resources_bounded
from ._run_status_artifact_index import artifact_index_usable
from ._run_status_lora import (
    BaseModuleDimensions,
    load_lora_config,
    lora_config_usable,
    lora_shapes_usable,
)
from ._run_status_safetensors import has_complete_merged_weights, safetensors_shapes
from ._run_status_torch_archive import torch_mapping
from .config import normalize_revision

PathLike = str | Path


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (RecursionError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _local_causal_lm_config(candidate: Path) -> Any:
    payload = _load_json_object(candidate / "config.json")
    if payload is None:
        return None
    model_type = payload.get("model_type")
    if not isinstance(model_type, str) or model_type not in CONFIG_MAPPING:
        return None
    try:
        config = CONFIG_MAPPING[model_type].from_dict(payload)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None
    return config if type(config) in MODEL_FOR_CAUSAL_LM_MAPPING else None


def _causal_lm_shapes(config: Any) -> dict[str, tuple[int, ...]] | None:
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


def _offline_pretrained(
    factory: Any,
    source: str | Path,
    *,
    revision: str | None = None,
) -> Any:
    loader = getattr(factory, "from_pretrained", None)
    if not callable(loader):
        raise TypeError("from_pretrained is not callable")
    revision_kwarg = {} if revision is None else {"revision": revision}
    return loader(
        source,
        **revision_kwarg,
        local_files_only=True,
        trust_remote_code=False,
    )


def _tokenizer_usable(candidate: Path) -> bool:
    try:
        tokenizer = _offline_pretrained(AutoTokenizer, candidate)  # nosec B615
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return len(tokenizer) > 0


def _adapter_base_config(config: Any) -> Any:
    base_model = config.base_model_name_or_path
    if not isinstance(base_model, str) or not base_model.strip():
        return None
    try:
        revision = normalize_revision(config.revision)
        base_config = _offline_pretrained(  # nosec B615
            AutoConfig,
            base_model,
            revision=revision,
        )
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return base_config if type(base_config) in MODEL_FOR_CAUSAL_LM_MAPPING else None


def _base_module_dimensions(config: Any) -> dict[str, BaseModuleDimensions] | None:
    if not architecture_resources_bounded(config):
        return None
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        dimensions = {}
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            direct_orientation = isinstance(module, (Conv1D, nn.Embedding))
            input_axis, output_axis = (0, 1) if direct_orientation else (1, 0)
            dimensions[name] = BaseModuleDimensions(
                input_size=weight.shape[input_axis],
                output_size=weight.shape[output_axis],
                embedding=isinstance(module, nn.Embedding),
            )
        return dimensions
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _revision_usable(payload: dict[str, Any]) -> bool:
    if "revision" not in payload or payload["revision"] is None:
        return True
    try:
        return normalize_revision(payload["revision"]) is not None
    except TypeError:
        return False


def is_merged_model_dir(path: PathLike) -> bool:
    """True for a complete, indexed merged-model export."""
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    if not artifact_index_usable(candidate):
        return False
    if not _tokenizer_usable(candidate):
        return False
    config = _local_causal_lm_config(candidate)
    expected_shapes = None if config is None else _causal_lm_shapes(config)
    return expected_shapes is not None and has_complete_merged_weights(candidate, expected_shapes)


def adapter_config_usable(adapter_path: PathLike | None) -> bool:
    """True for an Agoge LoRA config usable by the default export flow."""
    if adapter_path is None:
        return False
    payload = _load_json_object(Path(adapter_path) / "adapter_config.json")
    if payload is None:
        return False
    base = payload.get("base_model_name_or_path")
    return bool(
        _nonempty_string(base)
        and payload.get("peft_type") == "LORA"
        and _revision_usable(payload)
        and lora_config_usable(payload)
    )


def _adapter_lora_config(adapter_dir: Path) -> Any:
    payload = _load_json_object(adapter_dir / "adapter_config.json")
    return None if payload is None else load_lora_config(payload)


def _legacy_lora_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    payload = torch_mapping(path, require_data_record=True)
    return {
        key: tuple(value.shape)
        for key, value in (payload or {}).items()
        if isinstance(value, torch.Tensor)
    }


def _adapter_validation_context(
    adapter_path: PathLike | None,
) -> tuple[Path, Any, dict[str, BaseModuleDimensions]] | None:
    if adapter_path is None:
        return None
    adapter_dir = Path(adapter_path)
    config = _adapter_lora_config(adapter_dir)
    if config is None:
        return None
    base_config = _adapter_base_config(config)
    base_modules = None if base_config is None else _base_module_dimensions(base_config)
    if base_modules is None:
        return None
    return adapter_dir, config, base_modules


def _serialized_lora_shapes(
    adapter_dir: Path, allow_unsafe: bool
) -> dict[str, tuple[int, ...]] | None:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.is_file():
        return safetensors_shapes(safetensors_path)
    legacy = adapter_dir / "adapter_model.bin"
    if allow_unsafe and legacy.is_file():
        return _legacy_lora_shapes(legacy)
    return None


def adapter_weight_shapes(
    adapter_path: PathLike | None,
    *,
    allow_unsafe: bool = False,
) -> dict[str, tuple[int, ...]] | None:
    """Return a complete, config-compatible adapter tensor inventory."""
    context = _adapter_validation_context(adapter_path)
    if context is None:
        return None
    adapter_dir, config, base_modules = context
    shapes = _serialized_lora_shapes(adapter_dir, allow_unsafe)
    if shapes is None:
        return None
    if lora_shapes_usable(shapes, config, base_modules):
        return shapes
    return None


def adapter_weights_usable(
    adapter_path: PathLike | None,
    *,
    allow_unsafe: bool = False,
) -> bool:
    """Validate safetensors, or safely inspect explicitly opted-in legacy weights."""
    return adapter_weight_shapes(adapter_path, allow_unsafe=allow_unsafe) is not None
