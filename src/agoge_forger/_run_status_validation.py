"""Lightweight file validation for operator-facing run readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING, AutoModelForCausalLM

from ._run_status_artifact_index import artifact_index_usable
from ._run_status_lora import load_lora_config, lora_config_usable, lora_shapes_usable
from ._run_status_safetensors import has_complete_merged_weights, safetensors_shapes
from ._run_status_torch_archive import torch_mapping
from .config import normalize_revision

PathLike = str | Path

_MAX_ARCHITECTURE_MULTIPLICITY = 4_096
_MAX_ARCHITECTURE_DIMENSION = 16_777_216
_MAX_ARCHITECTURE_MODULES = 65_536
_MULTIPLICITY_TERMS = ("layer", "block", "expert", "stage", "head")
_MULTIPLICITY_SUFFIXES = tuple(
    suffix for term in _MULTIPLICITY_TERMS for suffix in (f"_{term}", f"_{term}s")
)
_MULTIPLICITY_KEYS = {"depth", "depths", "n_head", "n_heads", "n_layer", "n_layers"}


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
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


def _multiplicity_key(key: str) -> bool:
    prefixed = key.startswith("num_") and any(term in key for term in _MULTIPLICITY_TERMS)
    return any((key in _MULTIPLICITY_KEYS, key.endswith(_MULTIPLICITY_SUFFIXES), prefixed))


def _integer_resource_bounded(key: str, value: int) -> bool:
    if value > _MAX_ARCHITECTURE_DIMENSION:
        return False
    if _multiplicity_key(key):
        return value <= _MAX_ARCHITECTURE_MULTIPLICITY
    return True


def _direct_multiplicity_entry(key: Any, value: Any, terms: tuple[str, ...]) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if value <= 0:
        return False
    key_text = str(key)
    if not _multiplicity_key(key_text):
        return False
    return any(term in key_text for term in terms)


def _direct_multiplicity(value: dict[Any, Any], terms: tuple[str, ...]) -> int:
    counts = [
        child for key, child in value.items() if _direct_multiplicity_entry(key, child, terms)
    ]
    return max(counts, default=1)


def _combined_module_count_bounded(value: dict[Any, Any]) -> bool:
    structural = _direct_multiplicity(value, ("layer", "block", "stage", "depth"))
    experts = _direct_multiplicity(value, ("expert",))
    return structural * experts <= _MAX_ARCHITECTURE_MODULES


def _mapping_resources_bounded(value: dict[Any, Any]) -> bool:
    return (
        len(value) <= _MAX_ARCHITECTURE_MULTIPLICITY
        and _combined_module_count_bounded(value)
        and all(
            _architecture_value_bounded(str(child_key), child) for child_key, child in value.items()
        )
    )


def _sequence_resources_bounded(key: str, value: list[Any] | tuple[Any, ...]) -> bool:
    return len(value) <= _MAX_ARCHITECTURE_MULTIPLICITY and all(
        _architecture_value_bounded(key, child) for child in value
    )


def _architecture_value_bounded(key: str, value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, int):
        return _integer_resource_bounded(key, value)
    if isinstance(value, dict):
        return _mapping_resources_bounded(value)
    if isinstance(value, (list, tuple)):
        return _sequence_resources_bounded(key, value)
    return True


def _architecture_resources_bounded(config: Any) -> bool:
    try:
        payload = config.to_dict()
    except (AttributeError, TypeError, ValueError):
        return False
    return isinstance(payload, dict) and _architecture_value_bounded("", payload)


def _causal_lm_shapes(config: Any) -> dict[str, tuple[int, ...]] | None:
    if not _architecture_resources_bounded(config):
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
    if not candidate.is_dir() or not artifact_index_usable(candidate):
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
    payload = torch_mapping(path)
    return {
        key: tuple(value.shape)
        for key, value in (payload or {}).items()
        if isinstance(value, torch.Tensor)
    }


def adapter_weight_shapes(
    adapter_path: PathLike | None,
    *,
    allow_unsafe: bool = False,
) -> dict[str, tuple[int, ...]] | None:
    """Return a complete, config-compatible adapter tensor inventory."""
    if adapter_path is None:
        return None
    adapter_dir = Path(adapter_path)
    config = _adapter_lora_config(adapter_dir)
    if config is None:
        return None
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.is_file():
        shapes = safetensors_shapes(safetensors_path)
        return shapes if lora_shapes_usable(shapes, config) else None
    if allow_unsafe:
        legacy = adapter_dir / "adapter_model.bin"
        if legacy.is_file():
            shapes = _legacy_lora_shapes(legacy)
            return shapes if lora_shapes_usable(shapes, config) else None
    return None


def adapter_weights_usable(
    adapter_path: PathLike | None,
    *,
    allow_unsafe: bool = False,
) -> bool:
    """Validate safetensors, or safely inspect explicitly opted-in legacy weights."""
    return adapter_weight_shapes(adapter_path, allow_unsafe=allow_unsafe) is not None
