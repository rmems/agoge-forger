"""Conservative resource limits for untrusted model architecture configs."""

from __future__ import annotations

from typing import Any

_MAX_ARCHITECTURE_MULTIPLICITY = 4_096
_MAX_ARCHITECTURE_DIMENSION = 16_777_216
_MAX_ARCHITECTURE_MODULES = 65_536
_MULTIPLICITY_TERMS = ("layer", "block", "expert", "stage", "head")
_MULTIPLICITY_SUFFIXES = tuple(
    suffix for term in _MULTIPLICITY_TERMS for suffix in (f"_{term}", f"_{term}s")
)
_MULTIPLICITY_KEYS = {"depth", "depths", "n_head", "n_heads", "n_layer", "n_layers"}


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


def architecture_resources_bounded(config: Any) -> bool:
    """Return whether a model config is conservatively safe to instantiate on meta."""
    try:
        payload = config.to_dict()
    except (AttributeError, TypeError, ValueError):
        return False
    return isinstance(payload, dict) and _architecture_value_bounded("", payload)
