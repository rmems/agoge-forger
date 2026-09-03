"""Local PEFT LoRA configuration and parameter-name validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from peft import LoraConfig

from agoge_forger._run_status_lora_config import load_lora_config
from agoge_forger._run_status_target_pattern import (
    SafeTargetPattern,
    parse_safe_target_pattern,
)

_PAIRS = (
    ("lora_A", "lora_B", ".weight"),
    ("lora_embedding_A", "lora_embedding_B", ""),
)
_PAIR_SEGMENTS = frozenset(segment for pair in _PAIRS for segment in pair[:2])
_BASE_MODEL_PREFIX = "base_model.model."
Shape = tuple[int, ...]


@dataclass(frozen=True)
class BaseModuleDimensions:
    """PEFT-facing dimensions and adapter kind for a targeted base module."""

    input_size: int
    output_size: int
    embedding: bool


TargetSpec = SafeTargetPattern | frozenset[str]


def lora_config_usable(payload: dict[str, Any]) -> bool:
    return load_lora_config(payload) is not None


def _module_rank(config: LoraConfig, module: str) -> int:
    pattern = config.rank_pattern or {}
    pattern_key = next(
        (key for key in pattern if module == key or module.endswith(f".{key}")),
        None,
    )
    return config.r if pattern_key is None else pattern[pattern_key]


def _pair_for_key(key: str) -> tuple[str, str, str] | None:
    for left, right, suffix in _PAIRS:
        for segment, counterpart_segment in ((left, right), (right, left)):
            marker = f".{segment}{suffix}"
            if key.endswith(marker):
                module = key[: -len(marker)]
                if module:
                    counterpart = f"{module}.{counterpart_segment}{suffix}"
                    return module, segment, counterpart
    return None


def _expected_pair_shapes(
    base: BaseModuleDimensions | None, rank: int, embedding: bool
) -> tuple[Shape, Shape] | None:
    if base is None or embedding != base.embedding:
        return None
    return (rank, base.input_size), (base.output_size, rank)


def _regex_target_spec(target: str) -> SafeTargetPattern | None:
    return parse_safe_target_pattern(target)


def _literal_target_spec(targets: Any) -> frozenset[str] | None:
    literal_types = (frozenset, list, set, tuple)
    if not isinstance(targets, literal_types) or not targets:
        return None
    if not all(isinstance(target, str) and bool(target) for target in targets):
        return None
    return frozenset(targets)


def _target_spec(targets: Any) -> TargetSpec | None:
    return (
        _regex_target_spec(targets) if isinstance(targets, str) else _literal_target_spec(targets)
    )


def _module_matches_targets(module: str, targets: TargetSpec) -> bool:
    normalized = module.removeprefix(_BASE_MODEL_PREFIX)
    if isinstance(targets, SafeTargetPattern):
        return targets.fullmatch(normalized)
    candidates = {module, normalized}
    return any(
        candidate == target or candidate.endswith(f".{target}")
        for candidate in candidates
        for target in targets
    )


def _targets_cover_modules(targets: TargetSpec, modules: set[str]) -> bool:
    if not all(_module_matches_targets(module, targets) for module in modules):
        return False
    if isinstance(targets, SafeTargetPattern):
        return True
    return all(
        any(_module_matches_targets(module, frozenset({target})) for module in modules)
        for target in targets
    )


def _recognized_pairs(
    shapes: dict[str, tuple[int, ...]],
) -> dict[str, tuple[str, str, str]] | None:
    pairs = {key: _pair_for_key(key) for key in shapes}
    recognized = {key: pair for key, pair in pairs.items() if pair is not None}
    serialized_pair_keys = {key for key in shapes if not _PAIR_SEGMENTS.isdisjoint(key.split("."))}
    return recognized if recognized.keys() == serialized_pair_keys else None


def _pair_set_complete(pairs: dict[str, tuple[str, str, str]]) -> bool:
    return all(counterpart in pairs for _, _, counterpart in pairs.values())


def _left_pair_shapes_usable(
    shapes: dict[str, tuple[int, ...]],
    pairs: dict[str, tuple[str, str, str]],
    config: LoraConfig,
    base_modules: dict[str, BaseModuleDimensions],
) -> bool:
    right_segments = {"lora_B", "lora_embedding_B"}
    for key, (module, segment, counterpart) in pairs.items():
        if segment in right_segments:
            continue
        base_module = module.removeprefix(_BASE_MODEL_PREFIX)
        expected = _expected_pair_shapes(
            base_modules.get(base_module),
            _module_rank(config, module),
            segment == "lora_embedding_A",
        )
        if expected is None or (shapes[key], shapes[counterpart]) != expected:
            return False
    return True


def _left_pair_modules(pairs: dict[str, tuple[str, str, str]]) -> set[str]:
    right_segments = {"lora_B", "lora_embedding_B"}
    return {module for module, segment, _ in pairs.values() if segment not in right_segments}


def _dora_shapes_usable(
    shapes: dict[str, Shape],
    modules: set[str],
    config: LoraConfig,
    base_modules: dict[str, BaseModuleDimensions],
) -> bool:
    actual = {
        key: shape for key, shape in shapes.items() if "lora_magnitude_vector" in key.split(".")
    }
    if not config.use_dora:
        return not actual
    expected = {}
    for module in modules:
        base = base_modules.get(module.removeprefix(_BASE_MODEL_PREFIX))
        if base is None:
            return False
        expected[f"{module}.lora_magnitude_vector"] = (base.output_size,)
    return actual == expected


def _optional_embedding_base_shapes(
    modules: set[str],
    base_modules: dict[str, BaseModuleDimensions],
) -> dict[str, Shape]:
    optional: dict[str, Shape] = {}
    for module in modules:
        base = base_modules.get(module.removeprefix(_BASE_MODEL_PREFIX))
        if base is not None and base.embedding:
            optional[f"{module}.base_layer.weight"] = (base.input_size, base.output_size)
    return optional


def _inventory_usable(
    shapes: dict[str, Shape],
    pairs: dict[str, tuple[str, str, str]],
    modules: set[str],
    config: LoraConfig,
    base_modules: dict[str, BaseModuleDimensions],
) -> bool:
    required = set(pairs)
    if config.use_dora:
        required.update(f"{module}.lora_magnitude_vector" for module in modules)
    optional = _optional_embedding_base_shapes(modules, base_modules)
    extras = set(shapes) - required
    return extras <= optional.keys() and all(shapes[key] == optional[key] for key in extras)


def lora_shapes_usable(
    shapes: dict[str, tuple[int, ...]] | None,
    config: LoraConfig,
    base_modules: dict[str, BaseModuleDimensions],
) -> bool:
    if not shapes:
        return False
    targets = _target_spec(config.target_modules)
    if targets is None:
        return False
    pairs = _recognized_pairs(shapes)
    if not pairs or not _pair_set_complete(pairs):
        return False
    modules = _left_pair_modules(pairs)
    normalized_modules = {module.removeprefix(_BASE_MODEL_PREFIX) for module in modules}
    expected_modules = {
        module for module in base_modules if _module_matches_targets(module, targets)
    }
    checks = (
        _targets_cover_modules(targets, modules),
        normalized_modules == expected_modules,
        _left_pair_shapes_usable(shapes, pairs, config, base_modules),
        _dora_shapes_usable(shapes, modules, config, base_modules),
        _inventory_usable(shapes, pairs, modules, config, base_modules),
    )
    return all(checks)
