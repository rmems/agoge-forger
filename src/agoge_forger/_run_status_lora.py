"""Local PEFT LoRA configuration and parameter-name validation."""

from __future__ import annotations

from typing import Any

from peft import LoraConfig

from agoge_forger._run_status_lora_config import load_lora_config

_PAIRS = (("lora_A", "lora_B"), ("lora_embedding_A", "lora_embedding_B"))


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
    parts = key.split(".")
    for left, right in _PAIRS:
        segment = left if left in parts else None
        if right in parts:
            segment = right
        if segment is None:
            continue
        module = ".".join(parts[: parts.index(segment)])
        counterpart = parts.copy()
        counterpart[counterpart.index(segment)] = right if segment == left else left
        return module, segment, ".".join(counterpart)
    return None


def _pair_shapes_usable(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    rank: int,
) -> bool:
    return bool(
        len(left_shape) == 2
        and len(right_shape) == 2
        and all(dimension > 0 for dimension in left_shape + right_shape)
        and left_shape[0] == right_shape[1] == rank
    )


def _literal_targets_usable(targets: Any) -> bool:
    return bool(
        not isinstance(targets, str)
        and targets
        and all(isinstance(target, str) and bool(target) for target in targets)
    )


def _module_matches_literal_targets(module: str, targets: Any) -> bool:
    candidates = {module, module.removeprefix("base_model.model.")}
    return any(
        candidate == target or candidate.endswith(f".{target}")
        for candidate in candidates
        for target in targets
    )


def _targets_cover_modules(config: LoraConfig, modules: set[str]) -> bool:
    targets = config.target_modules
    if targets is None:
        return True
    if not _literal_targets_usable(targets):
        return False
    return all(_module_matches_literal_targets(module, targets) for module in modules) and all(
        any(_module_matches_literal_targets(module, {target}) for module in modules)
        for target in targets
    )


def _recognized_pairs(shapes: dict[str, tuple[int, ...]]) -> dict[str, tuple[str, str, str]]:
    pairs = {key: _pair_for_key(key) for key in shapes}
    return {key: pair for key, pair in pairs.items() if pair is not None}


def _pair_set_complete(pairs: dict[str, tuple[str, str, str]]) -> bool:
    return all(counterpart in pairs for _, _, counterpart in pairs.values())


def _left_pair_shapes_usable(
    shapes: dict[str, tuple[int, ...]],
    pairs: dict[str, tuple[str, str, str]],
    config: LoraConfig,
) -> bool:
    right_segments = {"lora_B", "lora_embedding_B"}
    for key, (module, segment, counterpart) in pairs.items():
        if segment in right_segments:
            continue
        if not _pair_shapes_usable(shapes[key], shapes[counterpart], _module_rank(config, module)):
            return False
    return True


def _left_pair_modules(pairs: dict[str, tuple[str, str, str]]) -> set[str]:
    right_segments = {"lora_B", "lora_embedding_B"}
    return {module for module, segment, _ in pairs.values() if segment not in right_segments}


def lora_shapes_usable(
    shapes: dict[str, tuple[int, ...]] | None,
    config: LoraConfig,
) -> bool:
    if not shapes:
        return False
    pairs = _recognized_pairs(shapes)
    return bool(
        pairs
        and _pair_set_complete(pairs)
        and _targets_cover_modules(config, _left_pair_modules(pairs))
        and _left_pair_shapes_usable(shapes, pairs, config)
    )
