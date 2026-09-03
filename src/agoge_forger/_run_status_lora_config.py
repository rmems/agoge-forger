"""Fail-closed validation for local PEFT LoRA configuration."""

from __future__ import annotations

import math
import re
from typing import Any

from peft import LoraConfig

_MODULE_PATTERN_RE = re.compile(r"\w+(?:\.\w+)*", re.ASCII)


def _positive_rank(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _pattern_usable(pattern: Any, value_validator: Any) -> bool:
    return bool(
        isinstance(pattern, dict)
        and all(_MODULE_PATTERN_RE.fullmatch(key) for key in pattern)
        and all(value_validator(value) for value in pattern.values())
    )


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _config_usable(config: LoraConfig) -> bool:
    return all(
        (
            _ranks_usable(config),
            _dropout_usable(config),
            _alphas_usable(config),
            _standard_structure_usable(config),
        )
    )


def _empty_optional(value: Any) -> bool:
    return value is None or value == [] or value == {} or value == set()


def _standard_structure_usable(config: LoraConfig) -> bool:
    optional_fields = (
        "auto_mapping",
        "exclude_modules",
        "modules_to_save",
        "layers_to_transform",
        "layers_pattern",
        "megatron_config",
        "trainable_token_indices",
        "loftq_config",
        "eva_config",
        "corda_config",
        "lora_ga_config",
        "alora_invocation_tokens",
        "layer_replication",
        "target_parameters",
        "arrow_config",
    )
    no_optional_structure = all(
        _empty_optional(getattr(config, field, None)) for field in optional_fields
    )
    supported_flags = all(
        (
            config.bias == "none",
            config.init_lora_weights is True,
            isinstance(config.fan_in_fan_out, bool),
            isinstance(config.inference_mode, bool),
            isinstance(config.use_dora, bool),
            config.use_rslora is False,
            config.use_qalora is False,
            config.use_bdlora is None,
            config.lora_bias is False,
            config.ensure_weight_tying is False,
            not config.runtime_config.ephemeral_gpu_offload,
        )
    )
    return no_optional_structure and supported_flags


def _ranks_usable(config: LoraConfig) -> bool:
    return _positive_rank(config.r) and _pattern_usable(config.rank_pattern or {}, _positive_rank)


def _dropout_usable(config: LoraConfig) -> bool:
    dropout = config.lora_dropout
    return bool(
        isinstance(dropout, (int, float)) and not isinstance(dropout, bool) and 0 <= dropout <= 1
    )


def _alphas_usable(config: LoraConfig) -> bool:
    return _finite_number(config.lora_alpha) and _pattern_usable(
        config.alpha_pattern or {}, _finite_number
    )


def load_lora_config(payload: dict[str, Any]) -> LoraConfig | None:
    """Build a usable LoRA config from an untrusted JSON object."""
    if payload.get("peft_type") != "LORA":
        return None
    try:
        config = LoraConfig(**payload)
        return config if _config_usable(config) else None
    except (AssertionError, AttributeError, ImportError, KeyError, TypeError, ValueError):
        return None
