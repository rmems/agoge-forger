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
    dropout = config.lora_dropout
    ranks_usable = _positive_rank(config.r) and _pattern_usable(
        config.rank_pattern or {}, _positive_rank
    )
    dropout_usable = (
        isinstance(dropout, (int, float)) and not isinstance(dropout, bool) and 0 <= dropout <= 1
    )
    alphas_usable = _finite_number(config.lora_alpha) and _pattern_usable(
        config.alpha_pattern or {}, _finite_number
    )
    return bool(ranks_usable and dropout_usable and alphas_usable)


def load_lora_config(payload: dict[str, Any]) -> LoraConfig | None:
    """Build a usable LoRA config from an untrusted JSON object."""
    if payload.get("peft_type") != "LORA":
        return None
    try:
        config = LoraConfig(**payload)
        return config if _config_usable(config) else None
    except (AssertionError, AttributeError, ImportError, KeyError, TypeError, ValueError):
        return None
