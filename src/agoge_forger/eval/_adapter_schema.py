"""Offline tensor-schema validation for PEFT adapters."""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Collection
from typing import Any

from ._artifact_schema import _ADAPTER_WEIGHTS_PATH, ArtifactValidationContext, IndexedArtifacts
from ._tensor_schema import read_verified_tensor_schema


def require_adapter_tensor_schema(
    indexed: IndexedArtifacts,
    adapter_config: dict[str, object],
    context: ArtifactValidationContext,
) -> None:
    actual = read_verified_tensor_schema(indexed, {_ADAPTER_WEIGHTS_PATH})
    expected = expected_adapter_tensor_schema(adapter_config, context)
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    wrong_shapes = sorted(
        key for key in expected.keys() & actual.keys() if expected[key] != actual[key]
    )
    if any((missing, unexpected, wrong_shapes)):
        raise ValueError(
            "PEFT adapter tensor schema does not match adapter_config.json and base model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"wrong_shapes={wrong_shapes[:5]}"
        )


def expected_adapter_tensor_schema(
    adapter_config: dict[str, object],
    context: ArtifactValidationContext,
) -> dict[str, tuple[int, ...]]:
    try:
        from accelerate import init_empty_weights
        from peft import LoraConfig, get_peft_model
        from peft.utils import get_peft_model_state_dict
        from transformers import AutoModelForCausalLM

        if adapter_config.get("peft_type") != "LORA":
            raise ValueError("only LORA adapters are supported")
        if adapter_config.get("task_type") != "CAUSAL_LM":
            raise ValueError("only CAUSAL_LM adapters are supported")
        if adapter_config.get("peft_version") != importlib.metadata.version("peft"):
            raise ValueError("adapter PEFT version does not match the validator runtime")
        config = load_base_config(context.model_repository, context.model_revision)
        config_values: Any = adapter_config
        lora_config = LoraConfig(**config_values)
        with init_empty_weights(include_buffers=True):
            base = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
            base.name_or_path = context.model_repository
            adapter = get_peft_model(base, lora_config, low_cpu_mem_usage=True)
        state = get_peft_model_state_dict(
            adapter,
            adapter_name="default",
            save_embedding_layers=saves_embedding_layers(lora_config),
        )
        return {name: tuple(tensor.shape) for name, tensor in state.items()}
    except (ImportError, OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            "PEFT adapter config cannot resolve a local, remote-code-disabled base schema"
        ) from exc


def load_base_config(repository: str, revision: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(  # nosec B615 - immutable revision is required below.
        repository,
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
    )


def saves_embedding_layers(config: Any) -> bool:
    if getattr(config, "trainable_token_indices", None) is not None:
        return False
    target_modules: Collection[str] | str | None = config.target_modules
    if isinstance(target_modules, str):
        return any(re.fullmatch(target_modules, name) for name in ("embed_tokens", "lm_head"))
    return bool(target_modules and {"embed_tokens", "lm_head"}.intersection(target_modules))
