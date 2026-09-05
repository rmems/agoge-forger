"""Offline tensor-schema validation for PEFT adapters."""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any

from ._artifact_schema import _ADAPTER_WEIGHTS_PATH, ArtifactValidationContext, IndexedArtifacts
from ._tensor_schema import (
    TensorSchemaEntry,
    read_verified_tensor_schema,
    require_matching_tensor_schema,
    torch_tensor_schema_entry,
)


def require_adapter_tensor_schema(
    indexed: IndexedArtifacts,
    adapter_config: dict[str, object],
    context: ArtifactValidationContext,
) -> None:
    actual = read_verified_tensor_schema(indexed, {_ADAPTER_WEIGHTS_PATH})
    expected = expected_adapter_tensor_schema(adapter_config, context)
    require_matching_tensor_schema(
        actual,
        expected,
        label="PEFT adapter tensor schema does not match adapter_config.json and base model",
    )


def expected_adapter_tensor_schema(
    adapter_config: dict[str, object],
    context: ArtifactValidationContext,
) -> dict[str, TensorSchemaEntry]:
    try:
        lora_config = _validated_lora_config(adapter_config)
        base_config = load_base_config(context.model_repository, context.model_revision)
    except (ImportError, OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            "PEFT adapter config cannot resolve a local, remote-code-disabled base schema"
        ) from exc
    try:
        adapter = _empty_adapter(base_config, lora_config, context.model_repository)
        return _saved_adapter_schema(adapter, lora_config)
    except (ImportError, OSError, ValueError, KeyError) as exc:
        raise ValueError(
            "PEFT adapter config cannot resolve a local, remote-code-disabled base schema"
        ) from exc


def _validated_lora_config(adapter_config: dict[str, object]) -> Any:
    from peft import LoraConfig

    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("only LORA adapters are supported")
    if adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("only CAUSAL_LM adapters are supported")
    config_values: Any = adapter_config
    return LoraConfig(**config_values)


def _empty_adapter(base_config: Any, lora_config: Any, repository: str) -> Any:
    from accelerate import init_empty_weights
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    with init_empty_weights(include_buffers=True):
        base = AutoModelForCausalLM.from_config(base_config, trust_remote_code=False)
        base.name_or_path = repository
        return get_peft_model(base, lora_config, low_cpu_mem_usage=True)


def _saved_adapter_schema(adapter: Any, lora_config: Any) -> dict[str, TensorSchemaEntry]:
    from peft.utils import get_peft_model_state_dict

    state = get_peft_model_state_dict(
        adapter,
        adapter_name="default",
        save_embedding_layers=saves_embedding_layers(lora_config),
    )
    return {name: torch_tensor_schema_entry(tensor) for name, tensor in state.items()}


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
        try:
            pattern = re.compile(target_modules)
        except re.error as exc:
            raise ValueError(f"invalid target_modules regex: {target_modules!r}") from exc
        return any(pattern.fullmatch(name) for name in ("embed_tokens", "lm_head"))
    return bool(target_modules and {"embed_tokens", "lm_head"}.intersection(target_modules))
