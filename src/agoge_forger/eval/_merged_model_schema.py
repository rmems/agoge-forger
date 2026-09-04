"""Offline tensor-schema validation for merged model artifacts."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from ._artifact_schema import IndexedArtifacts
from ._tensor_schema import (
    TensorSchemaEntry,
    read_verified_tensor_schema,
    require_matching_tensor_schema,
    torch_tensor_schema_entry,
)


def require_merged_tensor_schema(
    indexed: IndexedArtifacts,
    model_weights: set[PurePosixPath],
    config: dict[str, object],
) -> None:
    actual = read_verified_tensor_schema(indexed, model_weights)
    expected = _expected_merged_tensor_schema(config)
    require_matching_tensor_schema(
        actual,
        expected,
        label="merged-model tensor schema does not match config.json",
    )


def _expected_merged_tensor_schema(
    config_payload: dict[str, object],
) -> dict[str, TensorSchemaEntry]:
    model = _empty_causal_lm(config_payload)
    expected = {
        name: torch_tensor_schema_entry(tensor) for name, tensor in model.state_dict().items()
    }
    drop_omitted_model_keys(expected, model)
    return expected


def _empty_causal_lm(config_payload: dict[str, object]) -> Any:
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        values = dict(config_payload)
        model_type = values.pop("model_type")
        if not isinstance(model_type, str):
            raise TypeError("model_type must be a string")
        config = AutoConfig.for_model(model_type, **values)
        model_kwargs = {"dtype": config.dtype} if config.dtype is not None else {}
        with init_empty_weights(include_buffers=True):
            return AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=False,
                **model_kwargs,
            )
    except (ImportError, OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            "merged-model config cannot resolve a local, remote-code-disabled causal LM schema"
        ) from exc


def drop_omitted_model_keys(expected: dict[str, TensorSchemaEntry], model: object) -> None:
    tied = getattr(model, "all_tied_weights_keys", None)
    if isinstance(tied, dict):
        for omitted_alias in tied:
            expected.pop(omitted_alias, None)
    ignored = getattr(model, "_keys_to_ignore_on_save", None) or ()
    for ignored_name in ignored:
        expected.pop(ignored_name, None)
