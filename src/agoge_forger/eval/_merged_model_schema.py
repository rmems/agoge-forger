"""Offline tensor-schema validation for merged model artifacts."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from ._artifact_schema import IndexedArtifacts
from ._tensor_schema import read_verified_tensor_schema


def require_merged_tensor_schema(
    indexed: IndexedArtifacts,
    model_weights: set[PurePosixPath],
    config: dict[str, object],
) -> None:
    actual = read_verified_tensor_schema(indexed, model_weights)
    expected = _expected_merged_tensor_schema(config)
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    wrong_shapes = sorted(
        key for key in expected.keys() & actual.keys() if expected[key] != actual[key]
    )
    if any((missing, unexpected, wrong_shapes)):
        raise ValueError(
            "merged-model tensor schema does not match config.json: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"wrong_shapes={wrong_shapes[:5]}"
        )


def _expected_merged_tensor_schema(
    config_payload: dict[str, object],
) -> dict[str, tuple[int, ...]]:
    model = _empty_causal_lm(config_payload)
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
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
        with init_empty_weights(include_buffers=True):
            return AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    except (ImportError, OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            "merged-model config cannot resolve a local, remote-code-disabled causal LM schema"
        ) from exc


def drop_omitted_model_keys(expected: dict[str, tuple[int, ...]], model: object) -> None:
    tied = getattr(model, "all_tied_weights_keys", None)
    if isinstance(tied, dict):
        for omitted_alias in tied:
            expected.pop(omitted_alias, None)
    ignored = getattr(model, "_keys_to_ignore_on_save", None) or ()
    for ignored_name in ignored:
        expected.pop(ignored_name, None)
