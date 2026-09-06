"""Exact serialized LoRA tensor-inventory tests."""

import pytest

from tests.test_run_status_adapters import _lora_shapes_ready


@pytest.mark.parametrize("legacy", [False, True], ids=["safetensors", "legacy-bin"])
def test_embedding_base_layer_requires_exact_weight_shape(tmp_path, legacy):
    module = "base_model.model.model.embed_tokens"
    shapes = {
        f"{module}.lora_embedding_A": (1, 16),
        f"{module}.lora_embedding_B": (8, 1),
        f"{module}.base_layer.weight": (8, 16),
    }

    assert (
        _lora_shapes_ready(
            tmp_path,
            shapes,
            legacy=legacy,
            target_modules=["embed_tokens"],
        )
        is False
    )


@pytest.mark.parametrize("legacy", [False, True], ids=["safetensors", "legacy-bin"])
def test_foreign_tensor_is_rejected_beside_valid_lora_pair(tmp_path, legacy):
    module = "base_model.model.model.layers.0.self_attn.q_proj"
    shapes = {
        f"{module}.lora_A.weight": (1, 8),
        f"{module}.lora_B.weight": (8, 1),
        "foreign.tensor": (1,),
    }

    assert _lora_shapes_ready(tmp_path, shapes, legacy=legacy) is False
