"""Exact serialized LoRA tensor-inventory tests."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig

from agoge_forger._run_status_validation import adapter_optimizer_shapes
from agoge_forger.run_status import build_run_status
from tests.test_run_status import TINY_LLAMA_CONFIG, _make_run_dir, _safetensors_with_shapes
from tests.test_run_status_adapters import _lora_shapes_ready, _write_base_weights


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


@pytest.mark.parametrize("legacy", [False, True], ids=["safetensors", "legacy-bin"])
def test_lora_keys_without_peft_prefix_are_rejected(tmp_path, legacy):
    # PEFT loads with strict=False into base_model.model.* parameters, so unprefixed
    # keys are silently dropped and the merge would be a no-op.
    module = "model.layers.0.self_attn.q_proj"
    shapes = {
        f"{module}.lora_A.weight": (1, 8),
        f"{module}.lora_B.weight": (8, 1),
    }

    assert _lora_shapes_ready(tmp_path, shapes, legacy=legacy) is False


@pytest.mark.parametrize("legacy", [False, True], ids=["safetensors", "legacy-bin"])
def test_unprefixed_lora_copy_beside_prefixed_pair_is_rejected(tmp_path, legacy):
    module = "model.layers.0.self_attn.q_proj"
    shapes = {}
    for prefix in ("base_model.model.", ""):
        shapes[f"{prefix}{module}.lora_A.weight"] = (1, 8)
        shapes[f"{prefix}{module}.lora_B.weight"] = (8, 1)

    assert _lora_shapes_ready(tmp_path, shapes, legacy=legacy) is False


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_peft_saved_lm_head_adapter_is_ready(tmp_path, tied):
    base_config = LlamaConfig(**{**TINY_LLAMA_CONFIG, "tie_word_embeddings": tied})
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "config.json").write_text(base_config.to_json_string())
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_config(base_config, trust_remote_code=False)
    _write_base_weights(base_model, base)
    peft_model = get_peft_model(
        base,
        LoraConfig(r=2, target_modules=["q_proj", "lm_head"], task_type="CAUSAL_LM"),
    )
    peft_model.peft_config["default"].base_model_name_or_path = str(base_model)
    run_dir = _make_run_dir(tmp_path)
    peft_model.save_pretrained(run_dir, safe_serialization=True)
    lm_head_base = "base_model.model.lm_head.base_layer.weight"
    with safe_open(run_dir / "adapter_model.safetensors", framework="pt") as handle:
        assert lm_head_base in set(handle.keys())

    assert build_run_status(str(run_dir))["export"]["ready"] is True
    assert adapter_optimizer_shapes(run_dir) is not None

    # The allowance is the exact (out_features, in_features) lm_head weight shape.
    shapes = {
        key: tuple(value.shape)
        for key, value in load_file(run_dir / "adapter_model.safetensors").items()
    }
    shapes[lm_head_base] = (8, 16)
    (run_dir / "adapter_model.safetensors").write_bytes(_safetensors_with_shapes(shapes))
    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_base_layer_weight_is_not_allowed_for_ordinary_projections(tmp_path):
    module = "base_model.model.model.layers.0.self_attn.q_proj"
    shapes = {
        f"{module}.lora_A.weight": (1, 8),
        f"{module}.lora_B.weight": (8, 1),
        f"{module}.base_layer.weight": (8, 8),
    }

    assert _lora_shapes_ready(tmp_path, shapes) is False
