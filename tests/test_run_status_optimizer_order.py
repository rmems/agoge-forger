"""Producer-backed optimizer-order reconstruction tests."""

import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from test_run_status import TINY_LLAMA_CONFIG
from transformers import AutoModelForCausalLM, GPT2Config, LlamaConfig

from agoge_forger._run_status_optimizer_order import ordered_trainable_shapes


@pytest.mark.parametrize(
    ("base_config", "lora_config", "expected"),
    [
        (
            LlamaConfig(**TINY_LLAMA_CONFIG),
            LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
            [[(2, 8), (8, 2)], []],
        ),
        (
            LlamaConfig(**TINY_LLAMA_CONFIG),
            LoraConfig(
                r=2,
                target_modules=["q_proj"],
                task_type="CAUSAL_LM",
                use_dora=True,
            ),
            [[(2, 8), (8, 2), (8,)], []],
        ),
        (
            LlamaConfig(**TINY_LLAMA_CONFIG),
            LoraConfig(r=2, target_modules=["embed_tokens"], task_type="CAUSAL_LM"),
            [[(2, 16), (8, 2)], []],
        ),
        (
            GPT2Config(
                n_layer=1,
                n_head=2,
                n_embd=8,
                n_positions=16,
                n_ctx=16,
                vocab_size=16,
            ),
            LoraConfig(
                r=2,
                target_modules=["c_attn"],
                task_type="CAUSAL_LM",
                fan_in_fan_out=True,
            ),
            [[(2, 8), (24, 2)], []],
        ),
    ],
    ids=["linear", "dora", "embedding", "conv1d"],
)
def test_mapping_preserves_genuine_peft_variants_without_hub_probe(
    monkeypatch, base_config, lora_config, expected
):
    with torch.device("meta"):
        model = get_peft_model(
            AutoModelForCausalLM.from_config(base_config),
            lora_config,
        )
    serialized = get_peft_model_state_dict(model, save_embedding_layers=False)
    shapes = {name: tuple(value.shape) for name, value in serialized.items()}
    base_config._name_or_path = "cached-org/remote-base"
    lora_config.base_model_name_or_path = "cached-org/remote-base"

    def deny_hub_probe(*args, **kwargs):
        raise AssertionError("optimizer mapping attempted a Hugging Face Hub probe")

    monkeypatch.setattr(
        "peft.utils.save_and_load.check_file_exists_on_hf_hub",
        deny_hub_probe,
    )

    assert ordered_trainable_shapes(shapes, lora_config, base_config) == expected
