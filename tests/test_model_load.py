"""Exercise loader precision with tiny local weights, without Hub or GPU access."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

from agoge_forger.models.load import load_base_model


@pytest.fixture
def local_model(tmp_path):
    model = AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=3,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model.save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2})),
    )
    tokenizer.add_special_tokens({"unk_token": "[UNK]", "pad_token": "[PAD]"})
    tokenizer.save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize(
    ("obsolete_args", "obsolete_kwargs"),
    [((), {"bf16": False}), ((), {"bf16": True}), ((None, False), {})],
    ids=["keyword-false", "keyword-true", "positional"],
)
def test_loader_rejects_obsolete_bf16(local_model, obsolete_args, obsolete_kwargs):
    with pytest.raises(TypeError):
        load_base_model(
            local_model,
            False,
            *obsolete_args,
            **obsolete_kwargs,
            local_files_only=True,
            device_map="cpu",
        )


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("auto", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
    ],
)
def test_loader_uses_explicit_dtype(local_model, dtype, expected):
    model, tokenizer = load_base_model(
        local_model,
        False,
        torch_dtype_str=dtype,
        local_files_only=True,
        device_map="cpu",
    )
    assert model.dtype == expected
    assert tokenizer.pad_token_id == 1
