import json
from types import SimpleNamespace

import pytest
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM

from agoge_forger.eval import _adapter_schema, _tensor_schema
from agoge_forger.eval._artifact_schema import ArtifactValidationContext


def test_base_config_loader_is_offline_and_revision_pinned(monkeypatch):
    observed = {}

    def from_pretrained(repository, **kwargs):
        observed.update(repository=repository, **kwargs)
        return object()

    monkeypatch.setattr(AutoConfig, "from_pretrained", from_pretrained)

    loaded = _adapter_schema.load_base_config("example/base", "a" * 40)

    assert loaded is not None
    assert observed == {
        "repository": "example/base",
        "revision": "a" * 40,
        "local_files_only": True,
        "trust_remote_code": False,
    }


def test_adapter_schema_fails_closed_when_base_config_is_not_cached(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("not cached")

    monkeypatch.setattr(AutoConfig, "from_pretrained", unavailable)
    config = LoraConfig(
        r=2,
        target_modules=["q_proj"],
        task_type="CAUSAL_LM",
        base_model_name_or_path="example/base",
        revision="a" * 40,
    ).to_dict()
    context = ArtifactValidationContext("peft_adapter", "example/base", "a" * 40)

    with pytest.raises(ValueError, match="local, remote-code-disabled base schema"):
        _adapter_schema.expected_adapter_tensor_schema(config, context)


@pytest.mark.parametrize(
    ("targets", "trainable_tokens", "expected"),
    [
        (r".*(embed_tokens|lm_head)$", None, True),
        ({"q_proj", "v_proj"}, None, False),
        ({"embed_tokens"}, [1, 2], False),
    ],
)
def test_embedding_save_decision_matches_peft_rules(targets, trainable_tokens, expected):
    config = SimpleNamespace(
        target_modules=targets,
        trainable_token_indices=trainable_tokens,
    )

    assert _adapter_schema.saves_embedding_layers(config) is expected


@pytest.mark.parametrize("variant", ["modules-to-save", "tied-embedding"])
def test_expected_adapter_schema_matches_peft_save(tmp_path, monkeypatch, variant):
    tied = variant == "tied-embedding"
    base_config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tied,
    )
    base = LlamaForCausalLM(base_config)
    base.name_or_path = "example/base"
    lora_config = LoraConfig(
        r=2,
        target_modules=["embed_tokens"] if tied else ["q_proj"],
        modules_to_save=None if tied else ["lm_head"],
        ensure_weight_tying=tied,
        task_type="CAUSAL_LM",
        revision="a" * 40,
    )
    adapter = get_peft_model(base, lora_config)
    adapter.save_pretrained(
        tmp_path,
        save_embedding_layers=_adapter_schema.saves_embedding_layers(lora_config),
    )
    adapter_config = json.loads((tmp_path / "adapter_config.json").read_bytes())
    monkeypatch.setattr(
        _adapter_schema,
        "load_base_config",
        lambda repository, revision: base_config,
    )
    context = ArtifactValidationContext("peft_adapter", "example/base", "a" * 40)

    expected = _adapter_schema.expected_adapter_tensor_schema(adapter_config, context)
    with safe_open(tmp_path / "adapter_model.safetensors", framework="pt") as handle:
        tensor_keys = handle.keys()
        actual = {}
        for key in tensor_keys:
            tensor = handle.get_slice(key)
            actual[key] = _tensor_schema.TensorSchemaEntry(
                tuple(tensor.get_shape()), tensor.get_dtype()
            )

    assert expected == actual
