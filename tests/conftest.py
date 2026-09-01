import pytest
from transformers import LlamaConfig

from agoge_forger.eval import _adapter_schema


@pytest.fixture
def cached_test_base_config(monkeypatch):
    """Provide the immutable tiny base config used by evaluation fixtures."""

    def load_base_config(repository, revision):
        assert repository == "example/base-model"
        assert revision == "abcdef0123456789abcdef0123456789abcdef01"
        return LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )

    monkeypatch.setattr(_adapter_schema, "load_base_config", load_base_config)
