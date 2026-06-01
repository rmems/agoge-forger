import pytest
from agoge_forger.train.preflight import validate_lora_targets_exist
from agoge_forger.train.preflight import validate_tokenizer_dataset_compatibility


class DummyEmbeddings:
    def __init__(self, num_embeddings):
        self.num_embeddings = num_embeddings


class DummyModel:
    def __init__(self, num_embeddings):
        self._embeddings = DummyEmbeddings(num_embeddings)

    def named_modules(self):
        return [("q_proj", self), ("v_proj", self)]

    def get_input_embeddings(self):
        return self._embeddings


class DummyTokenizer:
    def __len__(self):
        return 10

    def __call__(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": [ord(c) % 10 for c in text]}

class DummyModule:
    def named_modules(self):
        return [("q_proj", self), ("v_proj", self)]

class DummyConfig:
    def __init__(self, targets, mode):
        self.target_modules = targets
        self.target_modules_mode = mode

def test_lora_target_validation_fails_on_missing_targets():
    model = DummyModule()
    
    # explicit failing
    cfg = DummyConfig(["non_existent"], "explicit")
    with pytest.raises(ValueError, match="Explicit target module 'non_existent' does not exist"):
        validate_lora_targets_exist(model, cfg)
        
    # discover required without targets
    cfg = DummyConfig([], "discover_required")
    with pytest.raises(ValueError, match="target_modules_mode is discover_required but no target_modules were provided"):
        validate_lora_targets_exist(model, cfg)
        
    # auto common works
    cfg = DummyConfig([], "auto_common")
    valid = validate_lora_targets_exist(model, cfg)
    assert set(valid) == {"q_proj", "v_proj"}


def test_tokenizer_dataset_compatibility_fails_on_oob_ids():
    model = DummyModel(num_embeddings=5)
    tokenizer = DummyTokenizer()
    dataset = [{"text": "abc"}]

    with pytest.raises(ValueError, match="exceeds model embedding size"):
        validate_tokenizer_dataset_compatibility(model, tokenizer, dataset)
