import pytest
from agoge_forger.config import load_config, ExperimentConfig

def test_load_smoke_config():
    config = load_config("configs/smoke_test.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.model_id == "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    assert config.training.batch_size == 1
    assert config.quantization.load_in_4bit == False
