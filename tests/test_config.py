from agoge_forger.config import load_config, ExperimentConfig

def test_load_smoke_config():
    config = load_config("configs/smoke_test.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.model_id == "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    assert config.training.batch_size == 1
    assert config.quantization.load_in_4bit is False

def test_config_loads_safetensors_fields():
    config = load_config("configs/smoke_test.yaml")
    assert config.runtime.save_safetensors is True
    assert config.runtime.allow_unsafe_serialization is False
    assert config.runtime.max_shard_size == "4GB"
    assert config.lora.target_modules_mode == "auto_common"
