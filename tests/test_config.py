from agoge_forger.config import load_config, ExperimentConfig
from pathlib import Path


def test_load_nested_training_config(tmp_path: Path):
    cfg_path = tmp_path / "nested.yaml"
    cfg_path.write_text(
        """
model_id: "Qwen/Qwen3.5-0.5B"
dataset_path: "datasets/samples/tiny_sft.jsonl"
output_dir: "adapters"
run_name: "nested_run"
training:
  max_seq_length: 512
  num_epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 0.00002
  warmup_steps: 5
  max_steps: 100
  save_steps: 30
  eval_steps: 0.1
  weight_decay: 0.001
  random_seed: 3407
  packing: false
  train_on_completions: true
  gradient_checkpointing: unsloth
  optim: adamw_8bit
  lr_scheduler_type: linear
lora:
  lora_r: 16
  lora_alpha: 16
  lora_dropout: 0
  target_modules:
    - q_proj
    - v_proj
""".strip()
    )

    config = load_config(str(cfg_path))
    assert isinstance(config, ExperimentConfig)
    assert config.model_id == "Qwen/Qwen3.5-0.5B"
    assert config.training.max_seq_length == 512
    assert config.training.num_train_epochs == 3
    assert config.training.warmup_steps == 5
    assert config.training.max_steps == 100
    assert config.training.train_on_completions is True
    assert config.training.gradient_checkpointing is True
    assert config.training.seed == 3407
    assert config.lora.target_modules == ["q_proj", "v_proj"]

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
