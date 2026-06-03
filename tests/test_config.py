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


def test_config_loads_checkpoint_controls(tmp_path):
    config_path = tmp_path / "checkpoint_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'model_id: "test-model"',
                'dataset_path: "datasets/samples/tiny_sft.jsonl"',
                "save_steps: 12",
                "save_total_limit: 3",
                "resume_from_latest_checkpoint: true",
                'resume_checkpoint_path: "adapters/run/checkpoint-12"',
                "disk_free_warning_gb: 9",
                "checkpoint_disk_buffer_gb: 4",
            ]
        )
    )

    config = load_config(str(config_path))

    assert config.training.save_steps == 12
    assert config.training.save_total_limit == 3
    assert config.training.resume_from_latest_checkpoint is True
    assert config.training.resume_checkpoint_path == "adapters/run/checkpoint-12"
    assert config.runtime.disk_free_warning_gb == 9
    assert config.runtime.checkpoint_disk_buffer_gb == 4
