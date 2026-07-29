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
    # The relative `dataset_path` is resolved against the config file's
    # directory (per the gemini-code-assist fix), so the dataset file
    # must exist relative to tmp_path.
    dataset = tmp_path / "datasets" / "samples" / "tiny_sft.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n")

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


def test_config_resolves_relative_dataset_path_against_config_dir(tmp_path, monkeypatch):
    """Relative `dataset_path` must resolve against the config file's
    directory, not the current working directory. This makes configs
    portable across environments.
    """
    # Place the config in one temp dir and the dataset in a sibling
    # temp dir; the CWD is somewhere else entirely.
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset = data_dir / "my_dataset.jsonl"
    dataset.write_text("{}\n")

    config_path = config_dir / "exp.yaml"
    # Use a path that's relative to the config file's directory.
    config_path.write_text('model_id: "m"\ndataset_path: "../data/my_dataset.jsonl"\n')

    # Run the test from a different CWD to prove the config path
    # resolution is independent of the process working directory.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_config(str(config_path))
    assert config.dataset_path == str(dataset.resolve())
