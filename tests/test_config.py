from agoge_forger.config import ExperimentConfig, load_config

SMOKE_REVISION = "d3040b7c81a0a810fa13c6f392f3e304a0e121d5"
MINICPM5_REVISION = "156170697656c48f69915b33a2fb44110242187c"
GRANITE_REVISION = "dacb9cb9157bec98e99b09f285c92a4d58405c96"


def test_load_smoke_config():
    config = load_config("configs/smoke_test.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.model_id == "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    assert config.revision == SMOKE_REVISION
    assert config.trust_remote_code is False
    assert config.training.batch_size == 1
    assert config.quantization.load_in_4bit is False


def test_load_minicpm5_canary_pins_hub_revision_and_requires_target_discovery():
    config = load_config("configs/minicpm5_canary.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.model_id == "openbmb/MiniCPM5-1B-Base"
    assert config.revision == MINICPM5_REVISION
    assert config.trust_remote_code is False
    assert config.lora.target_modules == []
    assert config.lora.target_modules_mode == "discover_required"


def test_revision_is_optional_and_defaults_to_none(tmp_path):
    (tmp_path / "data.jsonl").write_text("{}\n")
    config_path = tmp_path / "no_revision.yaml"
    config_path.write_text('model_id: "test-model"\ndataset_path: "data.jsonl"\n')

    config = load_config(str(config_path))
    assert config.revision is None


def test_load_granite_flagship_pins_hub_revision_and_requires_target_discovery():
    config = load_config("configs/granite_4_1_flagship.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.model_id == "ibm-granite/granite-4.1-3b-base"
    assert config.revision == GRANITE_REVISION
    assert config.trust_remote_code is False
    assert config.lora.target_modules == []
    assert config.lora.target_modules_mode == "discover_required"


def test_load_config_parses_explicit_revision(tmp_path):
    (tmp_path / "data.jsonl").write_text("{}\n")
    config_path = tmp_path / "pinned.yaml"
    config_path.write_text(
        'model_id: "org/model"\n'
        "dataset_path: data.jsonl\n"
        "revision: abcdef0123456789abcdef0123456789abcdef01\n"
    )

    config = load_config(str(config_path))
    assert config.revision == "abcdef0123456789abcdef0123456789abcdef01"


def test_load_config_coerces_unquoted_numeric_revision(tmp_path):
    (tmp_path / "data.jsonl").write_text("{}\n")
    config_path = tmp_path / "numeric_revision.yaml"
    config_path.write_text('model_id: "org/model"\ndataset_path: data.jsonl\nrevision: 123456\n')

    config = load_config(str(config_path))
    assert config.revision == "123456"


def test_load_config_normalizes_empty_revision_to_none(tmp_path):
    (tmp_path / "data.jsonl").write_text("{}\n")
    empty = tmp_path / "empty_revision.yaml"
    empty.write_text('model_id: "org/model"\ndataset_path: data.jsonl\nrevision: ""\n')
    assert load_config(str(empty)).revision is None

    whitespace = tmp_path / "whitespace_revision.yaml"
    whitespace.write_text('model_id: "org/model"\ndataset_path: data.jsonl\nrevision: "   "\n')
    assert load_config(str(whitespace)).revision is None


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
        'model_id: "test-model"\ndataset_path: "datasets/samples/tiny_sft.jsonl"\nsave_steps: 12\nsave_total_limit: 3\nresume_from_latest_checkpoint: true\nresume_checkpoint_path: "adapters/run/checkpoint-12"\ndisk_free_warning_gb: 9\ncheckpoint_disk_buffer_gb: 4'
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
