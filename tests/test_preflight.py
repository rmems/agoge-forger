import pytest
from agoge_forger.config import ExperimentConfig
from agoge_forger.train.preflight import collect_disk_pressure_report, validate_lora_targets_exist


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
    with pytest.raises(
        ValueError,
        match="target_modules_mode is discover_required but no target_modules were provided",
    ):
        validate_lora_targets_exist(model, cfg)

    # auto common works
    cfg = DummyConfig([], "auto_common")
    valid = validate_lora_targets_exist(model, cfg)
    assert set(valid) == {"q_proj", "v_proj"}


def test_collect_disk_pressure_report_uses_monitored_paths(tmp_path):
    hot_path = tmp_path / "huggingface"
    hot_path.mkdir()
    payload = hot_path / "blob.bin"
    payload.write_bytes(b"x" * 32)

    config = ExperimentConfig(
        model_id="test-model", dataset_path="dataset.jsonl", output_dir=str(tmp_path)
    )
    report = collect_disk_pressure_report(config, monitored_paths=[str(hot_path)])

    assert report["output_dir"] == str(tmp_path)
    assert report["paths"][0]["path"] == str(hot_path)
    assert report["paths"][0]["exists"] is True
    assert report["paths"][0]["size_gb"] > 0


def test_collect_disk_pressure_report_handles_fresh_output_dir(tmp_path):
    fresh_output = tmp_path / "new-run" / "adapter"
    config = ExperimentConfig(
        model_id="test-model",
        dataset_path="dataset.jsonl",
        output_dir=str(fresh_output),
    )

    report = collect_disk_pressure_report(config, monitored_paths=[])

    assert report["output_dir"] == str(fresh_output)
    assert report["free_gb"] > 0
