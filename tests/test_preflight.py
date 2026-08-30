import json

import pytest

from agoge_forger.config import ExperimentConfig, QuantizationConfig, TrainingConfig
from agoge_forger.datasets import load_jsonl_dataset
from agoge_forger.train.preflight import (
    BYTES_PER_GB,
    collect_disk_pressure_report,
    estimate_training_risk,
    get_gpu_report,
    validate_dataset_text_field,
    validate_dataset_text_field_in_source,
    validate_lora_targets_exist,
)


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


def test_dataset_text_field_validation_rejects_a_column_the_dataset_lacks():
    with pytest.raises(ValueError, match="dataset_text_field 'body' is not a column"):
        validate_dataset_text_field(["text"], "body")


def test_dataset_text_field_validation_accepts_a_present_column():
    assert validate_dataset_text_field(["text", "body"], "body") == "body"


@pytest.mark.parametrize(
    "row",
    [
        {"text": "User: hi\nAssistant: hello"},
        {"messages": [{"role": "user", "content": "hi"}]},
        {"instruction": "Define PyTorch.", "output": "A tensor and autograd library."},
    ],
    ids=["plain_text", "messages", "instruction"],
)
def test_production_loader_always_yields_the_text_column(row, tmp_path):
    """Guard the loader contract the preflight check depends on.

    `normalize_row` collapses all three accepted row formats onto `text`, which
    is why any other `dataset_text_field` cannot resolve. Build the fixture
    through the real loader so this stays honest if normalization changes.
    """
    dataset_path = tmp_path / "rows.jsonl"
    dataset_path.write_text(json.dumps(row) + "\n")

    dataset = load_jsonl_dataset(str(dataset_path))

    assert "text" in dataset.column_names
    assert validate_dataset_text_field(dataset.column_names, "text") == "text"
    with pytest.raises(ValueError, match="Row normalization always produces 'text'"):
        validate_dataset_text_field(dataset.column_names, "body")


@pytest.mark.parametrize(
    "row",
    [
        {"text": "User: hi\nAssistant: hello"},
        {"messages": [{"role": "user", "content": "hi"}]},
        {"instruction": "Define PyTorch.", "output": "A tensor and autograd library."},
    ],
    ids=["plain_text", "messages", "instruction"],
)
def test_source_scan_agrees_with_the_loader_without_a_tokenizer(row, tmp_path):
    """The pre-model-load scan must accept exactly what the real loader accepts.

    `run_training` rejects a bad `dataset_text_field` from this scan *before*
    `load_base_model` runs, so a scan that disagreed with the loader would
    either abort valid runs or fail to spare an invalid one the GPU cost.
    """
    dataset_path = tmp_path / "rows.jsonl"
    dataset_path.write_text(json.dumps(row) + "\n")

    assert "text" in load_jsonl_dataset(str(dataset_path)).column_names
    assert validate_dataset_text_field_in_source(str(dataset_path), "text") == "text"


@pytest.mark.parametrize(
    ("second_row", "reason"),
    [
        ({"text": "b"}, "is missing"),
        ({"text": "b", "body": None}, "is missing"),
        ({"text": "b", "body": 7}, r"is not a string \(got int\)"),
    ],
    ids=["key_omitted", "explicit_null", "wrong_type"],
)
def test_source_scan_rejects_a_field_absent_from_a_later_row(second_row, reason, tmp_path):
    """A first-row-only check is not enough, and both holes cost a model load.

    With `dataset_text_field="body"`, a file whose first row carries `body` but
    whose second does not fails in two distinct ways downstream — Arrow raises
    `DatasetGenerationError` when the key is omitted, and TRL tokenizes a
    `None` when it is explicitly null. Both land after the model is resident.
    """
    dataset_path = tmp_path / "mixed.jsonl"
    dataset_path.write_text(
        json.dumps({"text": "a", "body": "a"}) + "\n" + json.dumps(second_row) + "\n"
    )

    # The name-only check cannot see it: row 1 supplies the column.
    assert validate_dataset_text_field(["text", "body"], "body") == "body"

    with pytest.raises(ValueError, match=f"Line 2: .*'body' {reason}"):
        validate_dataset_text_field_in_source(str(dataset_path), "body")


def test_source_scan_skips_blank_lines_and_rejects_an_empty_dataset(tmp_path):
    padded = tmp_path / "padded.jsonl"
    padded.write_text('\n\n{"text": "hello"}\n')
    assert validate_dataset_text_field_in_source(str(padded), "text") == "text"

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(ValueError, match="contains no rows"):
        validate_dataset_text_field_in_source(str(empty), "text")


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


def test_get_gpu_report_uses_binary_gib_not_decimal_gb(monkeypatch):
    """16 GiB cards must report ~16.0 so the <=16.5 risk gate can fire.

    Dividing by 1e9 (decimal) would inflate the same card to ~17.18 and skip
    all local OOM warnings on the RTX 5080 path.
    """
    import torch

    sixteen_gib = 16 * BYTES_PER_GB

    class _Props:
        total_memory = sixteen_gib

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: "Fake RTX 5080")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _idx: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _idx: 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    report = get_gpu_report()

    assert report["total_vram_gb"] == pytest.approx(16.0)
    assert report["allocated_vram_gb"] == pytest.approx(0.0)
    # Regression lock: decimal SI would be ~17.18
    assert report["total_vram_gb"] != pytest.approx(sixteen_gib / 1e9, abs=0.01)


def test_estimate_training_risk_warns_on_16gib_risky_config(caplog):
    import logging

    config = ExperimentConfig(
        model_id="test-model",
        dataset_path="dataset.jsonl",
        quantization=QuantizationConfig(load_in_4bit=False),
        training=TrainingConfig(batch_size=2, max_seq_length=4096),
    )
    gpu_report = {"total_vram_gb": 16.0}

    with caplog.at_level(logging.WARNING):
        estimate_training_risk(config, gpu_report)

    text = "\n".join(caplog.messages)
    assert "without load_in_4bit" in text
    assert "Batch size > 1" in text
    assert "max_seq_length > 2048" in text


def test_estimate_training_risk_skips_when_vram_above_gate(caplog):
    import logging

    config = ExperimentConfig(
        model_id="test-model",
        dataset_path="dataset.jsonl",
        quantization=QuantizationConfig(load_in_4bit=False),
        training=TrainingConfig(batch_size=2, max_seq_length=4096),
    )
    # Decimal-bug fake reading (~17.18) would skip the gate; binary 24 GiB
    # correctly stays silent.
    with caplog.at_level(logging.WARNING):
        estimate_training_risk(config, {"total_vram_gb": 24.0})

    assert not any("RISK:" in m for m in caplog.messages)
