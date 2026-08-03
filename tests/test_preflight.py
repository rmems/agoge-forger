import json

import pytest

from agoge_forger.config import ExperimentConfig
from agoge_forger.datasets import load_jsonl_dataset
from agoge_forger.train.preflight import (
    collect_disk_pressure_report,
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
        {"instruction": "Define JAX.", "output": "An autograd library."},
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
        {"instruction": "Define JAX.", "output": "An autograd library."},
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
