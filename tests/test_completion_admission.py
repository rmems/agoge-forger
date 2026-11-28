"""Exercise offline admission through the installed console entry point."""

import json
import os
import subprocess  # nosec B404 - exercises the installed CLI with synthetic fixtures.
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from agoge_forger.train.completion import completion_tokens


@pytest.fixture
def admission_tokenizer(tmp_path):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "Prompt": 2, "café": 3}, "[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(  # nosec B106 - vocabulary markers, not credentials.
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    )
    directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(directory)
    return directory, tokenizer


def run_admission(source, tokenizer_dir, output, budget=3, completion_format="frozen-text-v1"):
    return subprocess.run(  # nosec B603 - fixed installed entry point, no shell.
        [
            str(Path(sys.executable).with_name("agoge")),
            "admit-completions",
            "--source",
            str(source),
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--output-dir",
            str(output),
            "--max-seq-length",
            str(budget),
            "--completion-format",
            completion_format,
        ],
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_entry_point_exact_budget_histogram_and_reproducibility(tmp_path, admission_tokenizer):
    directory, tokenizer = admission_tokenizer
    rows = [
        {
            "canonical_id": "short",
            "lineage_id": "one",
            "text": "Prompt café",
            "completion_start_char": 7,
        },
        {
            "canonical_id": "long",
            "lineage_id": "two",
            "text": "Prompt café café",
            "completion_start_char": 7,
        },
        {
            "canonical_id": "eos",
            "lineage_id": "three",
            "text": "Prompt café [EOS]",
            "completion_start_char": 7,
        },
    ]
    source = tmp_path / "source.jsonl"
    write_rows(source, rows)
    before = source.read_bytes()
    output = tmp_path / "admitted"
    result = run_admission(source, directory, output)
    assert result.returncode == 0, result.stderr
    admitted = [json.loads(line) for line in (output / "admitted.jsonl").read_text().splitlines()]
    assert admitted == [rows[0], rows[2]]
    report = json.loads((output / "admission_report.json").read_text())
    assert report["token_length_histogram"] == {"3": 2, "4": 1}
    assert (report["source_rows"], report["admitted_rows"], report["over_budget_rows"]) == (3, 2, 1)
    assert all(len(completion_tokens(row, tokenizer, 3)["input_ids"]) == 3 for row in admitted)
    assert source.read_bytes() == before
    second = tmp_path / "again"
    assert run_admission(source, directory, second).returncode == 0
    for name in ("admitted.jsonl", "decisions.jsonl", "admission_report.json"):
        assert (second / name).read_bytes() == (output / name).read_bytes()
    assert run_admission(source, directory, output).returncode == 1
    assert (output / "admitted.jsonl").read_text().count("\n") == 2


@pytest.mark.parametrize(
    "row",
    [
        {"text": "Prompt café", "completion_start_char": True},
        {"text": "Prompt café", "completion_start_char": 2},
        {"text": "Prompt café", "completion_start_char": 7, "labels": []},
        {"text": "Prompt ", "completion_start_char": 7},
    ],
)
def test_invalid_rows_abort_without_publishing(tmp_path, admission_tokenizer, row):
    directory, _ = admission_tokenizer
    source = tmp_path / "source.jsonl"
    write_rows(source, [{"text": "Prompt café", "completion_start_char": 7}, row])
    output = tmp_path / "admitted"
    result = run_admission(source, directory, output)
    assert result.returncode == 1
    assert not output.exists()
    assert "line 2" in result.stdout + result.stderr


def test_all_over_budget_is_a_valid_empty_admission(tmp_path, admission_tokenizer):
    directory, _ = admission_tokenizer
    source = tmp_path / "source.jsonl"
    write_rows(source, [{"text": "Prompt café café", "completion_start_char": 7}])
    output = tmp_path / "admitted"
    assert run_admission(source, directory, output, budget=2).returncode == 0
    assert (output / "admitted.jsonl").read_text() == ""
    assert json.loads((output / "admission_report.json").read_text())["over_budget_rows"] == 1


def test_empty_source_and_missing_tokenizer_fail(tmp_path, admission_tokenizer):
    directory, _ = admission_tokenizer
    source = tmp_path / "source.jsonl"
    source.write_text("\n")
    output = tmp_path / "admitted"
    assert run_admission(source, directory, output).returncode == 1
    assert not output.exists()
    assert run_admission(source, tmp_path / "missing", output).returncode == 1


def test_source_metadata_bytes_survive_admission(tmp_path, admission_tokenizer):
    directory, _ = admission_tokenizer
    raw = b'{"text":"Prompt caf\\u00e9","completion_start_char":7,"timestamp":1700000000.123456789}'
    source = tmp_path / "source.jsonl"
    source.write_bytes(raw)
    output = tmp_path / "admitted"
    assert run_admission(source, directory, output).returncode == 0
    assert (output / "admitted.jsonl").read_bytes() == raw + b"\n"


def test_admission_uses_training_pad_fallback(tmp_path, admission_tokenizer):
    from agoge_forger.models.tokenizer_setup import prepare_training_tokenizer

    directory, tokenizer = admission_tokenizer
    tokenizer.eos_token = tokenizer.unk_token
    tokenizer.save_pretrained(directory)
    row = {"text": "Prompt unseen", "completion_start_char": 7}
    with pytest.raises(ValueError, match="padding special token"):
        completion_tokens(row, prepare_training_tokenizer(tokenizer), 10)
    source = tmp_path / "source.jsonl"
    write_rows(source, [row])
    output = tmp_path / "admitted"
    result = run_admission(source, directory, output, 10)
    assert result.returncode == 1
    assert "special token" in result.stdout + result.stderr
    assert not output.exists()


def test_changed_source_aborts_publication(tmp_path, admission_tokenizer, monkeypatch):
    from agoge_forger import completion_admission

    directory, _ = admission_tokenizer
    source = tmp_path / "source.jsonl"
    write_rows(source, [{"text": "Prompt café", "completion_start_char": 7}])
    output = tmp_path / "admitted"
    original = completion_admission._admit_rows

    def change_after_read(*args):
        report = original(*args)
        source.write_text("changed during admission\n")
        return report

    monkeypatch.setattr(completion_admission, "_admit_rows", change_after_read)
    with pytest.raises(ValueError, match="source changed"):
        completion_admission.admit_completions(str(source), str(directory), str(output), 3)
    assert not output.exists()
