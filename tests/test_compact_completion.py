"""Pure text edits and offline CLI integration for compact targets."""

import json
import sys
from pathlib import Path

import pytest

from agoge_forger.compact_completion import (
    apply_search_replace,
    render_compact_row,
    search_replace_completion,
)
from agoge_forger.train.completion import completion_tokens
from tests import test_completion_admission as admission_cases
from tests.test_completion_admission import run_compact_admission, write_rows

admission_tokenizer = admission_cases.admission_tokenizer


@pytest.mark.parametrize(
    "before,after",
    [
        ("apple\npear\n", "apple\nplum\n"),
        ("apple\n", "pear\napple\n"),
        ("apple\n", "apple\npear\n"),
        ("apple\npear\n", "apple\n"),
        ("apple\n", ""),
        ("café 🍎\r\npear", "café 🍐\r\npear"),
        ("apple", "apples"),
        ("apple\n", "apple"),
        ("a\na\na\na\na\na\na\n", "a\na\na\nb\na\na\na\n"),
        ("first\nkeep\nlast\n", "FIRST\nkeep\nLAST\n"),
    ],
)
def test_exact_round_trip(before, after):
    completion = search_replace_completion(before, after)
    assert apply_search_replace(before, completion) == after
    assert completion == search_replace_completion(before, after)


@pytest.mark.parametrize("before,after", [("", "apple"), ("apple", "apple"), (None, "pear")])
def test_invalid_or_empty_change_is_rejected(before, after):
    with pytest.raises(ValueError):
        search_replace_completion(before, after)


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "{}",
        '{"search":"missing","replace":"pear"}',
        '{"search":"a","replace":"pear"}',
        '{"search":"aa","replace":"pear"}',  # overlapping matches also ambiguous
        '{"search":"","replace":"pear"}',
        '{"search":"aaa","replace":"aaa"}',
        '{"search":"aaa","replace":null}',
        '{"search":"aaa","replace":"pear","path":"fruit.txt"}',
        '{"search":"aaa","search":"aaa","replace":"pear"}',
    ],
)
def test_apply_rejects_empty_mismatched_ambiguous_or_malformed_edits(completion):
    with pytest.raises(ValueError):
        apply_search_replace("aaa", completion)


def test_large_text_has_compact_target_and_unicode_boundary():
    before = "".join(f"item {index}: apple\n" for index in range(100))
    after = before.replace("item 50: apple", "item 50: pear")
    row = render_compact_row(
        {
            "canonical_id": "fruit-50",
            "instruction": "Update fruit 🍐",
            "before": before,
            "after": after,
        }
    )
    prompt, completion = (
        row["text"][: row["completion_start_char"]],
        row["text"][row["completion_start_char"] :],
    )
    assert prompt.endswith("Edit:\n")
    assert completion.startswith('{"search":')
    assert len(completion) < len(after) // 4
    assert apply_search_replace(row["source_text"], completion) == after
    assert row["canonical_id"] == "fruit-50"
    assert "after" not in row


def test_compact_entry_point_counts_rendered_rows(tmp_path, admission_tokenizer):
    directory, tokenizer = admission_tokenizer
    pairs = [
        {
            "canonical_id": "short",
            "lineage_id": "fruit-a",
            "instruction": "Update fruit 🍐",
            "before": "apple\n",
            "after": "pear\n",
        },
        {
            "canonical_id": "long",
            "lineage_id": "fruit-b",
            "instruction": "Update fruit",
            "before": "apple " * 100,
            "after": "pear " * 100,
        },
    ]
    rendered = render_compact_row(pairs[0])
    budget = len(completion_tokens(rendered, tokenizer, 10000)["input_ids"])
    source = tmp_path / "pairs.jsonl"
    write_rows(source, pairs)
    original_bytes = source.read_bytes()
    output = tmp_path / "compact"
    result = run_compact_admission(source, directory, output, budget)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in (output / "admitted.jsonl").read_text().splitlines()]
    assert rows == [rendered]
    assert len(completion_tokens(rows[0], tokenizer, budget)["input_ids"]) == budget
    completion = rows[0]["text"][rows[0]["completion_start_char"] :]
    assert apply_search_replace(rows[0]["source_text"], completion) == pairs[0]["after"]
    report = json.loads((output / "admission_report.json").read_text())
    assert report["completion_format"] == "search-replace-json-v1"
    assert report["admitted_rows"] == report["over_budget_rows"] == 1
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("extra", [{"text": "conflicting"}, {"labels": []}, {"after": "apple"}])
def test_compact_entry_point_fails_closed(tmp_path, admission_tokenizer, extra):
    directory, _ = admission_tokenizer
    source = tmp_path / "pairs.jsonl"
    write_rows(
        source, [{"instruction": "Update fruit", "before": "apple", "after": "pear", **extra}]
    )
    output = tmp_path / "compact"
    result = run_compact_admission(source, directory, output, 1000)
    assert result.returncode == 1
    assert not output.exists()


def test_compact_metadata_precision_is_not_silently_lost():
    from agoge_forger.compact_json import compact_json_line
    from agoge_forger.split_schema import canonical_json_bytes

    raw = (
        b'{ "instruction": "Update fruit", "before":"apple", "after":"pear", '
        b'"timestamp":1700000000.123456789, "canonical\\u005fid":"fruit" }'
    )
    with pytest.raises(ValueError, match="precision-sensitive"):
        compact_json_line(raw, render_compact_row(json.loads(raw)))
    safe = raw.replace(b"1700000000.123456789", b'"1700000000.123456789"')
    row = render_compact_row(json.loads(safe))
    assert compact_json_line(safe, row) == canonical_json_bytes(row) + b"\n"
    assert row["timestamp"] == "1700000000.123456789"


def test_duplicate_context_expansion_is_bounded(monkeypatch):
    import agoge_forger.compact_completion as compact

    original = compact._unique_position
    attempts = []

    def observe(source, search):
        attempts.append(len(search))
        return original(source, search)

    monkeypatch.setattr(compact, "_unique_position", observe)
    before = "apple\n" * 2048
    after = "apple\n" * 1024 + "pear\n" + "apple\n" * 1023
    completion = compact.search_replace_completion(before, after)
    assert compact.apply_search_replace(before, completion) == after
    # Includes the serializer's round-trip check and this test's apply check.
    assert len(attempts) <= 14


def test_control_markers_are_escaped_before_cli_admission(tmp_path, admission_tokenizer):
    directory, tokenizer = admission_tokenizer
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "bos_token": "[BOS]"})
    tokenizer.save_pretrained(directory)
    pair = {
        "canonical_id": "fruit-markers",
        "lineage_id": "fruit-source",
        "instruction": "Replace the fruit label.",
        "before": "apple [EOS] [PAD] [BOS]\n",
        "after": "pear\n",
    }
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [pair])
    output = tmp_path / "admitted"
    result = run_compact_admission(source, directory, output, 1000)
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads((output / "admitted.jsonl").read_text())
    completion = row["text"][row["completion_start_char"] :]
    assert all(marker not in completion for marker in tokenizer.all_special_tokens)
    assert apply_search_replace(row["source_text"], completion) == pair["after"]
    labels = completion_tokens(row, tokenizer, 1000)["labels"]
    assert labels[-1] == tokenizer.eos_token_id
    assert not {
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
    }.intersection(labels[:-1])


@pytest.mark.parametrize("raw", [b"", b"{", b"{} trailing", b"[]", b'{"a":1,"a":2}'])
def test_compact_json_rejects_invalid_direct_inputs(raw):
    from agoge_forger.compact_json import compact_json_line

    with pytest.raises(ValueError):
        compact_json_line(raw, {})


def test_compact_output_can_be_frozen(tmp_path, admission_tokenizer):
    from tests.test_completion_admission import run_command

    directory, _ = admission_tokenizer
    source = tmp_path / "pairs.jsonl"
    write_rows(
        source,
        [
            {
                "canonical_id": f"fruit-{index}",
                "lineage_id": f"fruit-source-{index}",
                "instruction": f"Update fruit {index}",
                "before": f"apple {index}\n",
                "after": f"pear {index}\n",
            }
            for index in range(100)
        ],
    )
    admitted = tmp_path / "admitted"
    assert run_compact_admission(source, directory, admitted, 1000).returncode == 0
    args = [
        str(Path(sys.executable).with_name("agoge")),
        "freeze-split",
        "--source",
        str(admitted / "admitted.jsonl"),
        "--output-dir",
        str(tmp_path / "freeze"),
        "--source-repository",
        "example/offline",
        "--source-revision",
        "a" * 40,
        "--source-path",
        "admitted.jsonl",
        "--dataset-version",
        "synthetic-v1",
        "--seed",
        "1",
        "--salt",
        "synthetic",
    ]
    result = run_command(args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "freeze" / "split_manifest.json").is_file()
