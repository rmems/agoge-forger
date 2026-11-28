"""Pure text edits and offline CLI integration for compact targets."""

import json

import pytest

from agoge_forger.compact_completion import (
    apply_search_replace,
    render_compact_row,
    search_replace_completion,
)
from agoge_forger.train.completion import completion_tokens
from tests import test_completion_admission as admission_cases
from tests.test_completion_admission import run_admission, write_rows

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
    result = run_admission(source, directory, output, budget, "search-replace-json-v1")
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
    result = run_admission(source, directory, output, 1000, "search-replace-json-v1")
    assert result.returncode == 1
    assert not output.exists()


def test_compact_metadata_precision_and_escaped_keys_are_preserved():
    from agoge_forger.compact_completion import compact_json_line

    raw = (
        b'{ "instruction": "Update fruit", "before":"apple", "after":"pear", '
        b'"timestamp":1700000000.123456789, "nested":{"data":[1.000000000000000001,"a,b}"]}, '
        b'"canonical\\u005fid":"fruit" }'
    )
    row = render_compact_row(json.loads(raw))
    result = compact_json_line(raw, row)
    assert b'"timestamp":1700000000.123456789' in result
    assert b"1.000000000000000001" in result
    assert json.loads(result) == row
