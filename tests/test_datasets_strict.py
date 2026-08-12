import pytest

from agoge_forger.datasets import load_jsonl_dataset, normalize_row


def test_invalid_dataset_row_fails():
    with pytest.raises(ValueError, match="Line 1: 'text' field must be a string."):
        normalize_row({"text": 123}, index=1)

    with pytest.raises(ValueError, match="Line 2: 'messages' must be a list."):
        normalize_row({"messages": "hi"}, index=2)

    with pytest.raises(ValueError, match="Line 3: invalid role 'invalid'."):
        normalize_row({"messages": [{"role": "invalid", "content": "hi"}]}, index=3)

    with pytest.raises(ValueError, match="Line 4: Unknown format"):
        normalize_row({"unknown": "format"}, index=4)


def test_non_string_message_content_fails():
    with pytest.raises(ValueError, match=r"Line 1: messages\[1\]\.content must be a string\."):
        normalize_row({"messages": [{"role": "user", "content": 123}]}, index=1)

    with pytest.raises(ValueError, match=r"Line 2: messages\[2\]\.content must be a string\."):
        normalize_row(
            {
                "messages": [
                    {"role": "user", "content": "ok"},
                    {"role": "assistant", "content": ["parts"]},
                ]
            },
            index=2,
        )


def test_message_element_must_be_object():
    with pytest.raises(ValueError, match=r"Line 1: messages\[1\] must be an object\."):
        normalize_row({"messages": ["not-an-object"]}, index=1)


def test_load_jsonl_empty_file_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="Dataset is empty"):
        load_jsonl_dataset(str(path))


def test_load_jsonl_whitespace_only_raises(tmp_path):
    path = tmp_path / "blanks.jsonl"
    path.write_text("\n  \n\t\n")
    with pytest.raises(ValueError, match="Dataset is empty"):
        load_jsonl_dataset(str(path))


def test_load_jsonl_valid_text_row(tmp_path):
    path = tmp_path / "ok.jsonl"
    path.write_text('{"text": "hello"}\n')
    ds = load_jsonl_dataset(str(path))
    assert len(ds) == 1
    assert ds[0]["text"] == "hello"
