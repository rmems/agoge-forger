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


def test_non_object_jsonl_row_fails():
    with pytest.raises(ValueError, match="Line 1: JSONL row must be an object."):
        normalize_row(None, index=1)
    with pytest.raises(ValueError, match="Line 2: JSONL row must be an object."):
        normalize_row(1, index=2)
    with pytest.raises(ValueError, match="Line 3: JSONL row must be an object."):
        normalize_row(["not", "an", "object"], index=3)


def test_load_jsonl_scalar_row_raises(tmp_path):
    path = tmp_path / "scalar.jsonl"
    path.write_text("null\n")
    with pytest.raises(ValueError, match="Line 1: JSONL row must be an object."):
        load_jsonl_dataset(str(path))


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


@pytest.fixture
def isolated_hf_cache(tmp_path, monkeypatch):
    """Shared, isolated HF datasets cache so cache reuse is observable."""
    import datasets.config

    cache_dir = tmp_path / "hf-datasets-cache"
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", cache_dir)
    return cache_dir


def test_load_jsonl_rewritten_same_path_does_not_reuse_stale_cache(tmp_path, isolated_hf_cache):
    path = tmp_path / "data" / "train.jsonl"
    path.parent.mkdir()
    path.write_text('{"text": "hello world", "completion_start_char": 6}\n')
    first = load_jsonl_dataset(str(path))
    assert first["text"] == ["hello world"]

    path.write_text(
        '{"text": "NEW hello", "completion_start_char": 4}\n'
        '{"text": "NEW2 x", "completion_start_char": 5}\n'
    )
    second = load_jsonl_dataset(str(path))
    assert second.to_list() == [
        {"text": "NEW hello", "completion_start_char": 4},
        {"text": "NEW2 x", "completion_start_char": 5},
    ]


def test_load_jsonl_rewritten_messages_file_reloads_new_rows(tmp_path, isolated_hf_cache):
    path = tmp_path / "train.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "broken row"}]}\n')
    assert load_jsonl_dataset(str(path))["text"] == ["User: broken row"]

    path.write_text(
        '{"messages": [{"role": "user", "content": "fixed row"}]}\n{"text": "gamma delta"}\n'
    )
    assert load_jsonl_dataset(str(path))["text"] == ["User: fixed row", "gamma delta"]


def test_load_jsonl_unchanged_file_loads_consistently(tmp_path, isolated_hf_cache):
    path = tmp_path / "train.jsonl"
    path.write_text('{"text": "same"}\n\n{"instruction": "do it", "output": "done"}\n')
    expected = ["same", "Instruction: do it\nOutput: done"]
    assert load_jsonl_dataset(str(path))["text"] == expected
    assert load_jsonl_dataset(str(path))["text"] == expected


def test_load_jsonl_fails_closed_when_file_changes_after_hashing(
    tmp_path, isolated_hf_cache, monkeypatch
):
    import agoge_forger.datasets as datasets_module

    path = tmp_path / "train.jsonl"
    path.write_text('{"text": "original"}\n')
    real_sha256_file = datasets_module.sha256_file

    def hash_then_rewrite(target):
        digest = real_sha256_file(target)
        target.write_text('{"text": "swapped"}\n')
        return digest

    monkeypatch.setattr(datasets_module, "sha256_file", hash_then_rewrite)
    with pytest.raises(ValueError, match="changed after cache identity was established"):
        load_jsonl_dataset(str(path))

    # Nothing stale was committed under the old digest: a fresh load sees the current bytes.
    monkeypatch.setattr(datasets_module, "sha256_file", real_sha256_file)
    assert load_jsonl_dataset(str(path))["text"] == ["swapped"]


def test_load_jsonl_cache_hit_fails_closed_when_file_changes_after_hashing(
    tmp_path, isolated_hf_cache, monkeypatch
):
    """A cached digest skips the generator, so the loader must re-check the file itself."""
    import agoge_forger.datasets as datasets_module

    path = tmp_path / "train.jsonl"
    path.write_text('{"text": "original"}\n')
    assert load_jsonl_dataset(str(path))["text"] == ["original"]  # populate the cache
    real_sha256_file = datasets_module.sha256_file
    calls: list[object] = []

    def hash_then_rewrite_once(target):
        digest = real_sha256_file(target)
        if not calls:
            target.write_text('{"text": "swapped"}\n')
        calls.append(target)
        return digest

    monkeypatch.setattr(datasets_module, "sha256_file", hash_then_rewrite_once)
    with pytest.raises(ValueError, match="changed after cache identity was established"):
        load_jsonl_dataset(str(path))


def test_jsonl_readers_decode_utf8_regardless_of_locale(tmp_path):
    """JSONL is UTF-8; an ASCII locale must not reject or mis-decode it."""
    import os
    import subprocess  # nosec B404 - fixed interpreter and script, no shell
    import sys

    path = tmp_path / "train.jsonl"
    path.write_bytes('{"text": "café ✓"}\n'.encode())
    script = (
        "import sys\n"
        "from agoge_forger.datasets import iter_normalized_rows\n"
        "rows = [row['text'] for _, row in iter_normalized_rows(sys.argv[1])]\n"
        "assert rows == ['caf\\u00e9 \\u2713'], rows\n"
    )
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    result = subprocess.run(  # nosec B603 - fixed interpreter and script
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
