"""Resource and path-safety tests for offline tokenizer inspection."""

import json
import os
from pathlib import Path

import pytest

from agoge_forger import _run_status_pretrained as pretrained


class _Tokenizer:
    pad_token = None
    eos_token = object()

    def __len__(self) -> int:
        return 2


def _fail_tokenizer_load(*args, **kwargs):
    raise AssertionError("invalid tokenizer inventory reached AutoTokenizer")


@pytest.mark.parametrize(
    ("sizes", "count"),
    [
        ([64 * 1024 * 1024 + 1], 1),
        ([48 * 1024 * 1024] * 3, 3),
        ([0], 129),
    ],
    ids=["per-file", "aggregate", "file-count"],
)
def test_oversized_local_tokenizer_inventory_is_rejected_before_loading(
    tmp_path, monkeypatch, sizes, count
):
    model = tmp_path / "model"
    model.mkdir()
    for index in range(count):
        size = sizes[index % len(sizes)]
        with (model / f"tokenizer-{index}.json").open("wb") as handle:
            handle.truncate(size)
    monkeypatch.setattr(pretrained.AutoTokenizer, "from_pretrained", _fail_tokenizer_load)

    assert pretrained.tokenizer_usable(model) is False


@pytest.mark.parametrize("kind", ["escape", "broken", "fifo"])
def test_unsafe_local_tokenizer_entries_are_rejected(tmp_path, monkeypatch, kind):
    model = tmp_path / "model"
    model.mkdir()
    entry = model / "tokenizer.json"
    if kind == "escape":
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        entry.symlink_to(outside)
    elif kind == "broken":
        entry.symlink_to(tmp_path / "missing.json")
    else:
        os.mkfifo(entry)
    monkeypatch.setattr(pretrained.AutoTokenizer, "from_pretrained", _fail_tokenizer_load)

    assert pretrained.tokenizer_usable(model) is False


@pytest.mark.parametrize(
    "config",
    [
        {"tokenizer_file": "tokenizer.payload"},
        {"emoji_file": "tokenizer.payload"},
        {"fast_tokenizer_files": ["tokenizer.payload"]},
    ],
    ids=["tokenizer-file", "model-specific-file", "fast-tokenizer-files"],
)
def test_referenced_arbitrary_suffix_tokenizer_is_size_bounded(tmp_path, monkeypatch, config):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text(json.dumps(config))
    with (model / "tokenizer.payload").open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)
    monkeypatch.setattr(pretrained.AutoTokenizer, "from_pretrained", _fail_tokenizer_load)

    assert pretrained.tokenizer_usable(model) is False


@pytest.mark.parametrize("reference", ["absolute", "parent"])
def test_tokenizer_config_references_must_stay_inside_inventory(tmp_path, monkeypatch, reference):
    model = tmp_path / "model"
    model.mkdir()
    outside = tmp_path / "outside.payload"
    outside.write_text("tokenizer")
    configured_path = str(outside) if reference == "absolute" else "../outside.payload"
    (model / "tokenizer_config.json").write_text(json.dumps({"tokenizer_file": configured_path}))
    monkeypatch.setattr(pretrained.AutoTokenizer, "from_pretrained", _fail_tokenizer_load)

    assert pretrained.tokenizer_usable(model) is False


def test_tokenizer_config_is_size_bounded_before_parsing(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    with (model / "tokenizer_config.json").open("wb") as handle:
        handle.truncate(4 * 1024 * 1024 + 1)
    monkeypatch.setattr(pretrained.AutoTokenizer, "from_pretrained", _fail_tokenizer_load)

    assert pretrained.tokenizer_usable(model) is False


def test_excessive_ignored_inventory_entries_are_rejected(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for index in range(1025):
        (model / f"ignored-{index}.payload").touch()
    monkeypatch.setattr(pretrained.AutoTokenizer, "from_pretrained", _fail_tokenizer_load)

    assert pretrained.tokenizer_usable(model) is False


def _cached_snapshot(tmp_path: Path, *, escaping: bool) -> tuple[Path, Path]:
    repository = tmp_path / "models--org--model"
    snapshot = repository / "snapshots" / "revision"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    config_blob = blobs / "config"
    config_blob.write_text("{}")
    (snapshot / "config.json").symlink_to(os.path.relpath(config_blob, snapshot))
    tokenizer_config_blob = blobs / "tokenizer-config"
    tokenizer_config_blob.write_text(json.dumps({"tokenizer_file": "tokenizer.payload"}))
    (snapshot / "tokenizer_config.json").symlink_to(
        os.path.relpath(tokenizer_config_blob, snapshot)
    )
    tokenizer_target = tmp_path / "outside.payload" if escaping else blobs / "tokenizer"
    tokenizer_target.write_text("{}")
    (snapshot / "tokenizer.payload").symlink_to(os.path.relpath(tokenizer_target, snapshot))
    return snapshot, snapshot / "config.json"


@pytest.mark.parametrize("escaping", [False, True], ids=["contained", "escape"])
def test_cached_tokenizer_symlinks_must_stay_inside_repository_boundary(
    tmp_path, monkeypatch, escaping
):
    snapshot, cached_config = _cached_snapshot(tmp_path, escaping=escaping)
    monkeypatch.setattr(pretrained, "cached_file", lambda *args, **kwargs: str(cached_config))
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return _Tokenizer()

    monkeypatch.setattr(pretrained.AutoTokenizer, "from_pretrained", load)

    assert pretrained.tokenizer_usable("org/model", revision="revision") is not escaping
    assert bool(calls) is not escaping
    assert snapshot.is_dir()
