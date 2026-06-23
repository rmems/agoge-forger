import pytest

from agoge_forger.path_safety import resolve_existing_path, resolve_output_directory


def test_resolve_existing_path_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="must not contain"):
        resolve_existing_path(str(tmp_path / ".." / "etc" / "passwd"))


def test_resolve_existing_path_requires_existing_file(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        resolve_existing_path(str(missing), must_be_file=True)


def test_resolve_output_directory_creates_directory(tmp_path):
    out_dir = tmp_path / "nested" / "output"
    resolved = resolve_output_directory(str(out_dir))
    assert resolved.is_dir()


def test_resolve_output_directory_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="must not contain"):
        resolve_output_directory(str(tmp_path / ".." / "escape"))
