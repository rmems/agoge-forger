import pytest

from agoge_forger.path_safety import (
    resolve_existing_path,
    resolve_output_directory,
    resolve_output_path,
)


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


def test_resolve_output_path_does_not_create_directory(tmp_path):
    out_dir = tmp_path / "new" / "output"
    resolved = resolve_output_path(str(out_dir))
    assert resolved == out_dir
    assert not out_dir.exists()


def test_resolve_output_path_preserves_final_symlink(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.symlink_to(tmp_path / "missing", target_is_directory=True)
    assert resolve_output_path(str(out_dir)) == out_dir


def test_resolve_output_directory_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="must not contain"):
        resolve_output_directory(str(tmp_path / ".." / "escape"))


def test_resolve_existing_path_allows_legitimate_absolute_paths(tmp_path):
    """Absolute paths that don't traverse '..' are accepted.

    The CWE-367 / TOCTOU concern from amazon-q-developer is documented
    as a known limitation: this module guards against explicit '..'
    traversal, not symlink-based escape. Symlink policy is the
    responsibility of the calling code (which can use Path.is_symlink
    or a jail/allowlist).
    """
    target = tmp_path / "real.txt"
    target.write_text("ok")
    resolved = resolve_existing_path(str(target))
    assert resolved == target.resolve()
