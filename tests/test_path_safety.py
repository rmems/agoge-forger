from pathlib import Path

import pytest

from agoge_forger import path_safety
from agoge_forger.path_safety import (
    resolve_absent_output_directory,
    resolve_existing_path,
    resolve_output_directory,
)


def _allowlist_ambient_tmp(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Simulate macOS `/tmp` → `/private/tmp` with a pytest-owned prefix pair."""
    real_tmp = tmp_path / "private-tmp"
    ambient = tmp_path / "tmp"
    real_tmp.mkdir()
    ambient.symlink_to(real_tmp, target_is_directory=True)
    monkeypatch.setattr(path_safety, "AMBIENT_TEMP_PREFIXES", (ambient, real_tmp))
    return ambient, real_tmp


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


def test_resolve_absent_output_directory_creates_parent_only(tmp_path):
    out_dir = tmp_path / "nested" / "snapshot"
    resolved = resolve_absent_output_directory(str(out_dir))
    assert resolved == out_dir.resolve()
    assert resolved.parent.is_dir()
    assert not resolved.exists()


def test_resolve_absent_output_directory_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="must not contain"):
        resolve_absent_output_directory(str(tmp_path / ".." / "escape"))


def test_resolve_absent_output_directory_rejects_dangling_leaf_symlink(tmp_path):
    elsewhere = tmp_path / "elsewhere" / "hijacked"
    output = tmp_path / "snapshot"
    output.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked path"):
        resolve_absent_output_directory(str(output))

    assert output.is_symlink()
    assert not elsewhere.exists()
    assert not (tmp_path / "elsewhere").exists()


def test_resolve_absent_output_directory_rejects_existing_leaf_symlink(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    output = tmp_path / "snapshot"
    output.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked path"):
        resolve_absent_output_directory(str(output))

    assert output.is_symlink()
    assert list(elsewhere.iterdir()) == []


def test_resolve_absent_output_directory_rejects_symlinked_parent(tmp_path):
    real_parent = tmp_path / "real-parent"
    linked_parent = tmp_path / "linked-parent"
    real_parent.mkdir()
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked path"):
        resolve_absent_output_directory(str(linked_parent / "snapshot"))

    assert linked_parent.is_symlink()
    assert not (real_parent / "snapshot").exists()
    assert list(real_parent.iterdir()) == []


def test_resolve_absent_output_directory_rejects_symlink_above_existing_dir(tmp_path):
    """An existing real directory under a symlink must still be refused.

    Stopping at the first existing ancestor would treat link/existing as safe
    and let Path.resolve() publish under the symlink target.
    """
    real = tmp_path / "real"
    existing = real / "existing"
    linked = tmp_path / "linked"
    real.mkdir()
    existing.mkdir()
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked path"):
        resolve_absent_output_directory(str(linked / "existing" / "snapshot"))

    assert linked.is_symlink()
    assert not (existing / "snapshot").exists()
    assert list(existing.iterdir()) == []


def test_resolve_absent_output_directory_allows_allowlisted_temp_prefix(tmp_path, monkeypatch):
    ambient, real_tmp = _allowlist_ambient_tmp(tmp_path, monkeypatch)
    output = ambient / "nested" / "snapshot"

    resolved = resolve_absent_output_directory(str(output))

    assert ambient.is_symlink()
    assert resolved == (real_tmp / "nested" / "snapshot").resolve()
    assert resolved.parent.is_dir()
    assert not resolved.exists()


def test_resolve_absent_output_directory_allowlisted_prefix_still_refuses_leaf(
    tmp_path, monkeypatch
):
    ambient, _real_tmp = _allowlist_ambient_tmp(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere" / "hijacked"
    output = ambient / "snapshot"
    output.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked path"):
        resolve_absent_output_directory(str(output))

    assert output.is_symlink()
    assert not elsewhere.exists()


def test_resolve_absent_output_directory_allowlisted_prefix_still_refuses_parent(
    tmp_path, monkeypatch
):
    ambient, real_tmp = _allowlist_ambient_tmp(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    linked = ambient / "linked-parent"
    linked.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked path"):
        resolve_absent_output_directory(str(linked / "snapshot"))

    assert linked.is_symlink()
    assert not (elsewhere / "snapshot").exists()
    assert not (real_tmp / "linked-parent" / "snapshot").exists()


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
