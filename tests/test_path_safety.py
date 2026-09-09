from collections.abc import Callable
from pathlib import Path

import pytest

from agoge_forger import path_safety
from agoge_forger.path_safety import (
    resolve_absent_output_directory,
    resolve_existing_path,
    resolve_output_directory,
)

_Layout = Callable[[Path], tuple[Path, Path, tuple[Path, ...]]]


def _allowlist_ambient_tmp(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Simulate macOS `/tmp` → `/private/tmp` with a pytest-owned prefix pair."""
    real_tmp = tmp_path / "private-tmp"
    ambient = tmp_path / "tmp"
    real_tmp.mkdir()
    ambient.symlink_to(real_tmp, target_is_directory=True)
    monkeypatch.setattr(path_safety, "AMBIENT_TEMP_PREFIXES", (ambient, real_tmp))
    return ambient, real_tmp


def _dangling_leaf(root: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    target = root / "elsewhere" / "hijacked"
    output = root / "snapshot"
    output.symlink_to(target, target_is_directory=True)
    return output, output, (target, target.parent)


def _existing_leaf(root: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    target = root / "elsewhere"
    target.mkdir()
    output = root / "snapshot"
    output.symlink_to(target, target_is_directory=True)
    return output, output, ()


def _symlinked_parent(root: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    real_parent = root / "real-parent"
    linked = root / "linked-parent"
    real_parent.mkdir()
    linked.symlink_to(real_parent, target_is_directory=True)
    return linked / "snapshot", linked, (real_parent / "snapshot",)


def _symlink_above_existing(root: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    real = root / "real"
    existing = real / "existing"
    linked = root / "linked"
    real.mkdir()
    existing.mkdir()
    linked.symlink_to(real, target_is_directory=True)
    return linked / "existing" / "snapshot", linked, (existing / "snapshot",)


def _user_parent(root: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    elsewhere = root.parent / "elsewhere"
    elsewhere.mkdir()
    linked = root / "linked-parent"
    linked.symlink_to(elsewhere, target_is_directory=True)
    return linked / "snapshot", linked, (elsewhere / "snapshot",)


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


@pytest.mark.parametrize(
    ("allowlist", "layout"),
    [
        pytest.param(False, _dangling_leaf, id="dangling-leaf"),
        pytest.param(False, _existing_leaf, id="existing-leaf"),
        pytest.param(False, _symlinked_parent, id="symlinked-parent"),
        pytest.param(False, _symlink_above_existing, id="symlink-above-existing"),
        pytest.param(True, _dangling_leaf, id="allowlisted-prefix-leaf"),
        pytest.param(True, _user_parent, id="allowlisted-prefix-parent"),
    ],
)
def test_resolve_absent_output_directory_refuses_symlinks(
    tmp_path, monkeypatch, allowlist: bool, layout: _Layout
):
    root = tmp_path
    if allowlist:
        root, _real_tmp = _allowlist_ambient_tmp(tmp_path, monkeypatch)
    output, link, forbidden = layout(root)

    with pytest.raises(ValueError, match="symlinked path"):
        resolve_absent_output_directory(str(output))

    assert link.is_symlink()
    assert all(not path.exists() for path in forbidden)
    if output.exists():
        assert list(output.iterdir()) == []


def test_resolve_absent_output_directory_allows_allowlisted_temp_prefix(tmp_path, monkeypatch):
    ambient, real_tmp = _allowlist_ambient_tmp(tmp_path, monkeypatch)
    output = ambient / "nested" / "snapshot"

    resolved = resolve_absent_output_directory(str(output))

    assert ambient.is_symlink()
    assert resolved == (real_tmp / "nested" / "snapshot").resolve()
    assert resolved.parent.is_dir()
    assert not resolved.exists()


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
