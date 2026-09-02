import errno
from pathlib import Path

import pytest

from agoge_forger import _atomic_directory
from agoge_forger._atomic_directory import rename_noreplace, require_rename_noreplace_support


def test_support_probe_runs_on_destination_filesystem(tmp_path: Path) -> None:
    require_rename_noreplace_support(tmp_path)

    assert not list(tmp_path.glob(".agoge-rename-probe-*"))


@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EINVAL, errno.ENOTSUP])
def test_support_probe_translates_unsupported_filesystems(tmp_path, monkeypatch, error):
    def unsupported(source, destination):
        raise OSError(error, "unsupported")

    monkeypatch.setattr(_atomic_directory, "rename_noreplace", unsupported)

    with pytest.raises(OSError, match="destination filesystem") as caught:
        require_rename_noreplace_support(tmp_path)

    assert caught.value.errno == errno.ENOTSUP
    assert not list(tmp_path.glob(".agoge-rename-probe-*"))


def test_support_probe_preserves_unrelated_filesystem_error(tmp_path, monkeypatch):
    def denied(source, destination):
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr(_atomic_directory, "rename_noreplace", denied)

    with pytest.raises(OSError, match="denied") as caught:
        require_rename_noreplace_support(tmp_path)

    assert caught.value.errno == errno.EACCES


def test_support_probe_rejects_ignored_no_replace_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(_atomic_directory, "rename_noreplace", lambda *args: None)

    with pytest.raises(OSError, match="destination filesystem") as caught:
        require_rename_noreplace_support(tmp_path)

    assert caught.value.errno == errno.ENOTSUP
    assert not list(tmp_path.glob(".agoge-rename-probe-*"))


def test_rename_noreplace_rejects_embedded_nul_without_truncating(tmp_path: Path) -> None:
    source = tmp_path / "staged"
    source.mkdir()
    (source / "payload.txt").write_text("complete\n", encoding="utf-8")
    truncated_destination = tmp_path / "unexpected"
    destination = Path(f"{truncated_destination}\0ignored")

    with pytest.raises(ValueError, match="embedded NUL"):
        rename_noreplace(source, destination)

    assert source.is_dir()
    assert not truncated_destination.exists()
