from __future__ import annotations

from pathlib import Path

import pytest

from agoge_forger import _atomic_file as atomic_file


def test_publish_bytes_noreplace_fsyncs_parent_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "nested" / "artifact.bin"
    synced: list[Path] = []

    def capture(directory: Path) -> None:
        synced.append(directory.resolve())

    monkeypatch.setattr(atomic_file, "_fsync_directory", capture)
    atomic_file.publish_bytes_noreplace(
        destination,
        b"payload",
        refusal="refusing silent regeneration",
        writer=atomic_file.write_fsynced_bytes,
    )

    assert destination.read_bytes() == b"payload"
    assert synced == [destination.parent.resolve()]


def test_publish_bytes_noreplace_skips_parent_fsync_when_destination_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"existing")
    synced: list[Path] = []

    monkeypatch.setattr(atomic_file, "_fsync_directory", synced.append)
    with pytest.raises(FileExistsError, match="refusing silent regeneration"):
        atomic_file.publish_bytes_noreplace(
            destination,
            b"payload",
            refusal="refusing silent regeneration",
            writer=atomic_file.write_fsynced_bytes,
        )

    assert destination.read_bytes() == b"existing"
    assert synced == []


def test_publish_bytes_noreplace_propagates_parent_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"

    def boom(directory: Path) -> None:
        raise OSError("parent fsync failed")

    monkeypatch.setattr(atomic_file, "_fsync_directory", boom)
    with pytest.raises(OSError, match="parent fsync failed"):
        atomic_file.publish_bytes_noreplace(
            destination,
            b"payload",
            refusal="refusing silent regeneration",
            writer=atomic_file.write_fsynced_bytes,
        )

    assert destination.read_bytes() == b"payload"


def test_fsync_directory_opens_parent_as_directory(tmp_path: Path) -> None:
    atomic_file._fsync_directory(tmp_path)
