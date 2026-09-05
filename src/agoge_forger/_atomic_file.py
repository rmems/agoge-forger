"""Atomic no-replace publication for immutable files."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from ._atomic_directory import rename_noreplace, require_rename_noreplace_support

PayloadWriter = Callable[[Path, bytes], None]


def publish_bytes_noreplace(
    destination: Path,
    payload: bytes,
    *,
    refusal: str,
    writer: PayloadWriter,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_rename_noreplace_support(destination.parent)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ) as staging_dir:
        staged = Path(staging_dir) / destination.name
        writer(staged, payload)
        try:
            rename_noreplace(staged, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"{refusal}: {destination}") from exc
        _fsync_directory(destination.parent)


def write_fsynced_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
