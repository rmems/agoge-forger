"""Race-safe source snapshot creation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def copy_source_snapshot(source: Path, snapshot: Path) -> str:
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_handle, snapshot.open("xb") as snapshot_handle:
            identity_before = _file_identity(os.fstat(source_handle.fileno()))
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                digest.update(chunk)
                snapshot_handle.write(chunk)
            identity_after = _file_identity(os.fstat(source_handle.fileno()))
            path_identity_after = _file_identity(source.stat())
    except OSError as exc:
        raise ValueError(f"source changed while creating immutable snapshot: {source}") from exc
    if identity_before != identity_after or identity_after != path_identity_after:
        raise ValueError(f"source changed while creating immutable snapshot: {source}")
    return digest.hexdigest()


def _file_identity(status: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        size=status.st_size,
        modified_ns=status.st_mtime_ns,
        changed_ns=status.st_ctime_ns,
    )
