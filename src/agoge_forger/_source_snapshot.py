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
    try:
        copied, identity_before, identity_after, path_identity_after = _copy_source_bytes(
            source, snapshot
        )
    except OSError as exc:
        raise ValueError(f"source changed while creating immutable snapshot: {source}") from exc
    _require_unchanged_source(source, copied, identity_before, identity_after, path_identity_after)
    return copied


def _copy_source_bytes(
    source: Path, snapshot: Path
) -> tuple[str, _FileIdentity, _FileIdentity, _FileIdentity]:
    digest = hashlib.sha256()
    with source.open("rb") as source_handle, snapshot.open("xb") as snapshot_handle:
        identity_before = _file_identity(os.fstat(source_handle.fileno()))
        _copy_file_chunks(source_handle, snapshot_handle, digest)
        identity_after = _file_identity(os.fstat(source_handle.fileno()))
        path_identity_after = _file_identity(source.stat())
    return digest.hexdigest(), identity_before, identity_after, path_identity_after


def _copy_file_chunks(source_handle, snapshot_handle, digest) -> None:
    for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
        digest.update(chunk)
        snapshot_handle.write(chunk)


def _require_unchanged_source(
    source: Path,
    copied: str,
    identity_before: _FileIdentity,
    identity_after: _FileIdentity,
    path_identity_after: _FileIdentity,
) -> None:
    if identity_before != identity_after or identity_after != path_identity_after:
        raise ValueError(f"source changed while creating immutable snapshot: {source}")
    # Overlay filesystems can rewrite bytes without changing size/mtime/ctime.
    # Re-hash the live path so content races are not metadata-only.
    if _sha256_path(source) != copied:
        raise ValueError(f"source changed while creating immutable snapshot: {source}")


def nearest_existing_output_ancestor(destination: Path) -> Path:
    candidate = destination.absolute().parent
    while not candidate.exists():
        candidate = candidate.parent
    if not candidate.is_dir():
        raise ValueError(f"output path has a non-directory ancestor: {candidate}")
    return candidate.resolve(strict=True)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(status: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        size=status.st_size,
        modified_ns=status.st_mtime_ns,
        changed_ns=status.st_ctime_ns,
    )
