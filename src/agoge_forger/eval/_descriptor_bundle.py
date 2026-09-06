"""No-follow descriptor traversal for immutable bundle validation."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO


@dataclass(frozen=True)
class EntryIdentity:
    kind: str
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _ScannedEntry:
    name: str
    relative: PurePosixPath
    metadata: os.stat_result


@dataclass(frozen=True)
class _OpenedEntry:
    source: int
    parent: int
    name: str
    relative: PurePosixPath


def require_descriptor_support() -> None:
    supported = (
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.scandir in os.supports_fd
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
    )
    if not supported:
        raise ValueError(
            "artifact bundle cannot be validated safely on this platform: "
            "descriptor-relative no-follow traversal is unsupported"
        )


def open_bundle(root: Path) -> int:
    return _open_directory(root, f"artifact bundle is not readable: {root}")


def scan_bundle(root: Path) -> dict[PurePosixPath, EntryIdentity]:
    root_descriptor = open_bundle(root)
    try:
        identities = {PurePosixPath("."): _identity(os.fstat(root_descriptor), "directory")}
        _scan_directory(root_descriptor, PurePosixPath(), identities)
        return identities
    finally:
        os.close(root_descriptor)


def read_relative_file(root_descriptor: int, relative: PurePosixPath) -> tuple[bytes, str]:
    with _opened_relative_file(root_descriptor, relative) as opened:
        target = io.BytesIO()
        _, digest = _stream_descriptor(opened.source, target, "read")
        _require_unchanged_entry(opened, "read")
        return target.getvalue(), digest


def copy_relative_file(
    root_descriptor: int, relative: PurePosixPath, destination: Path
) -> tuple[int, str]:
    with _opened_relative_file(root_descriptor, relative) as opened:
        with destination.open("xb") as target:
            size, digest = _stream_descriptor(opened.source, target, "copied")
        _require_unchanged_entry(opened, "copied")
        return size, digest


def _scan_directory(
    descriptor: int,
    prefix: PurePosixPath,
    identities: dict[PurePosixPath, EntryIdentity],
) -> None:
    for entry in _directory_entries(descriptor):
        _scan_entry(descriptor, prefix, entry, identities)


def _directory_entries(descriptor: int) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(descriptor) as entries:
            return sorted(entries, key=lambda entry: entry.name)
    except OSError as exc:
        raise ValueError("artifact bundle directory is not readable") from exc


def _scan_entry(
    descriptor: int,
    prefix: PurePosixPath,
    entry: os.DirEntry[str],
    identities: dict[PurePosixPath, EntryIdentity],
) -> None:
    relative = prefix / entry.name
    try:
        metadata = entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"artifact bundle entry is not readable: {relative}") from exc
    scanned = _ScannedEntry(entry.name, relative, metadata)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"artifact bundle cannot contain symlinks: {relative}")
    if stat.S_ISREG(metadata.st_mode):
        identities[relative] = _identity(metadata, "file")
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"artifact bundle entry is not a regular file: {relative}")
    _scan_child_directory(descriptor, scanned, identities)


def _scan_child_directory(
    descriptor: int,
    entry: _ScannedEntry,
    identities: dict[PurePosixPath, EntryIdentity],
) -> None:
    child = _open_directory_component(descriptor, entry.name)
    try:
        opened = _identity(os.fstat(child), "directory")
        if opened != _identity(entry.metadata, "directory"):
            raise ValueError(f"artifact bundle directory changed while scanning: {entry.relative}")
        identities[entry.relative] = opened
        _scan_directory(child, entry.relative, identities)
    finally:
        os.close(child)


def _open_directory_component(parent: int, name: str) -> int:
    return _open_directory(
        name,
        f"artifact bundle directory is not readable: {name}",
        parent=parent,
    )


def _open_directory(path: str | Path, error_message: str, *, parent: int | None = None) -> int:
    try:
        return os.open(path, _directory_flags(), dir_fd=parent)
    except OSError as exc:
        raise ValueError(error_message) from exc


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_relative_file(root_descriptor: int, relative: PurePosixPath) -> _OpenedEntry:
    directory = os.dup(root_descriptor)
    try:
        for component in relative.parts[:-1]:
            child = _open_directory_component(directory, component)
            os.close(directory)
            directory = child
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        source = os.open(relative.parts[-1], flags, dir_fd=directory)
        return _OpenedEntry(source, directory, relative.parts[-1], relative)
    except (OSError, ValueError):
        os.close(directory)
        raise


@contextmanager
def _opened_relative_file(root_descriptor: int, relative: PurePosixPath) -> Iterator[_OpenedEntry]:
    opened = _open_relative_file(root_descriptor, relative)
    try:
        yield opened
    finally:
        os.close(opened.source)
        os.close(opened.parent)


def _stream_descriptor(source: int, target: BinaryIO, action: str) -> tuple[int, str]:
    before = os.fstat(source)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("artifact bundle entry is not a regular file")
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(os.dup(source), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            target.write(chunk)
    if _identity(os.fstat(source), "file") != _identity(before, "file"):
        raise ValueError(f"artifact bundle entry changed while being {action}")
    return size, digest.hexdigest()


def _require_unchanged_entry(entry: _OpenedEntry, action: str) -> None:
    try:
        path_metadata = os.stat(entry.name, dir_fd=entry.parent, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(
            f"artifact bundle entry changed while being {action}: {entry.relative}"
        ) from exc
    if _identity(path_metadata, "file") != _identity(os.fstat(entry.source), "file"):
        raise ValueError(f"artifact bundle entry changed while being {action}: {entry.relative}")


def _identity(metadata: os.stat_result, kind: str) -> EntryIdentity:
    return EntryIdentity(
        kind=kind,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )
