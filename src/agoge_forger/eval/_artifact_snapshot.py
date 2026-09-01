"""Descriptor-pinned private snapshots of evaluation artifact bundles."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ._artifact_schema import (
    ArtifactIndex,
    ArtifactIndexEntry,
    IndexedArtifacts,
    _parse_artifact_index,
    _portable_artifact_path,
)


@dataclass(frozen=True)
class _EntryIdentity:
    kind: str
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@contextmanager
def verified_artifact_snapshot(
    root: Path,
    index_path: Path,
    expected_index_sha256: str,
) -> Iterator[tuple[ArtifactIndex, IndexedArtifacts]]:
    _require_descriptor_support()
    initial = _scan_bundle(root)
    index_relative = PurePosixPath(index_path.relative_to(root).as_posix())
    root_descriptor = _open_root(root)
    try:
        index_payload, index_digest = _read_relative_file(root_descriptor, index_relative)
    finally:
        os.close(root_descriptor)
    if index_digest != expected_index_sha256:
        raise ValueError(
            f"SFT artifact-index SHA-256 mismatch: expected {expected_index_sha256}, "
            f"found {index_digest}"
        )
    index = _parse_artifact_index(index_path, index_payload)
    indexed = _indexed_artifacts(root, index)
    expected_files = set(indexed) | {index_relative}
    actual_files = {path for path, identity in initial.items() if identity.kind == "file"}
    if actual_files != expected_files:
        omitted = sorted(str(path) for path in actual_files - expected_files)
        missing = sorted(str(path) for path in expected_files - actual_files)
        if omitted:
            raise ValueError(
                f"artifact bundle contains files omitted from artifact index: {omitted}"
            )
        raise ValueError(f"artifact index references missing files: {missing}")

    with tempfile.TemporaryDirectory(prefix="agoge-artifact-snapshot-") as snapshot_dir:
        snapshot_root = Path(snapshot_dir)
        root_descriptor = _open_root(root)
        try:
            _, copied_index_digest = _read_relative_file(root_descriptor, index_relative)
            if copied_index_digest != expected_index_sha256:
                raise ValueError("SFT artifact index changed while creating bundle snapshot")
            snapshot_indexed = {
                portable: (
                    entry,
                    _copy_indexed_file(root_descriptor, snapshot_root, portable, entry),
                )
                for portable, (entry, _) in indexed.items()
            }
        finally:
            os.close(root_descriptor)

        yield index, snapshot_indexed

        if _scan_bundle(root) != initial:
            raise ValueError("artifact bundle changed while it was being validated")


def _require_descriptor_support() -> None:
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


def _indexed_artifacts(root: Path, index: ArtifactIndex) -> IndexedArtifacts:
    indexed: IndexedArtifacts = {}
    for entry in index.artifacts:
        portable = _portable_artifact_path(entry.file)
        if portable in indexed:
            raise ValueError("artifact index resolves duplicate targets")
        indexed[portable] = (entry, root.joinpath(*portable.parts))
    return indexed


def _open_root(root: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        return os.open(root, flags)
    except OSError as exc:
        raise ValueError(f"artifact bundle is not readable: {root}") from exc


def _scan_bundle(root: Path) -> dict[PurePosixPath, _EntryIdentity]:
    root_descriptor = _open_root(root)
    try:
        identities = {PurePosixPath("."): _identity(os.fstat(root_descriptor), "directory")}
        _scan_directory(root_descriptor, PurePosixPath(), identities)
        return identities
    finally:
        os.close(root_descriptor)


def _scan_directory(
    descriptor: int,
    prefix: PurePosixPath,
    identities: dict[PurePosixPath, _EntryIdentity],
) -> None:
    try:
        with os.scandir(descriptor) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
    except OSError as exc:
        raise ValueError("artifact bundle directory is not readable") from exc
    for entry in ordered:
        relative = prefix / entry.name
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"artifact bundle entry is not readable: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"artifact bundle cannot contain symlinks: {relative}")
        if stat.S_ISREG(metadata.st_mode):
            identities[relative] = _identity(metadata, "file")
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"artifact bundle entry is not a regular file: {relative}")
        child = _open_directory_component(descriptor, entry.name)
        try:
            opened = _identity(os.fstat(child), "directory")
            if opened != _identity(metadata, "directory"):
                raise ValueError(f"artifact bundle directory changed while scanning: {relative}")
            identities[relative] = opened
            _scan_directory(child, relative, identities)
        finally:
            os.close(child)


def _open_directory_component(parent: int, name: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        return os.open(name, flags, dir_fd=parent)
    except OSError as exc:
        raise ValueError(f"artifact bundle directory is not readable: {name}") from exc


def _open_relative_file(root_descriptor: int, relative: PurePosixPath) -> tuple[int, int, str]:
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
        return source, directory, relative.parts[-1]
    except (OSError, ValueError):
        os.close(directory)
        raise


def _read_relative_file(root_descriptor: int, relative: PurePosixPath) -> tuple[bytes, str]:
    source, parent, name = _open_relative_file(root_descriptor, relative)
    try:
        payload = _read_descriptor(source)
        _require_unchanged_entry(source, parent, name, relative)
        return payload, hashlib.sha256(payload).hexdigest()
    finally:
        os.close(source)
        os.close(parent)


def _read_descriptor(source: int) -> bytes:
    before = os.fstat(source)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("artifact bundle entry is not a regular file")
    with os.fdopen(os.dup(source), "rb") as handle:
        payload = handle.read()
    if _identity(os.fstat(source), "file") != _identity(before, "file"):
        raise ValueError("artifact bundle entry changed while being read")
    return payload


def _copy_indexed_file(
    root_descriptor: int,
    snapshot_root: Path,
    portable: PurePosixPath,
    entry: ArtifactIndexEntry,
) -> Path:
    source, parent, name = _open_relative_file(root_descriptor, portable)
    destination = snapshot_root.joinpath(*portable.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        size, digest = _copy_and_digest(source, destination)
        _require_unchanged_entry(source, parent, name, portable)
    finally:
        os.close(source)
        os.close(parent)
    if size != entry.size_bytes:
        raise ValueError(
            f"indexed artifact size mismatch for {entry.file}: "
            f"expected {entry.size_bytes}, found {size}"
        )
    if digest != entry.sha256:
        raise ValueError(
            f"indexed artifact SHA-256 mismatch for {entry.file}: "
            f"expected {entry.sha256}, found {digest}"
        )
    return destination


def _copy_and_digest(source: int, destination: Path | None) -> tuple[int, str]:
    before = os.fstat(source)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("artifact bundle entry is not a regular file")
    digest = hashlib.sha256()
    size = 0
    target = destination.open("xb") if destination is not None else None
    try:
        with os.fdopen(os.dup(source), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                if target is not None:
                    target.write(chunk)
    finally:
        if target is not None:
            target.close()
    if _identity(os.fstat(source), "file") != _identity(before, "file"):
        raise ValueError("artifact bundle entry changed while being copied")
    return size, digest.hexdigest()


def _require_unchanged_entry(
    source: int,
    parent: int,
    name: str,
    relative: PurePosixPath,
) -> None:
    try:
        path_metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"artifact bundle entry changed while being copied: {relative}") from exc
    if _identity(path_metadata, "file") != _identity(os.fstat(source), "file"):
        raise ValueError(f"artifact bundle entry changed while being copied: {relative}")


def _identity(metadata: os.stat_result, kind: str) -> _EntryIdentity:
    return _EntryIdentity(
        kind=kind,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )
