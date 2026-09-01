"""Descriptor-pinned private snapshots of evaluation artifact bundles."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from ._artifact_schema import (
    ArtifactIndex,
    ArtifactIndexEntry,
    IndexedArtifacts,
    _parse_artifact_index,
    _portable_artifact_path,
)
from ._descriptor_bundle import (
    EntryIdentity,
    copy_relative_file,
    open_bundle,
    read_relative_file,
    require_descriptor_support,
    scan_bundle,
)


@contextmanager
def verified_artifact_snapshot(
    root: Path,
    index_path: Path,
    expected_index_sha256: str,
) -> Iterator[tuple[ArtifactIndex, IndexedArtifacts]]:
    require_descriptor_support()
    initial = scan_bundle(root)
    index_relative = PurePosixPath(index_path.relative_to(root).as_posix())
    index_payload = _read_verified_index(root, index_relative, expected_index_sha256)
    index = _parse_artifact_index(index_path, index_payload)
    indexed = _indexed_artifacts(root, index)
    _require_complete_index(initial, set(indexed) | {index_relative})

    with tempfile.TemporaryDirectory(
        prefix=".agoge-artifact-snapshot-", dir=root.parent
    ) as snapshot_dir:
        snapshot_indexed = _copy_snapshot(
            root, Path(snapshot_dir), index_relative, expected_index_sha256, indexed
        )
        yield index, snapshot_indexed
        _require_bundle_unchanged(root, initial)


def _read_verified_index(root: Path, index_relative: PurePosixPath, expected_digest: str) -> bytes:
    root_descriptor = open_bundle(root)
    try:
        payload, digest = read_relative_file(root_descriptor, index_relative)
    finally:
        os.close(root_descriptor)
    if digest != expected_digest:
        raise ValueError(
            f"SFT artifact-index SHA-256 mismatch: expected {expected_digest}, found {digest}"
        )
    return payload


def _require_complete_index(
    identities: dict[PurePosixPath, EntryIdentity], expected_files: set[PurePosixPath]
) -> None:
    actual_files = {path for path, identity in identities.items() if identity.kind == "file"}
    omitted = sorted(str(path) for path in actual_files - expected_files)
    if omitted:
        raise ValueError(f"artifact bundle contains files omitted from artifact index: {omitted}")
    missing = sorted(str(path) for path in expected_files - actual_files)
    if missing:
        raise ValueError(f"artifact index references missing files: {missing}")


def _copy_snapshot(
    root: Path,
    snapshot_root: Path,
    index_relative: PurePosixPath,
    expected_index_digest: str,
    indexed: IndexedArtifacts,
) -> IndexedArtifacts:
    root_descriptor = open_bundle(root)
    try:
        _, index_digest = read_relative_file(root_descriptor, index_relative)
        if index_digest != expected_index_digest:
            raise ValueError("SFT artifact index changed while creating bundle snapshot")
        return {
            portable: (
                entry,
                _copy_indexed_file(root_descriptor, snapshot_root, portable, entry),
            )
            for portable, (entry, _) in indexed.items()
        }
    finally:
        os.close(root_descriptor)


def _indexed_artifacts(root: Path, index: ArtifactIndex) -> IndexedArtifacts:
    indexed: IndexedArtifacts = {}
    for entry in index.artifacts:
        portable = _portable_artifact_path(entry.file)
        if portable in indexed:
            raise ValueError("artifact index resolves duplicate targets")
        indexed[portable] = (entry, root.joinpath(*portable.parts))
    return indexed


def _copy_indexed_file(
    root_descriptor: int,
    snapshot_root: Path,
    portable: PurePosixPath,
    entry: ArtifactIndexEntry,
) -> Path:
    destination = snapshot_root.joinpath(*portable.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size, digest = copy_relative_file(root_descriptor, portable, destination)
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


def _require_bundle_unchanged(root: Path, initial: dict[PurePosixPath, EntryIdentity]) -> None:
    if scan_bundle(root) != initial:
        raise ValueError("artifact bundle changed while it was being validated")
