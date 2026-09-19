"""Canonical schema for an offline-verifiable reproducibility bundle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from .._atomic_file import publish_bytes_replace  # noinspection PyProtectedMember
from ..eval._artifact_schema import (  # noinspection PyProtectedMember
    FrozenEvaluationModel,
    portable_artifact_path,
)
from ..eval._descriptor_bundle import (  # noinspection PyProtectedMember
    hash_relative_file,
    open_bundle,
    scan_bundle,
)
from ..split_schema import canonical_json_bytes

BUNDLE_INVENTORY_NAME = "reproducibility-bundle.json"
BUNDLE_SCHEMA_VERSION: Literal["agoge.reproducibility-bundle.v1"] = (
    "agoge.reproducibility-bundle.v1"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_POINTER_FIELDS = (
    "run_manifest_path",
    "locked_config_path",
    "split_manifest_path",
    "adapter_artifact_index_path",
    "merged_artifact_index_path",
    "evaluation_contract_path",
)


@dataclass(frozen=True)
class BundlePointers:
    run_manifest_path: str
    locked_config_path: str
    split_manifest_path: str
    adapter_artifact_index_path: str
    evaluation_contract_path: str
    merged_artifact_index_path: str | None = None


class BundleFileEntry(FrozenEvaluationModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def require_portable_path(self) -> BundleFileEntry:
        portable = portable_artifact_path(self.path)
        if portable.as_posix() != self.path:
            raise ValueError(f"bundle inventory path must be canonical POSIX: {self.path}")
        if self.path == BUNDLE_INVENTORY_NAME:
            raise ValueError("bundle inventory cannot list itself")
        return self


class ReproducibilityBundle(FrozenEvaluationModel):
    schema_version: Literal["agoge.reproducibility-bundle.v1"] = BUNDLE_SCHEMA_VERSION
    files: tuple[BundleFileEntry, ...] = Field(min_length=1)
    run_manifest_path: str = Field(min_length=1)
    locked_config_path: str = Field(min_length=1)
    split_manifest_path: str = Field(min_length=1)
    adapter_artifact_index_path: str = Field(min_length=1)
    evaluation_contract_path: str = Field(min_length=1)
    merged_artifact_index_path: str | None = None

    @model_validator(mode="after")
    def require_unique_listed_pointers(self) -> ReproducibilityBundle:
        _require_unique_files(self.files)
        _require_listed_pointers(self)
        return self


def write_reproducibility_bundle(bundle_root: str | Path, pointers: BundlePointers) -> Path:
    """Seal a complete membership inventory over an already-populated bundle."""

    root = Path(bundle_root)
    document = ReproducibilityBundle(
        files=_inventory_entries(root),
        run_manifest_path=pointers.run_manifest_path,
        locked_config_path=pointers.locked_config_path,
        split_manifest_path=pointers.split_manifest_path,
        adapter_artifact_index_path=pointers.adapter_artifact_index_path,
        evaluation_contract_path=pointers.evaluation_contract_path,
        merged_artifact_index_path=pointers.merged_artifact_index_path,
    )
    destination = root / BUNDLE_INVENTORY_NAME
    publish_bytes_replace(
        destination,
        canonical_json_bytes(document.model_dump(mode="json")) + b"\n",
    )
    return destination


def _require_unique_files(files: tuple[BundleFileEntry, ...]) -> None:
    paths = tuple(entry.path for entry in files)
    if len(paths) != len(set(paths)):
        raise ValueError("bundle inventory contains duplicate file paths")


def _require_listed_pointers(document: ReproducibilityBundle) -> None:
    listed = {entry.path for entry in document.files}
    for field in _POINTER_FIELDS:
        _require_listed_pointer(listed, field, getattr(document, field))


def _require_listed_pointer(listed: set[str], field: str, pointer: str | None) -> None:
    if pointer is None:
        return
    canonical = portable_artifact_path(pointer).as_posix()
    if canonical != pointer:
        raise ValueError(f"{field} must be a canonical POSIX path: {pointer}")
    if pointer not in listed:
        raise ValueError(f"{field} is not listed in the bundle inventory: {pointer}")


def _inventory_entries(root: Path) -> tuple[BundleFileEntry, ...]:
    identities = scan_bundle(root)
    files = sorted(
        path
        for path, identity in identities.items()
        if identity.kind == "file" and path.as_posix() != BUNDLE_INVENTORY_NAME
    )
    if not files:
        raise ValueError("reproducibility bundle does not contain any files to inventory")
    descriptor = open_bundle(root)
    try:
        return tuple(_read_inventory_entry(descriptor, relative) for relative in files)
    finally:
        os.close(descriptor)


def _read_inventory_entry(root_descriptor: int, relative: PurePosixPath) -> BundleFileEntry:
    size, digest = hash_relative_file(root_descriptor, relative)
    return BundleFileEntry(path=relative.as_posix(), size_bytes=size, sha256=digest)
