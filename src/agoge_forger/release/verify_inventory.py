"""Inventory membership and streamed hash checks."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .._strict_json import decode_json_object  # noinspection PyProtectedMember
from ..artifacts.safetensors_io import UNSAFE_WEIGHT_PATTERNS
from ..eval._descriptor_bundle import (  # noinspection PyProtectedMember
    EntryIdentity,
    hash_relative_file,
    open_bundle,
)
from .report import BundleFailure
from .schema import BUNDLE_INVENTORY_NAME, BUNDLE_SCHEMA_VERSION, ReproducibilityBundle
from .verify_errors import json_failure, schema_failure, unknown_schema
from .verify_io import classify_io_error, read_relative


def load_inventory(root: Path) -> tuple[ReproducibilityBundle, bytes] | BundleFailure:
    relative = PurePosixPath(BUNDLE_INVENTORY_NAME)
    try:
        payload, _digest = read_relative(root, relative)
        value = decode_json_object(
            payload,
            str(root / BUNDLE_INVENTORY_NAME),
            object_label="reproducibility bundle",
        )
    except ValueError as exc:
        return json_failure(exc, BUNDLE_INVENTORY_NAME)
    except OSError as exc:
        return classify_io_error(exc, BUNDLE_INVENTORY_NAME)
    version = value.get("schema_version")
    if version != BUNDLE_SCHEMA_VERSION:
        return unknown_schema(
            BUNDLE_INVENTORY_NAME,
            f"unknown reproducibility-bundle schema version: {version!r}",
        )
    try:
        return ReproducibilityBundle.model_validate(value), payload
    except ValidationError as exc:
        return schema_failure(exc, BUNDLE_INVENTORY_NAME)


def membership_failures(
    identities: dict[PurePosixPath, EntryIdentity],
    document: ReproducibilityBundle,
) -> list[BundleFailure]:
    expected = {PurePosixPath(entry.path) for entry in document.files}
    actual = {
        path
        for path, identity in identities.items()
        if identity.kind == "file" and path.as_posix() != BUNDLE_INVENTORY_NAME
    }
    failures = [
        BundleFailure(
            code="missing_file",
            path=str(path),
            message="listed file is missing from the bundle",
        )
        for path in sorted(expected - actual, key=str)
    ]
    failures.extend(
        BundleFailure(
            code="extra_file",
            path=str(path),
            message="file is not listed in the bundle inventory",
        )
        for path in sorted(actual - expected, key=str)
    )
    return failures


def hash_failures(root: Path, document: ReproducibilityBundle) -> list[BundleFailure]:
    failures: list[BundleFailure] = []
    descriptor = open_bundle(root)
    try:
        for entry in document.files:
            failures.extend(_hash_entry(descriptor, entry.path, entry.size_bytes, entry.sha256))
    finally:
        os.close(descriptor)
    return failures


def unsafe_weight_failures(document: ReproducibilityBundle) -> list[BundleFailure]:
    return [
        BundleFailure(
            code="unsafe_path",
            path=entry.path,
            message="bundle contains an unsafe serialized weight file",
        )
        for entry in document.files
        if _is_unsafe_weight_name(PurePosixPath(entry.path).name)
    ]


def identity_drift_failures(
    root: Path,
    before: dict[PurePosixPath, EntryIdentity],
    after: dict[PurePosixPath, EntryIdentity],
) -> list[BundleFailure]:
    if before == after:
        return []
    return [
        BundleFailure(
            code="modified_file",
            path=str(root),
            message="bundle entries changed during verification",
        )
    ]


def _hash_entry(
    descriptor: int,
    relative: str,
    expected_size: int,
    expected_digest: str,
) -> list[BundleFailure]:
    try:
        size, digest = hash_relative_file(descriptor, PurePosixPath(relative))
    except (OSError, ValueError) as exc:
        return [classify_io_error(exc, relative)]
    failures: list[BundleFailure] = []
    if size != expected_size:
        failures.append(
            BundleFailure(
                code="modified_file",
                path=relative,
                message=f"size mismatch: expected {expected_size} bytes, found {size}",
            )
        )
    if digest != expected_digest:
        failures.append(
            BundleFailure(
                code="modified_file",
                path=relative,
                message=f"SHA-256 mismatch: expected {expected_digest}, found {digest}",
            )
        )
    return failures


def _is_unsafe_weight_name(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in UNSAFE_WEIGHT_PATTERNS)
