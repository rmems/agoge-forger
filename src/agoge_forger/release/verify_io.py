"""Descriptor I/O helpers for offline bundle verification."""

from __future__ import annotations

import errno
import os
from pathlib import Path, PurePosixPath

from .._strict_json import decode_json_object  # noinspection PyProtectedMember
from ..eval._artifact_schema import (  # noinspection PyProtectedMember
    portable_contract_reference,
)
from ..eval._descriptor_bundle import (  # noinspection PyProtectedMember
    hash_relative_file,
    open_bundle,
    read_relative_file,
)
from .report import BundleFailure
from .verify_errors import classify_scan_error, json_failure


def read_relative(root: Path, relative: PurePosixPath) -> tuple[bytes, str]:
    descriptor = open_bundle(root)
    try:
        return read_relative_file(descriptor, relative)
    finally:
        os.close(descriptor)


def hash_relative(root: Path, relative: PurePosixPath) -> tuple[int, str]:
    descriptor = open_bundle(root)
    try:
        return hash_relative_file(descriptor, relative)
    finally:
        os.close(descriptor)


def load_object(root: Path, relative: str, label: str) -> dict[str, object] | BundleFailure:
    try:
        payload, _digest = read_relative(root, PurePosixPath(relative))
        return decode_json_object(payload, str(root / relative), object_label=label)
    except ValueError as exc:
        return json_failure(exc, relative)
    except OSError as exc:
        return classify_os_error(exc, relative)


def classify_os_error(exc: OSError, path: str) -> BundleFailure:
    if isinstance(exc, FileNotFoundError) or exc.errno == errno.ENOENT:
        return BundleFailure(code="missing_file", path=path, message=str(exc))
    return BundleFailure(code="unsafe_path", path=path, message=str(exc))


def classify_io_error(exc: Exception, path: str) -> BundleFailure:
    if isinstance(exc, OSError):
        return classify_os_error(exc, path)
    if isinstance(exc, ValueError):
        message = str(exc)
        if "changed" in message.lower():
            return BundleFailure(code="modified_file", path=path, message=message)
        return classify_scan_error(exc, path)
    return BundleFailure(code="unsafe_path", path=path, message=str(exc))


def confine_reference(
    anchor: Path,
    relative: str,
    bundle_root: Path,
    contract_path: str,
) -> Path | BundleFailure:
    try:
        portable_contract_reference(relative)
        resolved = (anchor / Path(relative)).resolve(strict=True)
        root = bundle_root.resolve(strict=True)
    except OSError as exc:
        return classify_os_error(exc, contract_path)
    except ValueError as exc:
        return BundleFailure(code="path_escaping", path=contract_path, message=str(exc))
    if not resolved.is_relative_to(root):
        return BundleFailure(
            code="path_escaping",
            path=contract_path,
            message=f"{contract_path} reference escapes the bundle: {relative}",
        )
    return resolved


def resolve_existing(root: Path, relative: str, path: str) -> Path | BundleFailure:
    try:
        return (root / relative).resolve(strict=True)
    except OSError as exc:
        return classify_os_error(exc, path)
