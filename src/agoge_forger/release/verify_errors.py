"""Shared failure constructors for offline bundle verification."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from .report import BundleFailure, BundleVerificationReport, sort_failures


def json_failure(exc: ValueError, path: str) -> BundleFailure:
    return BundleFailure(code="invalid_json", path=path, message=str(exc))


def schema_failure(exc: ValidationError, path: str) -> BundleFailure:
    message = str(exc)
    lowered = message.lower()
    if "duplicate file paths" in lowered:
        return BundleFailure(code="duplicate_path", path=path, message=message)
    if "must stay relative" in lowered or "canonical posix" in lowered:
        return BundleFailure(code="path_escaping", path=path, message=message)
    if "schema version" in lowered:
        return BundleFailure(code="unknown_schema_version", path=path, message=message)
    return BundleFailure(code="invalid_json", path=path, message=message)


def index_parse_failure(path: str, exc: ValueError) -> BundleFailure:
    message = str(exc)
    lowered = message.lower()
    if "duplicate" in lowered:
        return BundleFailure(code="duplicate_path", path=path, message=message)
    if "invalid json" in lowered:
        return BundleFailure(code="invalid_json", path=path, message=message)
    if isinstance(exc, ValidationError):
        return schema_failure(exc, path)
    return BundleFailure(code="artifact_index", path=path, message=message)


def classify_scan_error(exc: ValueError, path: str | None = None) -> BundleFailure:
    message = str(exc)
    if "symlink" in message.lower():
        return BundleFailure(code="unsafe_symlink", path=path, message=message)
    if "escape" in message.lower() or ".." in message:
        return BundleFailure(code="path_escaping", path=path, message=message)
    return BundleFailure(code="unsafe_path", path=path, message=message)


def unknown_schema(path: str, message: str) -> BundleFailure:
    return BundleFailure(code="unknown_schema_version", path=path, message=message)


def fatal_report(root: Path, failure: BundleFailure) -> BundleVerificationReport:
    return make_report(root, None, [failure])


def make_report(
    root: Path,
    bundle_schema_version: str | None,
    failures: list[BundleFailure],
) -> BundleVerificationReport:
    ordered = sort_failures(failures)
    return BundleVerificationReport(
        verdict="pass" if not ordered else "fail",
        bundle_path=str(root),
        bundle_schema_version=bundle_schema_version,
        failures=ordered,
    )
