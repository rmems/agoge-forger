"""Artifact-index integrity checks used by run-status."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _load_index(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_relative_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return value


def _artifact_metadata(item: Any) -> tuple[int, str] | None:
    if not isinstance(item, dict):
        return None
    size, digest = item.get("size_bytes"), item.get("sha256")
    if not isinstance(size, int):
        return None
    if isinstance(size, bool):
        return None
    if not isinstance(digest, str):
        return None
    return size, digest


def _parse_artifact_entry(item: Any) -> tuple[str, int, str] | None:
    name = _safe_relative_name(item.get("file")) if isinstance(item, dict) else None
    metadata = _artifact_metadata(item)
    if name is None or metadata is None:
        return None
    return name, *metadata


def _unique_entries(parsed: list[tuple[str, int, str]]) -> dict[str, tuple[int, str]] | None:
    entries = {name: (size, digest) for name, size, digest in parsed}
    return entries if len(entries) == len(parsed) else None


def _artifact_entries(index: dict[str, Any]) -> dict[str, tuple[int, str]] | None:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return None
    parsed = [_parse_artifact_entry(item) for item in artifacts]
    if any(item is None for item in parsed):
        return None
    return _unique_entries([item for item in parsed if item is not None])


def _current_artifacts(candidate: Path) -> dict[str, Path] | None:
    paths: dict[str, Path] = {}
    for path in candidate.rglob("*"):
        if path.name == "artifact_index.json" or not path.is_file():
            continue
        if path.is_symlink():
            return None
        paths[str(path.relative_to(candidate))] = path
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_matches(path: Path, expected: tuple[int, str]) -> bool:
    size, digest = expected
    return path.stat().st_size == size and _sha256(path) == digest


def _artifact_names_match(
    entries: dict[str, tuple[int, str]],
    current: dict[str, Path],
) -> bool:
    return entries.keys() == current.keys()


def artifact_index_usable(candidate: Path) -> bool:
    """Verify the post-tokenizer completion index against the current tree."""
    index = _load_index(candidate / "artifact_index.json")
    if index is None:
        return False
    entries = _artifact_entries(index)
    current = _current_artifacts(candidate)
    if entries is None:
        return False
    if current is None:
        return False
    if not _artifact_names_match(entries, current):
        return False
    return all(_artifact_matches(path, entries[name]) for name, path in current.items())
