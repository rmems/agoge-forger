"""Guarded offline Transformers loading for run-status validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from transformers.utils import cached_file

_MAX_TOKENIZER_FILES = 128
_MAX_TOKENIZER_INVENTORY_ENTRIES = 1024
_MAX_TOKENIZER_FILE_BYTES = 64 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 128 * 1024 * 1024
_MAX_TOKENIZER_CONFIG_BYTES = 4 * 1024 * 1024
_TOKENIZER_FILE_SUFFIXES = frozenset({".codes", ".jinja", ".json", ".model", ".tokenizer", ".txt"})


def offline_pretrained(
    factory: Any,
    source: str | Path,
    *,
    revision: str | None = None,
) -> Any:
    loader = getattr(factory, "from_pretrained", None)
    # The `is None` arm is spelled out separately from `callable(...)` so static
    # analysis can narrow the optional away before the call below. Qodana's
    # PyCallingNonCallable does not narrow through `callable()` on its own and
    # reports the call as "'None' object is not callable" without it.
    if loader is None or not callable(loader):
        raise TypeError("from_pretrained is not callable")
    revision_kwarg = {} if revision is None else {"revision": revision}
    return loader(
        source,
        **revision_kwarg,
        local_files_only=True,
        trust_remote_code=False,
    )


def _local_inventory(source: Path) -> tuple[Path, Path, bool] | None:
    if source.is_symlink() or not source.is_dir():
        return None
    resolved = source.resolve(strict=True)
    return resolved, resolved, False


def _cached_snapshot(candidate: str | Path, revision: str | None) -> tuple[Path, Path, bool] | None:
    source = Path(candidate)
    if source.exists() or source.is_symlink():
        return _local_inventory(source)
    config = cached_file(
        candidate,
        "config.json",
        revision=revision,
        local_files_only=True,
    )
    if config is None:
        return None
    snapshot = Path(config).parent
    snapshots = snapshot.parent
    if any((snapshots.name != "snapshots", snapshot.is_symlink(), not snapshot.is_dir())):
        return None
    boundary = snapshots.parent.resolve(strict=True)
    resolved_snapshot = snapshot.resolve(strict=True)
    if not resolved_snapshot.is_relative_to(boundary):
        return None
    return resolved_snapshot, boundary, True


def _trusted_inventory_file(path: Path, boundary: Path, *, allow_symlink: bool) -> Path | None:
    if path.is_symlink():
        if not allow_symlink:
            return None
        resolved = path.resolve(strict=True)
        trusted = all((resolved.is_file(), resolved.is_relative_to(boundary)))
        return resolved if trusted else None
    return path if path.is_file() else None


def _inventory_files(
    root: Path, boundary: Path, *, allow_symlink: bool
) -> list[tuple[Path, Path]] | None:
    files = []
    for entry_count, path in enumerate(root.rglob("*"), start=1):
        if entry_count > _MAX_TOKENIZER_INVENTORY_ENTRIES:
            return None
        if (path.is_dir(), path.is_symlink()) == (True, False):
            continue
        trusted = _trusted_inventory_file(path, boundary, allow_symlink=allow_symlink)
        if trusted is None:
            return None
        files.append((path, trusted))
    return files


def _tokenizer_files(files: list[tuple[Path, Path]]) -> list[Path]:
    return [trusted for path, trusted in files if path.suffix.lower() in _TOKENIZER_FILE_SUFFIXES]


def _trusted_tokenizer_reference(
    value: object,
    root: Path,
    boundary: Path,
    *,
    allow_symlink: bool,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return _trusted_inventory_file(root / relative, boundary, allow_symlink=allow_symlink)


def _configured_tokenizer_paths(config: dict[str, object]) -> list[object] | None:
    paths = [
        value for field, value in config.items() if field.endswith("_file") and value is not None
    ]
    fast_files = config.get("fast_tokenizer_files")
    if fast_files is None:
        return paths
    if not isinstance(fast_files, list):
        return None
    paths.extend(fast_files)
    return paths


def _load_tokenizer_config(
    config_path: Path,
    boundary: Path,
    *,
    allow_symlink: bool,
) -> dict[str, object] | None:
    trusted_config = _trusted_inventory_file(config_path, boundary, allow_symlink=allow_symlink)
    if trusted_config is None or trusted_config.stat().st_size > _MAX_TOKENIZER_CONFIG_BYTES:
        return None
    with trusted_config.open("rb") as handle:
        encoded = handle.read(_MAX_TOKENIZER_CONFIG_BYTES + 1)
    if len(encoded) > _MAX_TOKENIZER_CONFIG_BYTES:
        return None
    try:
        config = json.loads(encoded)
    except (MemoryError, RecursionError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _resolve_tokenizer_references(
    config: dict[str, object],
    root: Path,
    boundary: Path,
    *,
    allow_symlink: bool,
) -> list[Path] | None:
    references = []
    configured_paths = _configured_tokenizer_paths(config)
    if configured_paths is None:
        return None
    for value in configured_paths:
        trusted = _trusted_tokenizer_reference(value, root, boundary, allow_symlink=allow_symlink)
        if trusted is None:
            return None
        references.append(trusted)
    return references


def _tokenizer_config_references(
    root: Path,
    boundary: Path,
    *,
    allow_symlink: bool,
) -> list[Path] | None:
    config_path = root / "tokenizer_config.json"
    if not config_path.exists() and not config_path.is_symlink():
        return []
    config = _load_tokenizer_config(config_path, boundary, allow_symlink=allow_symlink)
    if config is None:
        return None
    return _resolve_tokenizer_references(config, root, boundary, allow_symlink=allow_symlink)


def _inventory_size_usable(files: list[Path]) -> bool:
    sizes = [path.stat().st_size for path in files]
    return all(
        (
            len(files) <= _MAX_TOKENIZER_FILES,
            all(size <= _MAX_TOKENIZER_FILE_BYTES for size in sizes),
            sum(sizes) <= _MAX_TOKENIZER_BYTES,
        )
    )


def _tokenizer_inventory_usable(
    candidate: str | Path,
    *,
    revision: str | None,
) -> bool:
    location = _cached_snapshot(candidate, revision)
    if location is None:
        return False
    root, boundary, allow_symlink = location
    files = _inventory_files(root, boundary, allow_symlink=allow_symlink)
    if files is None:
        return False
    references = _tokenizer_config_references(root, boundary, allow_symlink=allow_symlink)
    if references is None:
        return False
    tokenizer_files = _tokenizer_files(files)
    tokenizer_files.extend(path for path in references if path not in tokenizer_files)
    return _inventory_size_usable(tokenizer_files)


def tokenizer_usable(candidate: str | Path, *, revision: str | None = None) -> bool:
    try:
        if not _tokenizer_inventory_usable(candidate, revision=revision):
            return False
        tokenizer = offline_pretrained(  # nosec B615
            AutoTokenizer,
            candidate,
            revision=revision,
        )
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return all(
        (
            len(tokenizer) > 0,
            any(token is not None for token in (tokenizer.pad_token, tokenizer.eos_token)),
        )
    )
