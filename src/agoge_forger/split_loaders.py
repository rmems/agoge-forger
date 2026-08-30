"""Verified record iterators and dataset adapters for frozen splits."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset  # type: ignore[attr-defined]

from .datasets import normalize_row
from .split_schema import SplitManifest, SplitName
from .split_validation import resolve_split_path, validate_split_manifest


def iter_materialized_records(
    path: Path, manifest: SplitManifest, split: SplitName
) -> Iterator[dict[str, Any]]:
    artifact_path = resolve_split_path(path, manifest.splits[split])
    with artifact_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"materialized {split} row is not an object")
            yield row


def iter_frozen_records(manifest_path: str | Path, split: SplitName) -> Iterator[dict[str, Any]]:
    """Yield raw source records from a verified frozen partition."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = validate_split_manifest(path)
    yield from iter_materialized_records(path, manifest, split)


def load_frozen_dataset(
    manifest_path: str | Path, split: SplitName, tokenizer: Any = None
) -> Dataset:
    """Training loader for a frozen split; it never derives new membership."""

    def generate() -> Iterator[dict[str, Any]]:
        for index, row in enumerate(iter_frozen_records(manifest_path, split), 1):
            yield normalize_row(row, tokenizer=tokenizer, index=index)

    return Dataset.from_generator(generate)
