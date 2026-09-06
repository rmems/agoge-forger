"""Verified record iterators and dataset adapters for frozen splits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset  # type: ignore[attr-defined]

from ._strict_json import decode_json_object
from .datasets import normalize_row
from .split_schema import SplitManifest, SplitName, sha256_bytes
from .split_validation import (
    validate_split_manifest,
    validate_split_manifest_snapshot,
    verified_split_snapshot,
)


@dataclass(frozen=True)
class FrozenSplitBinding:
    manifest_path: Path
    manifest: SplitManifest
    manifest_sha256: str
    split: SplitName
    split_sha256: str


@dataclass(frozen=True)
class _FrozenDatasetRequest:
    manifest_path: str
    split: SplitName
    manifest_sha256: str
    split_sha256: str
    tokenizer: Any


def iter_materialized_records(
    path: Path, manifest: SplitManifest, split: SplitName
) -> Iterator[dict[str, Any]]:
    artifact = manifest.splits[split]
    with verified_split_snapshot(path, split, artifact) as snapshot_path:
        yield from _iter_snapshot_records(snapshot_path, split, artifact.path)


def _iter_snapshot_records(
    snapshot_path: Path, split: SplitName, declared_path: str
) -> Iterator[dict[str, Any]]:
    with snapshot_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            yield decode_json_object(
                raw_line,
                f"{declared_path}:{line_number}",
                object_label=f"materialized {split} row",
            )


def iter_frozen_records(manifest_path: str | Path, split: SplitName) -> Iterator[dict[str, Any]]:
    """Yield raw source records from a verified frozen partition."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = validate_split_manifest(path)
    yield from iter_materialized_records(path, manifest, split)


def load_frozen_dataset(
    manifest_path: str | Path,
    split: SplitName,
    tokenizer: Any = None,
    *,
    expected_binding: FrozenSplitBinding | None = None,
) -> Dataset:
    """Training loader for a frozen split; it never derives new membership."""

    binding = bind_frozen_split(manifest_path, split)
    if expected_binding is not None:
        _require_same_binding(binding, expected_binding)
    return Dataset.from_generator(
        _generate_frozen_records,
        gen_kwargs={
            "request": _FrozenDatasetRequest(
                manifest_path=str(binding.manifest_path),
                split=split,
                manifest_sha256=binding.manifest_sha256,
                split_sha256=binding.split_sha256,
                tokenizer=tokenizer,
            )
        },
    )


def _require_same_binding(actual: FrozenSplitBinding, expected: FrozenSplitBinding) -> None:
    if actual.manifest_sha256 != expected.manifest_sha256:
        raise ValueError("frozen dataset manifest changed after training binding was established")
    if actual.split_sha256 != expected.split_sha256:
        raise ValueError("frozen dataset split changed after training binding was established")


def bind_frozen_split(manifest_path: str | Path, split: SplitName) -> FrozenSplitBinding:
    """Bind one validated manifest snapshot and selected artifact to immutable digests."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest_snapshot = path.read_bytes()
    manifest = validate_split_manifest_snapshot(path, manifest_snapshot)
    return FrozenSplitBinding(
        manifest_path=path,
        manifest=manifest,
        manifest_sha256=sha256_bytes(manifest_snapshot),
        split=split,
        split_sha256=manifest.splits[split].sha256,
    )


def _generate_frozen_records(request: _FrozenDatasetRequest) -> Iterator[dict[str, Any]]:
    path = Path(request.manifest_path)
    manifest_snapshot = path.read_bytes()
    if sha256_bytes(manifest_snapshot) != request.manifest_sha256:
        raise ValueError("frozen dataset manifest changed after cache identity was established")
    manifest = validate_split_manifest_snapshot(path, manifest_snapshot)
    if manifest.splits[request.split].sha256 != request.split_sha256:
        raise ValueError("frozen dataset split digest changed after cache identity was established")
    records = iter_materialized_records(path, manifest, request.split)
    for index, row in enumerate(records, 1):
        normalized = normalize_row(row, tokenizer=request.tokenizer, index=index)
        yield {"text": normalized["text"]}
