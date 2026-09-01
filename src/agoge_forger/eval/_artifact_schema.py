"""Canonical schema types for immutable evaluation artifact bundles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ArtifactKind = Literal["peft_adapter", "merged_model"]

_ADAPTER_CONFIG_PATH = PurePosixPath("adapter_config.json")
_ADAPTER_WEIGHTS_PATH = PurePosixPath("adapter_model.safetensors")
_MERGED_CONFIG_PATH = PurePosixPath("config.json")
_MERGED_WEIGHTS_PATH = PurePosixPath("model.safetensors")
_MERGED_WEIGHTS_INDEX_PATH = PurePosixPath("model.safetensors.index.json")
_MODEL_SHARD_PATTERN = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FrozenEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactIndexEntry(FrozenEvaluationModel):
    file: str = Field(min_length=1)
    size_bytes: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class ArtifactProducerProvenance(FrozenEvaluationModel):
    base_model_name_or_path: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    training_split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    training_split_name: Literal["train"]
    training_split_sha256: str = Field(pattern=_SHA256_PATTERN)


class ArtifactIndex(FrozenEvaluationModel):
    output_dir: str = Field(min_length=1)
    artifacts: tuple[ArtifactIndexEntry, ...] = Field(min_length=1)
    producer_provenance: ArtifactProducerProvenance | None = None

    @model_validator(mode="after")
    def require_unique_paths(self) -> ArtifactIndex:
        paths = tuple(entry.file for entry in self.artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("artifact index contains duplicate file paths")
        return self


class ArtifactIndexReference(FrozenEvaluationModel):
    kind: ArtifactKind
    artifact_index_path: str = Field(min_length=1)
    artifact_index_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True)
class ArtifactValidationContext:
    kind: ArtifactKind
    model_repository: str
    model_revision: str
    split_manifest_sha256: str
    train_split_sha256: str


IndexedArtifacts = dict[PurePosixPath, tuple[ArtifactIndexEntry, Path]]


def _parse_artifact_index(path: Path, payload: bytes) -> ArtifactIndex:
    try:
        return ArtifactIndex.model_validate(
            json.loads(payload, object_pairs_hook=_unique_json_object)
        )
    except ValueError as exc:
        raise ValueError(f"invalid SFT artifact index: {path}") from exc


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _portable_artifact_path(value: str) -> PurePosixPath:
    if not value.strip():
        raise ValueError(f"artifact index path must stay relative to its directory: {value}")
    portable = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if _is_unsafe_portable_path(portable, windows):
        raise ValueError(f"artifact index path must stay relative to its directory: {value}")
    return portable


def _is_unsafe_portable_path(portable: PurePosixPath, windows: PureWindowsPath) -> bool:
    return (
        portable.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in portable.parts
    )


def _require_canonical_model_shard_set(shards: set[PurePosixPath]) -> None:
    parsed = [_parse_model_shard_name(shard) for shard in shards]
    total = _require_common_shard_total(parsed)
    _require_contiguous_shard_numbers(parsed, total)


def _parse_model_shard_name(shard: PurePosixPath) -> tuple[int, int]:
    match = _MODEL_SHARD_PATTERN.fullmatch(shard.as_posix())
    if match is None:
        raise ValueError("merged-model shard names must use model-NNNNN-of-NNNNN.safetensors")
    return int(match.group(1)), int(match.group(2))


def _require_common_shard_total(parsed: list[tuple[int, int]]) -> int:
    totals = {total for _, total in parsed}
    if len(totals) != 1:
        raise ValueError("merged-model shard names disagree on their total shard count")
    return next(iter(totals))


def _require_contiguous_shard_numbers(parsed: list[tuple[int, int]], total: int) -> None:
    numbers = {number for number, _ in parsed}
    if total < 1 or numbers != set(range(1, total + 1)):
        raise ValueError("merged-model shard set must be complete and contiguous")
