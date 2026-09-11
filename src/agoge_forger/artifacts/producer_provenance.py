"""Build fail-closed ``ArtifactProducerProvenance`` for train/export writers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from ..config import ExperimentConfig
from ..eval import ArtifactProducerProvenance
from ..split_contract import bind_frozen_split, sha256_file
from ..split_schema import IMMUTABLE_REVISION_PATTERN

_REVISION_RE = re.compile(IMMUTABLE_REVISION_PATTERN)
_CANNOT_CONSTRUCT = "cannot construct producer_provenance"
_SPLIT_MANIFEST_NAME = "split_manifest.json"
PathLike = str | Path


def producer_provenance_from_config(config: ExperimentConfig) -> ArtifactProducerProvenance:
    """Derive provenance from pinned model identity and frozen train-split digests."""

    binding = _frozen_train_binding(config.dataset_path)
    return ArtifactProducerProvenance(
        base_model_name_or_path=config.model_id,
        revision=_require_immutable_revision(config.revision),
        training_split_manifest_sha256=binding.manifest_sha256,
        training_split_name="train",
        training_split_sha256=binding.split_sha256,
    )


def producer_provenance_from_adapter(adapter_path: PathLike) -> ArtifactProducerProvenance:
    """Read provenance already sealed into an adapter ``artifact_index.json``."""

    index_path = Path(adapter_path) / "artifact_index.json"
    payload = _load_artifact_index_payload(index_path)
    provenance = payload.get("producer_provenance")
    if provenance is None:
        raise ValueError(
            f"{_CANNOT_CONSTRUCT}: adapter artifact index requires producer_provenance: "
            f"{index_path}"
        )
    return ArtifactProducerProvenance.model_validate(provenance)


def require_producer_provenance(
    producer_provenance: ArtifactProducerProvenance | Mapping[str, object] | None,
    adapter_path: PathLike,
) -> ArtifactProducerProvenance:
    """Return supplied provenance, or the adapter index copy, or fail closed."""

    if producer_provenance is None:
        return producer_provenance_from_adapter(adapter_path)
    if isinstance(producer_provenance, ArtifactProducerProvenance):
        return producer_provenance
    return ArtifactProducerProvenance.model_validate(producer_provenance)


def _require_immutable_revision(revision: str | None) -> str:
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError(
            f"{_CANNOT_CONSTRUCT}: ExperimentConfig.revision must be a "
            "content-addressed digest matching ^[0-9a-f]{40,64}$"
        )
    return revision


def _frozen_train_binding(dataset_path: str):
    dataset = Path(dataset_path).expanduser().resolve(strict=True)
    manifest_path = _split_manifest_for_dataset(dataset)
    if manifest_path is None:
        raise ValueError(
            f"{_CANNOT_CONSTRUCT}: dataset_path is not a frozen train split "
            f"(no {_SPLIT_MANIFEST_NAME} beside the dataset or its parent): {dataset}"
        )
    binding = bind_frozen_split(manifest_path, "train")
    if sha256_file(dataset) != binding.split_sha256:
        raise ValueError(
            f"{_CANNOT_CONSTRUCT}: dataset_path digest does not match the frozen "
            f"train split: {dataset}"
        )
    return binding


def _split_manifest_for_dataset(dataset: Path) -> Path | None:
    for candidate in (
        dataset.parent / _SPLIT_MANIFEST_NAME,
        dataset.parent.parent / _SPLIT_MANIFEST_NAME,
    ):
        if candidate.is_file():
            return candidate
    return None


def _load_artifact_index_payload(index_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"{_CANNOT_CONSTRUCT}: adapter is missing artifact_index.json: {index_path}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"{_CANNOT_CONSTRUCT}: invalid artifact_index.json: {index_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{_CANNOT_CONSTRUCT}: invalid artifact_index.json: {index_path}")
    return payload
