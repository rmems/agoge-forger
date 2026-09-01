"""Architecture-neutral validation for paired held-out evaluation contracts.

This module freezes comparability metadata only. It intentionally performs no
model loading, inference, scoring, or result generation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..split_contract import (
    SplitManifest,
    canonical_json_bytes,
    sha256_bytes,
)
from ..split_validation import validate_split_manifest_snapshot
from . import _artifact_schema, _artifact_validation

ArtifactIndex = _artifact_schema.ArtifactIndex
ArtifactIndexEntry = _artifact_schema.ArtifactIndexEntry
ArtifactIndexReference = _artifact_schema.ArtifactIndexReference
ArtifactKind = _artifact_schema.ArtifactKind
ArtifactValidationContext = _artifact_schema.ArtifactValidationContext
FrozenEvaluationModel = _artifact_schema.FrozenEvaluationModel
IndexedArtifacts = _artifact_schema.IndexedArtifacts
require_artifact_index = _artifact_validation.require_artifact_index

EVALUATION_CONTRACT_VERSION: Literal["agoge.evaluation-contract.v1"] = (
    "agoge.evaluation-contract.v1"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMPARABLE_ARM_FIELDS = (
    "model_repository",
    "model_revision",
    "tokenizer_repository",
    "tokenizer_revision",
    "serializer_id",
    "serializer_version",
    "serializer_sha256",
    "logical_task_set_sha256",
    "context_window",
    "truncation_policy",
    "decoding",
    "scoring_version",
)


class DecodingContract(FrozenEvaluationModel):
    do_sample: bool = False
    seed: int
    max_new_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0, allow_inf_nan=False)
    top_p: float = Field(gt=0, le=1)


class EvaluationArm(FrozenEvaluationModel):
    role: Literal["causal_base", "causal_sft"]
    model_repository: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    artifact: ArtifactIndexReference | None = None
    tokenizer_repository: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=_SHA256_PATTERN)
    logical_task_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_window: int = Field(ge=1)
    truncation_policy: Literal["reject", "mark_unsupported", "left", "right"]
    decoding: DecodingContract
    scoring_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_sft_artifact(self) -> EvaluationArm:
        if self.role == "causal_sft" and self.artifact is None:
            raise ValueError("causal_sft arm requires a verified artifact-index reference")
        if self.role == "causal_base" and self.artifact is not None:
            raise ValueError("causal_base arm cannot reference a trained artifact")
        return self


class PairedEvaluationContract(FrozenEvaluationModel):
    schema_version: Literal["agoge.evaluation-contract.v1"] = EVALUATION_CONTRACT_VERSION
    split_manifest_path: str = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    held_out_split_sha256: str = Field(pattern=_SHA256_PATTERN)
    logical_task_ids: tuple[str, ...] = Field(min_length=1)
    logical_task_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    base: EvaluationArm
    sft: EvaluationArm

    @model_validator(mode="after")
    def require_pair_comparability(self) -> PairedEvaluationContract:
        _require_arm_roles(self.base, self.sft)
        _require_task_identity(self)
        _require_no_arm_drift(self.base, self.sft)
        return self


def _require_arm_roles(base: EvaluationArm, sft: EvaluationArm) -> None:
    if base.role != "causal_base" or sft.role != "causal_sft":
        raise ValueError("paired contract requires causal_base and causal_sft arms")


def _require_task_identity(contract: PairedEvaluationContract) -> None:
    if len(set(contract.logical_task_ids)) != len(contract.logical_task_ids):
        raise ValueError("logical_task_ids must be unique")
    expected_digest = logical_task_set_sha256(contract.logical_task_ids)
    if contract.logical_task_set_sha256 != expected_digest:
        raise ValueError("logical task-set digest does not match logical_task_ids")
    if contract.base.logical_task_set_sha256 != contract.logical_task_set_sha256:
        raise ValueError("paired arms do not reference the contract logical task set")


def _require_no_arm_drift(base: EvaluationArm, sft: EvaluationArm) -> None:
    drift = [
        field for field in COMPARABLE_ARM_FIELDS if getattr(base, field) != getattr(sft, field)
    ]
    if drift:
        raise ValueError(f"paired evaluation arms are non-comparable; drift in: {drift}")


def logical_task_set_sha256(task_ids: tuple[str, ...] | list[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(task_ids)))


def held_out_task_ids(manifest: SplitManifest) -> tuple[str, ...]:
    return tuple(member.canonical_id for member in manifest.splits["held_out"].members)


def load_evaluation_contract(contract_path: str | Path) -> PairedEvaluationContract:
    path = Path(contract_path).expanduser().resolve(strict=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid evaluation contract JSON: {path}") from exc
    return PairedEvaluationContract.model_validate(value)


def validate_evaluation_contract(contract_path: str | Path) -> PairedEvaluationContract:
    """Validate pair comparability and its immutable #99 held-out reference."""

    path = Path(contract_path).expanduser().resolve(strict=True)
    contract = load_evaluation_contract(path)
    manifest = _validate_manifest_reference(path, contract)
    _validate_held_out_reference(contract, manifest)
    _validate_sft_artifact(path, contract.sft, manifest, contract.split_manifest_sha256)
    return contract


def _validate_manifest_reference(
    contract_path: Path,
    contract: PairedEvaluationContract,
) -> SplitManifest:
    manifest_path = (contract_path.parent / contract.split_manifest_path).resolve(strict=True)
    manifest_snapshot = manifest_path.read_bytes()
    if sha256_bytes(manifest_snapshot) != contract.split_manifest_sha256:
        raise ValueError("evaluation contract split-manifest SHA-256 mismatch")
    return validate_split_manifest_snapshot(manifest_path, manifest_snapshot)


def _validate_held_out_reference(
    contract: PairedEvaluationContract,
    manifest: SplitManifest,
) -> None:
    if manifest.splits["held_out"].sha256 != contract.held_out_split_sha256:
        raise ValueError("evaluation contract held-out split SHA-256 mismatch")
    expected_ids = held_out_task_ids(manifest)
    if contract.logical_task_ids != expected_ids:
        raise ValueError("evaluation contract task IDs differ from the frozen held-out manifest")
    if logical_task_set_sha256(expected_ids) != contract.logical_task_set_sha256:
        raise ValueError("evaluation contract task-set digest differs from held-out membership")


def build_evaluation_contract(
    *,
    manifest_path: str | Path,
    contract_path: str | Path,
    base: EvaluationArm,
    sft: EvaluationArm,
) -> PairedEvaluationContract:
    """Build and exclusively write a paired contract without evaluating models."""

    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    destination = Path(contract_path).expanduser()
    validated_base = EvaluationArm.model_validate(base.model_dump(mode="json"))
    validated_sft = EvaluationArm.model_validate(sft.model_dump(mode="json"))
    manifest_snapshot = manifest_file.read_bytes()
    manifest = validate_split_manifest_snapshot(manifest_file, manifest_snapshot)
    task_ids = held_out_task_ids(manifest)
    task_digest = logical_task_set_sha256(task_ids)
    manifest_digest = sha256_bytes(manifest_snapshot)
    normalized_sft = _normalize_sft_artifact(validated_sft, destination, manifest, manifest_digest)
    contract = PairedEvaluationContract(
        split_manifest_path=_portable_relative_path(manifest_file, destination.parent.resolve()),
        split_manifest_sha256=manifest_digest,
        held_out_split_sha256=manifest.splits["held_out"].sha256,
        logical_task_ids=task_ids,
        logical_task_set_sha256=task_digest,
        base=validated_base,
        sft=normalized_sft,
    )
    payload = canonical_json_bytes(contract.model_dump(mode="json")) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite evaluation contract: {destination}") from exc
    return contract


def _normalize_sft_artifact(
    sft: EvaluationArm,
    destination: Path,
    manifest: SplitManifest,
    manifest_digest: str,
) -> EvaluationArm:
    artifact, context = _sft_artifact_validation(sft, manifest, manifest_digest)
    anchor = destination.parent.resolve()
    supplied = Path(artifact.artifact_index_path).expanduser()
    artifact_path = (
        supplied.resolve(strict=True)
        if supplied.is_absolute()
        else (anchor / supplied).resolve(strict=True)
    )
    _require_contract_outside_artifact_bundle(destination, artifact_path)
    require_artifact_index(artifact_path, artifact.artifact_index_sha256, context)
    normalized = ArtifactIndexReference.model_validate(
        {
            **artifact.model_dump(mode="json"),
            "artifact_index_path": _portable_relative_path(artifact_path, anchor),
        }
    )
    return EvaluationArm.model_validate(
        {**sft.model_dump(mode="json"), "artifact": normalized.model_dump(mode="json")}
    )


def _require_contract_outside_artifact_bundle(
    destination: Path,
    artifact_index_path: Path,
) -> None:
    artifact_root = artifact_index_path.parent.resolve(strict=True)
    if destination.resolve().is_relative_to(artifact_root):
        raise ValueError("evaluation contract cannot be written inside its artifact bundle")


def _validate_sft_artifact(
    contract_path: Path,
    sft: EvaluationArm,
    manifest: SplitManifest,
    manifest_digest: str,
) -> None:
    artifact, context = _sft_artifact_validation(sft, manifest, manifest_digest)
    artifact_path = (contract_path.parent / artifact.artifact_index_path).resolve(strict=True)
    require_artifact_index(artifact_path, artifact.artifact_index_sha256, context)


def _sft_artifact_validation(
    sft: EvaluationArm,
    manifest: SplitManifest,
    manifest_digest: str,
) -> tuple[ArtifactIndexReference, ArtifactValidationContext]:
    artifact = sft.artifact
    if artifact is None:
        raise ValueError("causal_sft arm requires a verified artifact-index reference")
    context = ArtifactValidationContext(
        kind=artifact.kind,
        model_repository=sft.model_repository,
        model_revision=sft.model_revision,
        split_manifest_sha256=manifest_digest,
        train_split_sha256=manifest.splits["train"].sha256,
    )
    return artifact, context


def _portable_relative_path(path: Path, anchor: Path) -> str:
    return os.path.relpath(path, anchor).replace("\\", "/")
