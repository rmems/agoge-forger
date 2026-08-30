"""Architecture-neutral validation for paired held-out evaluation contracts.

This module freezes comparability metadata only. It intentionally performs no
model loading, inference, scoring, or result generation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..split_contract import (
    SplitManifest,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_split_manifest,
)

EVALUATION_CONTRACT_VERSION: Literal["agoge.evaluation-contract.v1"] = (
    "agoge.evaluation-contract.v1"
)

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


class FrozenEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecodingContract(FrozenEvaluationModel):
    do_sample: bool = False
    seed: int
    max_new_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)


class ArtifactIndexReference(FrozenEvaluationModel):
    kind: Literal["peft_adapter", "merged_model"]
    artifact_index_path: str = Field(min_length=1)
    artifact_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationArm(FrozenEvaluationModel):
    role: Literal["causal_base", "causal_sft"]
    model_repository: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    artifact: ArtifactIndexReference | None = None
    tokenizer_repository: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_task_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    held_out_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_task_ids: tuple[str, ...] = Field(min_length=1)
    logical_task_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
    _validate_sft_artifact(path, contract.sft)
    return contract


def _validate_manifest_reference(
    contract_path: Path, contract: PairedEvaluationContract
) -> SplitManifest:
    manifest_path = (contract_path.parent / contract.split_manifest_path).resolve(strict=True)
    if sha256_file(manifest_path) != contract.split_manifest_sha256:
        raise ValueError("evaluation contract split-manifest SHA-256 mismatch")
    return validate_split_manifest(manifest_path)


def _validate_held_out_reference(
    contract: PairedEvaluationContract, manifest: SplitManifest
) -> None:
    if manifest.splits["held_out"].sha256 != contract.held_out_split_sha256:
        raise ValueError("evaluation contract held-out split SHA-256 mismatch")
    expected_ids = held_out_task_ids(manifest)
    if contract.logical_task_ids != expected_ids:
        raise ValueError("evaluation contract task IDs differ from the frozen held-out manifest")
    if logical_task_set_sha256(expected_ids) != contract.logical_task_set_sha256:
        raise ValueError("evaluation contract task-set digest differs from held-out membership")


def build_evaluation_contract(
    *, manifest_path: str | Path, contract_path: str | Path, base: EvaluationArm, sft: EvaluationArm
) -> PairedEvaluationContract:
    """Build and exclusively write a paired contract without evaluating models."""

    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    destination = Path(contract_path).expanduser()
    manifest = validate_split_manifest(manifest_file)
    task_ids = held_out_task_ids(manifest)
    task_digest = logical_task_set_sha256(task_ids)
    normalized_sft = _normalize_sft_artifact(sft, destination)
    contract = PairedEvaluationContract(
        split_manifest_path=str(Path(os.path.relpath(manifest_file, destination.parent.resolve()))),
        split_manifest_sha256=sha256_file(manifest_file),
        held_out_split_sha256=manifest.splits["held_out"].sha256,
        logical_task_ids=task_ids,
        logical_task_set_sha256=task_digest,
        base=base,
        sft=normalized_sft,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(canonical_json_bytes(contract.model_dump(mode="json")) + b"\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite evaluation contract: {destination}") from exc
    return contract


def _normalize_sft_artifact(sft: EvaluationArm, destination: Path) -> EvaluationArm:
    if sft.artifact is None:
        raise ValueError("causal_sft arm requires a verified artifact-index reference")
    anchor = destination.parent.resolve()
    supplied = Path(sft.artifact.artifact_index_path).expanduser()
    artifact_path = (
        supplied.resolve(strict=True)
        if supplied.is_absolute()
        else (anchor / supplied).resolve(strict=True)
    )
    _require_artifact_digest(artifact_path, sft.artifact.artifact_index_sha256)
    normalized = sft.artifact.model_copy(
        update={"artifact_index_path": str(Path(os.path.relpath(artifact_path, anchor)))}
    )
    return sft.model_copy(update={"artifact": normalized})


def _validate_sft_artifact(contract_path: Path, sft: EvaluationArm) -> None:
    if sft.artifact is None:
        raise ValueError("causal_sft arm requires a verified artifact-index reference")
    artifact_path = (contract_path.parent / sft.artifact.artifact_index_path).resolve(strict=True)
    _require_artifact_digest(artifact_path, sft.artifact.artifact_index_sha256)


def _require_artifact_digest(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SFT artifact-index SHA-256 mismatch: expected {expected}, found {actual}"
        )
