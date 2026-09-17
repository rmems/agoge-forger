"""Pre-registered experiment freeze for base-before-SFT measured runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, field_validator, model_validator

from .._atomic_file import publish_bytes_noreplace, write_fsynced_bytes
from .._strict_json import decode_json_object
from ..split_contract import SplitManifest, canonical_json_bytes, sha256_bytes, sha256_file
from ..split_validation import validate_split_manifest_snapshot
from ._artifact_schema import FrozenEvaluationModel, portable_contract_reference
from .contract import (
    DecodingContract,
    EvaluationArm,
    held_out_task_ids,
    logical_task_set_sha256,
)

EXPERIMENT_CONTRACT_VERSION: Literal["agoge.experiment-contract.v1"] = (
    "agoge.experiment-contract.v1"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REVISION_PATTERN = r"^[0-9a-f]{40,64}$"


class DatasetProvenance(FrozenEvaluationModel):
    factory_repository: str = Field(min_length=1)
    factory_revision: str = Field(pattern=_REVISION_PATTERN)
    export_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    agoge_jsonl_sha256: str = Field(pattern=_SHA256_PATTERN)


class SplitPin(FrozenEvaluationModel):
    split_manifest_path: str = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    train_split_sha256: str = Field(pattern=_SHA256_PATTERN)
    validation_split_sha256: str = Field(pattern=_SHA256_PATTERN)
    held_out_split_sha256: str = Field(pattern=_SHA256_PATTERN)
    train_record_count: int = Field(ge=0)
    validation_record_count: int = Field(ge=0)
    held_out_record_count: int = Field(ge=0)
    split_seed: int
    split_salt: str = Field(min_length=1)
    train_weight: int = Field(ge=0)
    validation_weight: int = Field(ge=0)
    held_out_weight: int = Field(ge=0)


class ModelPin(FrozenEvaluationModel):
    model_repository: str = Field(min_length=1)
    model_revision: str = Field(pattern=_REVISION_PATTERN)
    trust_remote_code: bool = False


class SerializerPin(FrozenEvaluationModel):
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=_SHA256_PATTERN)


class TrainingPin(FrozenEvaluationModel):
    completion_only_loss: bool = True
    dataset_text_field: Literal["text"] = "text"
    max_seq_length: int = Field(ge=1)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["bfloat16"] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    target_modules: tuple[str, ...] = Field(min_length=1)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    num_train_epochs: int = Field(ge=1)
    gradient_checkpointing: bool = True
    seed: int
    loss_type: Literal["nll", "chunked_nll"] = "nll"
    activation_offloading: bool = False


class EvaluationPin(FrozenEvaluationModel):
    context_window: int = Field(ge=1)
    max_new_tokens: int = Field(ge=1)
    decoding: DecodingContract
    scoring_version: str = Field(min_length=1)
    truncation_policy: Literal["reject", "mark_unsupported"]


class ExperimentContract(FrozenEvaluationModel):
    schema_version: Literal["agoge.experiment-contract.v1"] = EXPERIMENT_CONTRACT_VERSION
    experiment_id: str = Field(min_length=1)
    agoge_git_sha: str = Field(pattern=_REVISION_PATTERN)
    model: ModelPin
    dataset: DatasetProvenance
    split: SplitPin
    serializer: SerializerPin
    training: TrainingPin
    evaluation: EvaluationPin

    @field_validator("split", mode="before")
    @classmethod
    def normalize_split_manifest_path(cls, value: object) -> object:
        if isinstance(value, dict) and isinstance(value.get("split_manifest_path"), str):
            updated = dict(value)
            updated["split_manifest_path"] = portable_contract_reference(
                value["split_manifest_path"]
            )
            return updated
        return value

    @model_validator(mode="after")
    def require_greedy_decoding(self) -> ExperimentContract:
        decoding = self.evaluation.decoding
        if decoding.do_sample or decoding.temperature != 0:
            raise ValueError("experiment contract requires greedy decoding")
        return self


class G0EvaluationContract(FrozenEvaluationModel):
    """Held-out base evaluation slice frozen before any SFT artifact exists."""

    schema_version: Literal["agoge.g0-evaluation-contract.v1"] = "agoge.g0-evaluation-contract.v1"
    experiment_id: str = Field(min_length=1)
    split_manifest_path: str = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    held_out_split_sha256: str = Field(pattern=_SHA256_PATTERN)
    logical_task_ids: tuple[str, ...] = Field(min_length=1)
    logical_task_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    base: EvaluationArm

    @field_validator("split_manifest_path")
    @classmethod
    def require_relative_manifest_path(cls, value: str) -> str:
        return portable_contract_reference(value)

    @model_validator(mode="after")
    def require_base_only(self) -> G0EvaluationContract:
        if self.base.role != "causal_base" or self.base.artifact is not None:
            raise ValueError("G0 contract requires a causal_base arm without an artifact")
        if logical_task_set_sha256(self.logical_task_ids) != self.logical_task_set_sha256:
            raise ValueError("logical task-set digest does not match logical_task_ids")
        if self.base.logical_task_set_sha256 != self.logical_task_set_sha256:
            raise ValueError("base arm task-set digest mismatch")
        return self


def load_experiment_contract(contract_path: str | Path) -> ExperimentContract:
    path = Path(contract_path).expanduser().resolve(strict=True)
    try:
        value = decode_json_object(
            path.read_bytes(),
            str(path),
            object_label="experiment contract",
        )
    except ValueError as exc:
        raise ValueError(f"invalid experiment contract JSON: {path}") from exc
    return ExperimentContract.model_validate(value)


def validate_experiment_contract(contract_path: str | Path) -> ExperimentContract:
    path = Path(contract_path).expanduser().resolve(strict=True)
    contract = load_experiment_contract(path)
    manifest = _validate_split_pin(path, contract.split)
    _validate_held_out_counts(contract, manifest)
    return contract


def build_experiment_contract(
    *,
    contract_path: str | Path,
    contract: ExperimentContract,
) -> ExperimentContract:
    destination = Path(contract_path).expanduser()
    validated = ExperimentContract.model_validate(contract.model_dump(mode="json"))
    payload = canonical_json_bytes(validated.model_dump(mode="json")) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes_noreplace(
        destination,
        payload,
        refusal="refusing to overwrite experiment contract",
        writer=_write_contract_payload,
    )
    return validated


def compose_g0_evaluation_contract(
    *,
    manifest_path: str | Path,
    contract_path: str | Path,
    experiment_id: str,
    base: EvaluationArm,
) -> G0EvaluationContract:
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    destination = Path(contract_path).expanduser()
    validated_base = EvaluationArm.model_validate(base.model_dump(mode="json"))
    if validated_base.role != "causal_base" or validated_base.artifact is not None:
        raise ValueError("G0 contract requires causal_base without artifact")
    manifest_snapshot = manifest_file.read_bytes()
    manifest = validate_split_manifest_snapshot(manifest_file, manifest_snapshot)
    task_ids = held_out_task_ids(manifest)
    task_digest = logical_task_set_sha256(task_ids)
    return G0EvaluationContract(
        experiment_id=experiment_id,
        split_manifest_path=_portable_relative_path(manifest_file, destination.parent.resolve()),
        split_manifest_sha256=sha256_bytes(manifest_snapshot),
        held_out_split_sha256=manifest.splits["held_out"].sha256,
        logical_task_ids=task_ids,
        logical_task_set_sha256=task_digest,
        base=validated_base.model_copy(update={"logical_task_set_sha256": task_digest}),
    )


def build_g0_evaluation_contract(
    *,
    manifest_path: str | Path,
    contract_path: str | Path,
    experiment_id: str,
    base: EvaluationArm,
) -> G0EvaluationContract:
    destination = Path(contract_path).expanduser()
    contract = compose_g0_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=destination,
        experiment_id=experiment_id,
        base=base,
    )
    payload = canonical_json_bytes(contract.model_dump(mode="json")) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes_noreplace(
        destination,
        payload,
        refusal="refusing to overwrite G0 evaluation contract",
        writer=_write_contract_payload,
    )
    return contract


def split_pin_from_manifest(
    manifest_path: str | Path,
    *,
    contract_anchor: Path,
    split_seed: int,
    split_salt: str,
    train_weight: int,
    validation_weight: int,
    held_out_weight: int,
) -> SplitPin:
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = validate_split_manifest_snapshot(manifest_file, manifest_file.read_bytes())
    return SplitPin(
        split_manifest_path=_portable_relative_path(manifest_file, contract_anchor),
        split_manifest_sha256=sha256_file(manifest_file),
        train_split_sha256=manifest.splits["train"].sha256,
        validation_split_sha256=manifest.splits["validation"].sha256,
        held_out_split_sha256=manifest.splits["held_out"].sha256,
        train_record_count=manifest.splits["train"].record_count,
        validation_record_count=manifest.splits["validation"].record_count,
        held_out_record_count=manifest.splits["held_out"].record_count,
        split_seed=split_seed,
        split_salt=split_salt,
        train_weight=train_weight,
        validation_weight=validation_weight,
        held_out_weight=held_out_weight,
    )


def _validate_split_pin(contract_path: Path, split: SplitPin) -> SplitManifest:
    manifest_path = (contract_path.parent / split.split_manifest_path).resolve(strict=True)
    manifest_snapshot = manifest_path.read_bytes()
    if sha256_bytes(manifest_snapshot) != split.split_manifest_sha256:
        raise ValueError("experiment contract split-manifest SHA-256 mismatch")
    manifest = validate_split_manifest_snapshot(manifest_path, manifest_snapshot)
    for name in ("train", "validation", "held_out"):
        digest = getattr(split, f"{name}_split_sha256")
        if manifest.splits[cast(Literal["train", "validation", "held_out"], name)].sha256 != digest:
            raise ValueError(f"experiment contract {name} split SHA-256 mismatch")
    return manifest


def _validate_held_out_counts(contract: ExperimentContract, manifest: SplitManifest) -> None:
    for name in ("train", "validation", "held_out"):
        expected = getattr(contract.split, f"{name}_record_count")
        split_name = cast(Literal["train", "validation", "held_out"], name)
        if manifest.splits[split_name].record_count != expected:
            raise ValueError(f"experiment contract {name} record count mismatch")


def _portable_relative_path(path: Path, anchor: Path) -> str:
    return os.path.relpath(path, anchor).replace("\\", "/")


def _write_contract_payload(path: Path, payload: bytes) -> None:
    write_fsynced_bytes(path, payload)
