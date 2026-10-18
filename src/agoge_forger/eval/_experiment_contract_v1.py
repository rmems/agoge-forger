"""Schema for ``agoge.experiment-contract.v1`` and the pins v2 extends."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from ._artifact_schema import FrozenEvaluationModel, portable_contract_reference
from .contract import DecodingContract

EXPERIMENT_CONTRACT_VERSION: Literal["agoge.experiment-contract.v1"] = (
    "agoge.experiment-contract.v1"
)
SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40,64}$"


class DatasetProvenance(FrozenEvaluationModel):
    factory_repository: str = Field(min_length=1)
    factory_revision: str = Field(pattern=REVISION_PATTERN)
    export_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    agoge_jsonl_sha256: str = Field(pattern=SHA256_PATTERN)


class SplitPin(FrozenEvaluationModel):
    split_manifest_path: str = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    train_split_sha256: str = Field(pattern=SHA256_PATTERN)
    validation_split_sha256: str = Field(pattern=SHA256_PATTERN)
    held_out_split_sha256: str = Field(pattern=SHA256_PATTERN)
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
    model_revision: str = Field(pattern=REVISION_PATTERN)
    trust_remote_code: bool = False


class SerializerPin(FrozenEvaluationModel):
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=SHA256_PATTERN)


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


def normalize_split_pin_path(value: object) -> object:
    """Reject host-absolute split-manifest references before field validation."""
    if isinstance(value, dict) and isinstance(value.get("split_manifest_path"), str):
        updated = dict(value)
        updated["split_manifest_path"] = portable_contract_reference(value["split_manifest_path"])
        return updated
    return value


def require_greedy_evaluation(evaluation: EvaluationPin) -> None:
    decoding = evaluation.decoding
    if decoding.do_sample or decoding.temperature != 0:
        raise ValueError("experiment contract requires greedy decoding")


class ExperimentContract(FrozenEvaluationModel):
    schema_version: Literal["agoge.experiment-contract.v1"] = EXPERIMENT_CONTRACT_VERSION
    experiment_id: str = Field(min_length=1)
    agoge_git_sha: str = Field(pattern=REVISION_PATTERN)
    model: ModelPin
    dataset: DatasetProvenance
    split: SplitPin
    serializer: SerializerPin
    training: TrainingPin
    evaluation: EvaluationPin

    @field_validator("split", mode="before")
    @classmethod
    def normalize_split_manifest_path(cls, value: object) -> object:
        return normalize_split_pin_path(value)

    @model_validator(mode="after")
    def require_greedy_decoding(self) -> ExperimentContract:
        require_greedy_evaluation(self.evaluation)
        return self
