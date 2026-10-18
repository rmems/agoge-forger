"""Schema for ``agoge.experiment-contract.v2``.

v2 keeps every v1 pin and adds what a multi-arm, self-generation experiment
must freeze before G0: the hypothesis, the arms and their statistics, the
self-generation loop, the self-generated row policy, the parent adapter, and
how the trained adapter is initialised. Every added field is required; the
optional blocks (``model.parent_adapter``, ``dataset.self_generated_rows``,
``self_generation``) must still be written out, as ``null`` when unused.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from ._artifact_schema import FrozenEvaluationModel
from ._experiment_contract_v1 import (
    REVISION_PATTERN,
    SHA256_PATTERN,
    DatasetProvenance,
    EvaluationPin,
    ModelPin,
    SerializerPin,
    SplitPin,
    TrainingPin,
    normalize_split_pin_path,
    require_greedy_evaluation,
)
from .contract import DecodingContract

EXPERIMENT_CONTRACT_V2_VERSION: Literal["agoge.experiment-contract.v2"] = (
    "agoge.experiment-contract.v2"
)

ArmId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
NonEmptyText = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
Rate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


def _require_unique(values: Iterable[str], field: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{field} contains duplicate entry {value!r}")
        seen.add(value)


class ParentAdapterPin(FrozenEvaluationModel):
    """Adapter this experiment descends from (for example, G1 for a G2 run)."""

    experiment_id: NonEmptyText
    adapter_sha256: Sha256


class ModelPinV2(ModelPin):
    parent_adapter: ParentAdapterPin | None


class SelfGeneratedRowsPin(FrozenEvaluationModel):
    max_fraction_of_train: float = Field(gt=0, le=1, allow_inf_nan=False)
    min_survival_rate_to_continue: Rate
    provenance_required: tuple[NonEmptyText, ...] = Field(min_length=1)

    @field_validator("provenance_required")
    @classmethod
    def unique_provenance_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(value, "dataset.self_generated_rows.provenance_required")
        return value


class DatasetProvenanceV2(DatasetProvenance):
    self_generated_rows: SelfGeneratedRowsPin | None


class TrainingPinV2(TrainingPin):
    init: Literal["fresh", "stacked"]


class ExperimentArmPin(FrozenEvaluationModel):
    arm_id: ArmId
    role: Literal["base", "parent", "treatment", "control"]
    training_rows: Literal["none", "base", "base_plus_self_generated"]


class IntervalPin(FrozenEvaluationModel):
    method: Literal["wilson"]
    confidence: float = Field(gt=0, lt=1, allow_inf_nan=False)


class PairedComparisonPin(FrozenEvaluationModel):
    treatment: ArmId
    control: ArmId

    @model_validator(mode="after")
    def distinct_arms(self) -> PairedComparisonPin:
        if self.treatment == self.control:
            raise ValueError("a paired comparison needs two different arms")
        return self


class PairedTestPin(FrozenEvaluationModel):
    method: Literal["mcnemar"]
    comparisons: tuple[PairedComparisonPin, ...] = Field(min_length=1)


class EvaluationPinV2(EvaluationPin):
    arms: tuple[ExperimentArmPin, ...] = Field(min_length=2)
    runs: int = Field(ge=1)
    interval: IntervalPin
    paired_test: PairedTestPin

    @model_validator(mode="after")
    def comparisons_name_declared_arms(self) -> EvaluationPinV2:
        _require_unique((arm.arm_id for arm in self.arms), "evaluation.arms[].arm_id")
        declared = self.arm_ids()
        for index, comparison in enumerate(self.paired_test.comparisons):
            for side in ("treatment", "control"):
                arm_id = getattr(comparison, side)
                if arm_id not in declared:
                    raise ValueError(
                        f"evaluation.paired_test.comparisons[{index}].{side} names "
                        f"undeclared arm {arm_id!r}"
                    )
        return self

    def arm_ids(self) -> frozenset[str]:
        return frozenset(arm.arm_id for arm in self.arms)


class SeedCatalogPin(FrozenEvaluationModel):
    catalog_id: NonEmptyText
    catalog_sha256: Sha256
    held_out_denylist_sha256: Sha256


class ProposePin(FrozenEvaluationModel):
    proposer_arm: ArmId
    proposals_per_program: int = Field(ge=1)
    decoding: DecodingContract


class VerifyTaskPin(FrozenEvaluationModel):
    verifier_id: NonEmptyText
    checks: tuple[NonEmptyText, ...] = Field(min_length=1)

    @field_validator("checks")
    @classmethod
    def unique_checks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(value, "self_generation.verify_task.checks")
        return value


class SolvePin(FrozenEvaluationModel):
    solver_arm: ArmId
    attempts: int = Field(ge=1)
    output_format: NonEmptyText


class VerifySolutionPin(FrozenEvaluationModel):
    required_verdict: NonEmptyText
    tests_untouched: bool
    hidden_cases_required: bool


class DifficultyBandPin(FrozenEvaluationModel):
    metric: NonEmptyText
    min_pass_rate: Rate
    max_pass_rate: Rate

    @model_validator(mode="after")
    def ordered_band(self) -> DifficultyBandPin:
        if self.min_pass_rate >= self.max_pass_rate:
            raise ValueError(
                "self_generation.difficulty_band.min_pass_rate must be below max_pass_rate"
            )
        return self


class SelfGenerationPin(FrozenEvaluationModel):
    seed_catalog: SeedCatalogPin
    propose: ProposePin
    verify_task: VerifyTaskPin
    solve: SolvePin
    verify_solution: VerifySolutionPin
    difficulty_band: DifficultyBandPin
    iterations: int = Field(ge=1)
    stop_rules: tuple[NonEmptyText, ...] = Field(min_length=1)

    @field_validator("stop_rules")
    @classmethod
    def unique_stop_rules(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(value, "self_generation.stop_rules")
        return value


class ExperimentContractV2(FrozenEvaluationModel):
    schema_version: Literal["agoge.experiment-contract.v2"]
    experiment_id: NonEmptyText
    hypothesis: NonEmptyText
    agoge_git_sha: str = Field(pattern=REVISION_PATTERN)
    model: ModelPinV2
    dataset: DatasetProvenanceV2
    split: SplitPin
    serializer: SerializerPin
    training: TrainingPinV2
    evaluation: EvaluationPinV2
    self_generation: SelfGenerationPin | None

    @field_validator("split", mode="before")
    @classmethod
    def normalize_split_manifest_path(cls, value: object) -> object:
        return normalize_split_pin_path(value)

    @model_validator(mode="after")
    def require_consistent_design(self) -> ExperimentContractV2:
        require_greedy_evaluation(self.evaluation)
        _require_parent_for_stacked_init(self)
        _require_self_generation_pairing(self)
        _require_self_generation_arms(self)
        return self


def _require_parent_for_stacked_init(contract: ExperimentContractV2) -> None:
    if contract.training.init == "stacked" and contract.model.parent_adapter is None:
        raise ValueError("training.init='stacked' requires model.parent_adapter")


def _require_self_generation_pairing(contract: ExperimentContractV2) -> None:
    has_loop = contract.self_generation is not None
    has_policy = contract.dataset.self_generated_rows is not None
    if has_loop != has_policy:
        raise ValueError(
            "self_generation and dataset.self_generated_rows must both be set or both be null"
        )
    uses_self_rows = any(
        arm.training_rows == "base_plus_self_generated" for arm in contract.evaluation.arms
    )
    if uses_self_rows and not has_loop:
        raise ValueError(
            "an evaluation arm trains on base_plus_self_generated rows but self_generation is null"
        )


def _require_self_generation_arms(contract: ExperimentContractV2) -> None:
    loop = contract.self_generation
    if loop is None:
        return
    declared = contract.evaluation.arm_ids()
    for field, arm_id in (
        ("self_generation.propose.proposer_arm", loop.propose.proposer_arm),
        ("self_generation.solve.solver_arm", loop.solve.solver_arm),
    ):
        if arm_id not in declared:
            raise ValueError(f"{field} names undeclared arm {arm_id!r}")
