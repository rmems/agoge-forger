"""Pre-registered contract for the first Granite 4.1 base-versus-SFT comparison.

The numeric budget, decoding settings, and interpretation rule are literals.
Changing them requires a new schema version and a new experiment id. They are
not inferred from a measured baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .._strict_json import decode_json_object
from ..eval._artifact_schema import FrozenEvaluationModel
from ..eval.compare import EvalConclusion, PairOutcome, compare_arms
from ..eval.score import OBJECTIVE_SCORING_VERSION, ExampleScore, ExampleVerdict
from ..eval.serializers import SERIALIZER_ID, SERIALIZER_VERSION
from ..split_schema import canonical_json_bytes, sha256_bytes

CONTRACT_VERSION: Literal["agoge.granite-first-sft-contract.v1"] = (
    "agoge.granite-first-sft-contract.v1"
)
COMPARISON_VERSION: Literal["agoge.granite-first-sft-comparison.v1"] = (
    "agoge.granite-first-sft-comparison.v1"
)
TRAINING_RESULTS_VERSION: Literal["agoge.granite-first-sft-training.v1"] = (
    "agoge.granite-first-sft-training.v1"
)
QUALIFICATION_VERSION: Literal["agoge.model-compatibility.v1"] = "agoge.model-compatibility.v1"
BUDGET_ID: Literal["granite-4.1-first-sft-budget-v1"] = "granite-4.1-first-sft-budget-v1"
MODEL_REPOSITORY: Literal["ibm-granite/granite-4.1-3b-base"] = "ibm-granite/granite-4.1-3b-base"
REPORT_PARENT_NAME = "granite-4.1-first-sft"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REVISION_PATTERN = r"^[0-9a-f]{40,64}$"
_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_EXPERIMENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"

RESEARCH_QUESTION = (
    "Under one frozen code-repair dataset, training objective, and held-out "
    "evaluation contract, does a bounded SFT intervention change Granite 4.1 3B "
    "Base performance relative to the exact untouched starting checkpoint?"
)
INTERPRETATION_RULE = (
    "Score every held-out task with exact-match-v1. "
    "Pair each task as improved, regressed, tied, or invalid. "
    "Conclude improved when at least one task improved and none regressed; "
    "regressed when at least one task regressed and none improved; "
    "mixed when both improved and regressed tasks exist; "
    "null when every comparable task tied; "
    "inconclusive when no comparable task remains. "
    "Invalid tasks stay listed and do not invent a quality delta. "
    "Training loss is diagnostic and cannot replace the held-out conclusion. "
    "Positive, negative, null, mixed, and inconclusive are all valid outcomes. "
    "Do not change this rule after seeing G0 or G1."
)


def interpretation_rule_sha256() -> str:
    return sha256_bytes(INTERPRETATION_RULE.encode("utf-8"))


class QualificationReport(FrozenEvaluationModel):
    """Evidence that #104 passed load, train, save, clean reload, and generation."""

    schema_version: Literal["agoge.model-compatibility.v1"] = QUALIFICATION_VERSION
    model_repository: Literal["ibm-granite/granite-4.1-3b-base"]
    model_revision: str = Field(pattern=_REVISION_PATTERN)
    tokenizer_repository: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=_REVISION_PATTERN)
    tokenizer_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_modules: tuple[str, ...] = Field(min_length=1)
    verdict: Literal["pass", "fail"]
    lifecycle: Literal["load-train-save-reload-generate"]
    trust_remote_code: Literal[False] = False

    @model_validator(mode="after")
    def require_unique_targets(self) -> QualificationReport:
        _require_target_modules(self.target_modules)
        return self


class SplitPin(FrozenEvaluationModel):
    source_repository: str = Field(min_length=1)
    source_revision: str = Field(pattern=_REVISION_PATTERN)
    dataset_version: str = Field(min_length=1)
    split_manifest_path: str = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    train_sha256: str = Field(pattern=_SHA256_PATTERN)
    validation_sha256: str = Field(pattern=_SHA256_PATTERN)
    held_out_sha256: str = Field(pattern=_SHA256_PATTERN)
    logical_task_ids: tuple[str, ...] = Field(min_length=1)
    logical_task_set_sha256: str = Field(pattern=_SHA256_PATTERN)


class RegisteredBudget(FrozenEvaluationModel):
    budget_id: Literal["granite-4.1-first-sft-budget-v1"] = BUDGET_ID
    optimizer: Literal["adamw_torch"] = "adamw_torch"
    learning_rate: float = 0.0001
    num_train_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = 8
    seed: int = 42
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    bf16: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["bfloat16"] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = Field(min_length=1)
    target_modules_mode: Literal["explicit"] = "explicit"
    save_steps: int = 50
    save_total_limit: int = 2
    completion_only_loss: Literal[True] = True
    dataset_text_field: Literal["text"] = "text"
    truncation: Literal["reject_over_budget"] = "reject_over_budget"
    boundary_unit: Literal["unicode_code_point"] = "unicode_code_point"

    @model_validator(mode="after")
    def require_registered_values(self) -> RegisteredBudget:
        _require_registered_budget(self)
        return self


class DecodingBudget(FrozenEvaluationModel):
    do_sample: Literal[False] = False
    seed: Literal[17] = 17
    max_new_tokens: Literal[128] = 128
    temperature: float = 0.0
    top_p: float = 1.0
    context_window: Literal[2048] = 2048
    truncation_policy: Literal["mark_unsupported"] = "mark_unsupported"
    scoring_version: Literal["exact-match-v1"] = "exact-match-v1"
    primary_metric: Literal["held_out_exact_match_accuracy"] = "held_out_exact_match_accuracy"
    protected_metric: None = None

    @model_validator(mode="after")
    def require_registered_decoding(self) -> DecodingBudget:
        if self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("decoding budget drifted from the pre-registered greedy contract")
        if self.scoring_version != OBJECTIVE_SCORING_VERSION:
            raise ValueError("scoring version drifted from the held-out harness")
        return self


class BlockerAttestation(FrozenEvaluationModel):
    completion_only_loss: Literal["satisfied"] = "satisfied"
    held_out_harness: Literal["satisfied"] = "satisfied"
    model_qualification: Literal["satisfied"] = "satisfied"
    split_pin: Literal["satisfied"] = "satisfied"
    qualification_report_path: str = Field(min_length=1)
    qualification_report_sha256: str = Field(pattern=_SHA256_PATTERN)


class GraniteFirstSftContract(FrozenEvaluationModel):
    schema_version: Literal["agoge.granite-first-sft-contract.v1"] = CONTRACT_VERSION
    experiment_id: str = Field(pattern=_EXPERIMENT_ID_PATTERN)
    agoge_commit: str = Field(pattern=_COMMIT_PATTERN)
    research_question: str
    model_repository: Literal["ibm-granite/granite-4.1-3b-base"] = MODEL_REPOSITORY
    model_revision: str = Field(pattern=_REVISION_PATTERN)
    tokenizer_repository: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=_REVISION_PATTERN)
    tokenizer_sha256: str = Field(pattern=_SHA256_PATTERN)
    serializer_id: Literal["code-repair-prompt-v1"] = "code-repair-prompt-v1"
    serializer_version: Literal["1"] = "1"
    serializer_sha256: str = Field(pattern=_SHA256_PATTERN)
    split: SplitPin
    budget: RegisteredBudget
    decoding: DecodingBudget
    interpretation_rule: str
    interpretation_rule_sha256: str = Field(pattern=_SHA256_PATTERN)
    blockers: BlockerAttestation

    @model_validator(mode="after")
    def require_frozen_rule(self) -> GraniteFirstSftContract:
        if self.research_question != RESEARCH_QUESTION:
            raise ValueError("research question drifted from the pre-registered text")
        if self.interpretation_rule != INTERPRETATION_RULE:
            raise ValueError("interpretation rule drifted from the pre-registered text")
        if self.interpretation_rule_sha256 != interpretation_rule_sha256():
            raise ValueError("interpretation rule SHA-256 does not match the rule text")
        if self.serializer_id != SERIALIZER_ID or self.serializer_version != SERIALIZER_VERSION:
            raise ValueError("serializer id drifted from code-repair-prompt-v1")
        if len(set(self.split.logical_task_ids)) != len(self.split.logical_task_ids):
            raise ValueError("logical_task_ids must be unique")
        return self


class SoftwareEnvironment(FrozenEvaluationModel):
    python_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    cuda_version: str | None = None
    gpu_name: str | None = None
    transformer_version: str = Field(min_length=1)


class TrainingResults(FrozenEvaluationModel):
    schema_version: Literal["agoge.granite-first-sft-training.v1"] = TRAINING_RESULTS_VERSION
    status: Literal["completed", "failed"]
    accepted_records: int | None = None
    accepted_tokens: int | None = None
    train_loss: float | None = None
    validation_loss: float | None = None
    wall_clock_seconds: float | None = None
    tokens_per_second: float | None = None
    peak_vram_bytes: int | None = None
    adapter_bytes: int | None = None
    adapter_sha256: str | None = None
    checkpoint_bytes: int | None = None
    failure_reason: str | None = None
    environment: SoftwareEnvironment | None = None

    @model_validator(mode="after")
    def require_measurements_when_completed(self) -> TrainingResults:
        if self.status == "failed":
            if not self.failure_reason:
                raise ValueError("failed training requires failure_reason")
            return self
        _require_completed_measurements(self)
        return self


class PairRow(FrozenEvaluationModel):
    task_id: str = Field(min_length=1)
    outcome: PairOutcome
    base_verdict: ExampleVerdict
    sft_verdict: ExampleVerdict


class ExperimentComparison(FrozenEvaluationModel):
    schema_version: Literal["agoge.granite-first-sft-comparison.v1"] = COMPARISON_VERSION
    status: Literal["scored", "not_run"]
    reason: str | None = None
    scoring_version: str | None = None
    g0_accuracy: float | None = None
    g1_accuracy: float | None = None
    delta_accuracy: float | None = None
    n_tasks: int | None = None
    n_improved: int | None = None
    n_regressed: int | None = None
    n_tied: int | None = None
    n_invalid: int | None = None
    conclusion: EvalConclusion | None = None
    interpretation_rule_sha256: str | None = None
    pairs: tuple[PairRow, ...] = ()

    @model_validator(mode="after")
    def require_pairs_match_conclusion(self) -> ExperimentComparison:
        if self.status == "not_run":
            _require_unscored(self)
            return self
        _require_scored(self)
        return self


class ReloadObservation(FrozenEvaluationModel):
    status: Literal["reloaded", "failed"]
    failure_reason: str | None = None

    @model_validator(mode="after")
    def require_failure_reason(self) -> ReloadObservation:
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed reload requires failure_reason")
        return self


def load_experiment_contract(path: Path) -> GraniteFirstSftContract:
    payload = path.read_bytes()
    document = decode_json_object(payload, str(path), object_label="experiment contract")
    return GraniteFirstSftContract.model_validate(document)


def contract_json_bytes(contract: GraniteFirstSftContract) -> bytes:
    return canonical_json_bytes(contract.model_dump(mode="json")) + b"\n"


def _require_target_modules(target_modules: tuple[str, ...]) -> None:
    if any(not module.strip() for module in target_modules):
        raise ValueError("target_modules entries must be nonempty")
    if len(set(target_modules)) != len(target_modules):
        raise ValueError("target_modules must be unique")


def _require_registered_budget(budget: RegisteredBudget) -> None:
    expected = {
        "learning_rate": 0.0001,
        "num_train_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 8,
        "seed": 42,
        "max_seq_length": 2048,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "save_steps": 50,
        "save_total_limit": 2,
    }
    drifted = [name for name, value in expected.items() if getattr(budget, name) != value]
    if drifted:
        raise ValueError(f"training budget drifted from {BUDGET_ID}: {drifted}")
    if not budget.gradient_checkpointing or not budget.bf16 or not budget.load_in_4bit:
        raise ValueError("quantization or checkpointing budget drifted")
    if not budget.bnb_4bit_use_double_quant:
        raise ValueError("quantization budget drifted")
    _require_target_modules(budget.target_modules)
    if budget.effective_batch_size != budget.batch_size * budget.gradient_accumulation_steps:
        raise ValueError("effective_batch_size must equal batch_size * gradient_accumulation_steps")


def _require_completed_measurements(results: TrainingResults) -> None:
    required = (
        "accepted_records",
        "accepted_tokens",
        "train_loss",
        "wall_clock_seconds",
        "tokens_per_second",
        "peak_vram_bytes",
        "adapter_bytes",
        "adapter_sha256",
        "checkpoint_bytes",
        "environment",
    )
    missing = [name for name in required if getattr(results, name) is None]
    if missing:
        raise ValueError(f"completed training is missing measurements: {missing}")
    if results.failure_reason is not None:
        raise ValueError("completed training cannot carry failure_reason")
    if results.accepted_records is None or results.accepted_records < 1:
        raise ValueError("completed training requires at least one accepted record")
    if results.accepted_tokens is None or results.accepted_tokens < 1:
        raise ValueError("completed training requires accepted tokens")
    if results.wall_clock_seconds is None or results.wall_clock_seconds <= 0:
        raise ValueError("completed training requires positive wall_clock_seconds")
    _require_throughput(results)
    if results.adapter_sha256 is None or len(results.adapter_sha256) != 64:
        raise ValueError("completed training requires an adapter SHA-256")
    if results.environment is None or not results.environment.gpu_name:
        raise ValueError("completed training requires a recorded GPU name")


def _require_throughput(results: TrainingResults) -> None:
    if (
        results.accepted_tokens is None
        or results.wall_clock_seconds is None
        or results.tokens_per_second is None
    ):
        raise ValueError("completed training requires throughput inputs")
    expected = results.accepted_tokens / results.wall_clock_seconds
    if abs(results.tokens_per_second - expected) > max(1e-6, abs(expected) * 0.01):
        raise ValueError("tokens_per_second does not match accepted_tokens / wall_clock_seconds")


def _require_unscored(comparison: ExperimentComparison) -> None:
    if not comparison.reason:
        raise ValueError("unscored comparison requires a reason")
    if comparison.conclusion is not None or comparison.pairs:
        raise ValueError("unscored comparison cannot carry a conclusion or pairs")


def _require_scored(comparison: ExperimentComparison) -> None:
    if comparison.scoring_version != OBJECTIVE_SCORING_VERSION:
        raise ValueError("scored comparison scoring version drifted")
    if comparison.interpretation_rule_sha256 != interpretation_rule_sha256():
        raise ValueError("scored comparison interpretation rule drifted")
    summary = compare_arms(
        tuple(_score(pair.task_id, pair.base_verdict) for pair in comparison.pairs),
        tuple(_score(pair.task_id, pair.sft_verdict) for pair in comparison.pairs),
    )
    for pair, item in zip(comparison.pairs, summary.outcomes, strict=True):
        if pair.outcome != item.outcome:
            raise ValueError("pair outcome does not match verdicts")
    if comparison.conclusion != summary.conclusion:
        raise ValueError("comparison conclusion does not match paired verdicts")
    if (
        comparison.n_tasks != summary.n_tasks
        or comparison.n_improved != summary.n_improved
        or comparison.n_regressed != summary.n_regressed
        or comparison.n_tied != summary.n_tied
        or comparison.n_invalid != summary.n_invalid
    ):
        raise ValueError("comparison counts do not match paired verdicts")
    _require_accuracy(comparison, summary.delta_accuracy)


def _score(task_id: str, verdict: ExampleVerdict) -> ExampleScore:
    return ExampleScore(task_id=task_id, scoring_version=OBJECTIVE_SCORING_VERSION, verdict=verdict)


def _require_accuracy(comparison: ExperimentComparison, delta_accuracy: float | None) -> None:
    comparable = tuple(pair for pair in comparison.pairs if pair.outcome != "invalid")
    if not comparable:
        if comparison.g0_accuracy is not None or comparison.g1_accuracy is not None:
            raise ValueError("inconclusive comparison cannot report accuracy")
        if comparison.delta_accuracy is not None or delta_accuracy is not None:
            raise ValueError("inconclusive comparison cannot report an accuracy delta")
        return
    g0_accuracy = sum(pair.base_verdict == "correct" for pair in comparable) / len(comparable)
    g1_accuracy = sum(pair.sft_verdict == "correct" for pair in comparable) / len(comparable)
    if comparison.g0_accuracy != g0_accuracy or comparison.g1_accuracy != g1_accuracy:
        raise ValueError("arm accuracy does not match paired verdicts")
    if comparison.delta_accuracy != delta_accuracy:
        raise ValueError("accuracy delta does not match paired verdicts")
