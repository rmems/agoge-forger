"""Deterministic objective scoring for the code-repair held-out canary."""

from __future__ import annotations

from typing import Literal

from ._artifact_schema import FrozenEvaluationModel

OBJECTIVE_SCORING_VERSION = "exact-match-v1"
ExampleVerdict = Literal["correct", "incorrect", "invalid"]
GenerationStatus = Literal["ok", "unsupported", "invalid"]


class GenerationRecord(FrozenEvaluationModel):
    task_id: str
    prompt: str | None = None
    expected_completion: str | None = None
    raw_text: str | None = None
    completion: str | None = None
    prompt_token_count: int | None = None
    completion_token_count: int | None = None
    status: GenerationStatus
    reason: str | None = None


class ExampleScore(FrozenEvaluationModel):
    task_id: str
    scoring_version: str
    verdict: ExampleVerdict
    expected_completion: str | None = None
    predicted_completion: str | None = None
    reason: str | None = None


class ArmMetrics(FrozenEvaluationModel):
    role: Literal["causal_base", "causal_sft"]
    scoring_version: str
    n_tasks: int
    n_correct: int
    n_incorrect: int
    n_invalid: int
    n_scored: int
    accuracy: float | None


def normalize_completion(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip()


def score_generation(
    record: GenerationRecord, *, scoring_version: str = OBJECTIVE_SCORING_VERSION
) -> ExampleScore:
    if scoring_version != OBJECTIVE_SCORING_VERSION:
        raise ValueError(
            f"unsupported objective scoring version {scoring_version!r}; "
            f"canary scorer is {OBJECTIVE_SCORING_VERSION}"
        )
    if record.status != "ok":
        return ExampleScore(
            task_id=record.task_id,
            scoring_version=scoring_version,
            verdict="invalid",
            expected_completion=record.expected_completion,
            predicted_completion=record.completion,
            reason=record.reason or record.status,
        )
    if record.expected_completion is None or record.completion is None:
        return ExampleScore(
            task_id=record.task_id,
            scoring_version=scoring_version,
            verdict="invalid",
            expected_completion=record.expected_completion,
            predicted_completion=record.completion,
            reason="missing expected or predicted completion",
        )
    expected = normalize_completion(record.expected_completion)
    predicted = normalize_completion(record.completion)
    if expected == "":
        return ExampleScore(
            task_id=record.task_id,
            scoring_version=scoring_version,
            verdict="invalid",
            expected_completion=record.expected_completion,
            predicted_completion=record.completion,
            reason="empty expected completion",
        )
    correct = predicted == expected
    return ExampleScore(
        task_id=record.task_id,
        scoring_version=scoring_version,
        verdict="correct" if correct else "incorrect",
        expected_completion=record.expected_completion,
        predicted_completion=record.completion,
    )


def score_arm(
    role: Literal["causal_base", "causal_sft"],
    records: tuple[GenerationRecord, ...],
    *,
    scoring_version: str = OBJECTIVE_SCORING_VERSION,
) -> tuple[tuple[ExampleScore, ...], ArmMetrics]:
    scores = tuple(score_generation(record, scoring_version=scoring_version) for record in records)
    n_correct = sum(score.verdict == "correct" for score in scores)
    n_incorrect = sum(score.verdict == "incorrect" for score in scores)
    n_invalid = sum(score.verdict == "invalid" for score in scores)
    n_scored = n_correct + n_incorrect
    accuracy = None if n_scored == 0 else n_correct / n_scored
    metrics = ArmMetrics(
        role=role,
        scoring_version=scoring_version,
        n_tasks=len(scores),
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        n_invalid=n_invalid,
        n_scored=n_scored,
        accuracy=accuracy,
    )
    return scores, metrics
