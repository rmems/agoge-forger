"""Paired base-vs-SFT comparison for the held-out canary."""

from __future__ import annotations

from typing import Literal

from ._artifact_schema import FrozenEvaluationModel
from .score import OBJECTIVE_SCORING_VERSION, ExampleScore

PairOutcome = Literal["improved", "regressed", "tied", "invalid"]
EvalConclusion = Literal["improved", "regressed", "mixed", "null", "inconclusive"]
COMPARISON_SCHEMA_VERSION: Literal["agoge.paired-comparison.v1"] = "agoge.paired-comparison.v1"


class PairComparison(FrozenEvaluationModel):
    task_id: str
    outcome: PairOutcome
    base_verdict: str
    sft_verdict: str


class ComparisonSummary(FrozenEvaluationModel):
    schema_version: Literal["agoge.paired-comparison.v1"] = COMPARISON_SCHEMA_VERSION
    scoring_version: str
    n_tasks: int
    n_improved: int
    n_regressed: int
    n_tied: int
    n_invalid: int
    delta_accuracy: float | None
    conclusion: EvalConclusion
    outcomes: tuple[PairComparison, ...]


def compare_example(base: ExampleScore, sft: ExampleScore) -> PairComparison:
    if base.task_id != sft.task_id:
        raise ValueError("paired comparison requires matching task IDs")
    if base.scoring_version != sft.scoring_version:
        raise ValueError("paired comparison scoring versions drifted")
    if base.verdict == "invalid" or sft.verdict == "invalid":
        outcome: PairOutcome = "invalid"
    elif base.verdict == sft.verdict:
        outcome = "tied"
    elif sft.verdict == "correct":
        outcome = "improved"
    else:
        outcome = "regressed"
    return PairComparison(
        task_id=base.task_id,
        outcome=outcome,
        base_verdict=base.verdict,
        sft_verdict=sft.verdict,
    )


def compare_arms(
    base_scores: tuple[ExampleScore, ...],
    sft_scores: tuple[ExampleScore, ...],
    *,
    scoring_version: str = OBJECTIVE_SCORING_VERSION,
) -> ComparisonSummary:
    if scoring_version != OBJECTIVE_SCORING_VERSION:
        raise ValueError(
            f"unsupported objective scoring version {scoring_version!r}; "
            f"canary scorer is {OBJECTIVE_SCORING_VERSION}"
        )
    if tuple(score.task_id for score in base_scores) != tuple(
        score.task_id for score in sft_scores
    ):
        raise ValueError("paired comparison task IDs drifted between arms")
    outcomes = tuple(
        compare_example(base, sft) for base, sft in zip(base_scores, sft_scores, strict=True)
    )
    n_improved = sum(item.outcome == "improved" for item in outcomes)
    n_regressed = sum(item.outcome == "regressed" for item in outcomes)
    n_tied = sum(item.outcome == "tied" for item in outcomes)
    n_invalid = sum(item.outcome == "invalid" for item in outcomes)
    comparable = tuple(
        (base, sft)
        for base, sft, item in zip(base_scores, sft_scores, outcomes, strict=True)
        if item.outcome != "invalid"
    )
    delta_accuracy = None
    if comparable:
        base_correct = sum(base.verdict == "correct" for base, _sft in comparable)
        sft_correct = sum(sft.verdict == "correct" for _base, sft in comparable)
        delta_accuracy = (sft_correct - base_correct) / len(comparable)
    return ComparisonSummary(
        scoring_version=scoring_version,
        n_tasks=len(outcomes),
        n_improved=n_improved,
        n_regressed=n_regressed,
        n_tied=n_tied,
        n_invalid=n_invalid,
        delta_accuracy=delta_accuracy,
        conclusion=_conclusion(n_improved, n_regressed, n_tied),
        outcomes=outcomes,
    )


def _conclusion(n_improved: int, n_regressed: int, n_tied: int) -> EvalConclusion:
    comparable = n_improved + n_regressed + n_tied
    if comparable == 0:
        return "inconclusive"
    if n_improved and n_regressed:
        return "mixed"
    if n_improved:
        return "improved"
    if n_regressed:
        return "regressed"
    if n_tied == comparable:
        return "null"
    return "inconclusive"
