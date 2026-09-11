"""Paired base-vs-SFT comparison for the held-out canary."""

from __future__ import annotations

from typing import Literal

from ._artifact_schema import FrozenEvaluationModel
from .score import OBJECTIVE_SCORING_VERSION, ExampleScore

PairOutcome = Literal["improved", "regressed", "tied", "invalid"]
EvalConclusion = Literal["improved", "regressed", "mixed", "null", "inconclusive"]
COMPARISON_SCHEMA_VERSION: Literal["agoge.paired-comparison.v1"] = "agoge.paired-comparison.v1"
_CONCLUSION_BY_SIGNALS: dict[tuple[bool, bool, bool], EvalConclusion] = {
    (True, True, True): "mixed",
    (True, True, False): "mixed",
    (True, False, True): "improved",
    (True, False, False): "improved",
    (False, True, True): "regressed",
    (False, True, False): "regressed",
    (False, False, True): "null",
    (False, False, False): "inconclusive",
}


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
    _require_paired_scores(base, sft)
    return PairComparison(
        task_id=base.task_id,
        outcome=_pair_outcome(base.verdict, sft.verdict),
        base_verdict=base.verdict,
        sft_verdict=sft.verdict,
    )


def compare_arms(
    base_scores: tuple[ExampleScore, ...],
    sft_scores: tuple[ExampleScore, ...],
    *,
    scoring_version: str = OBJECTIVE_SCORING_VERSION,
) -> ComparisonSummary:
    _require_scoring_version(scoring_version)
    _require_matching_task_ids(base_scores, sft_scores)
    outcomes = tuple(
        compare_example(base, sft) for base, sft in zip(base_scores, sft_scores, strict=True)
    )
    n_improved = sum(item.outcome == "improved" for item in outcomes)
    n_regressed = sum(item.outcome == "regressed" for item in outcomes)
    n_tied = sum(item.outcome == "tied" for item in outcomes)
    n_invalid = sum(item.outcome == "invalid" for item in outcomes)
    return ComparisonSummary(
        scoring_version=scoring_version,
        n_tasks=len(outcomes),
        n_improved=n_improved,
        n_regressed=n_regressed,
        n_tied=n_tied,
        n_invalid=n_invalid,
        delta_accuracy=_delta_accuracy(base_scores, sft_scores, outcomes),
        conclusion=_conclusion(n_improved, n_regressed, n_tied),
        outcomes=outcomes,
    )


def _require_paired_scores(base: ExampleScore, sft: ExampleScore) -> None:
    if base.task_id != sft.task_id:
        raise ValueError("paired comparison requires matching task IDs")
    if base.scoring_version != sft.scoring_version:
        raise ValueError("paired comparison scoring versions drifted")


def _pair_outcome(base_verdict: str, sft_verdict: str) -> PairOutcome:
    if base_verdict == "invalid" or sft_verdict == "invalid":
        return "invalid"
    if base_verdict == sft_verdict:
        return "tied"
    if sft_verdict == "correct":
        return "improved"
    return "regressed"


def _require_scoring_version(scoring_version: str) -> None:
    if scoring_version != OBJECTIVE_SCORING_VERSION:
        raise ValueError(
            f"unsupported objective scoring version {scoring_version!r}; "
            f"canary scorer is {OBJECTIVE_SCORING_VERSION}"
        )


def _require_matching_task_ids(
    base_scores: tuple[ExampleScore, ...], sft_scores: tuple[ExampleScore, ...]
) -> None:
    if tuple(score.task_id for score in base_scores) != tuple(
        score.task_id for score in sft_scores
    ):
        raise ValueError("paired comparison task IDs drifted between arms")


def _delta_accuracy(
    base_scores: tuple[ExampleScore, ...],
    sft_scores: tuple[ExampleScore, ...],
    outcomes: tuple[PairComparison, ...],
) -> float | None:
    comparable = tuple(
        (base, sft)
        for base, sft, item in zip(base_scores, sft_scores, outcomes, strict=True)
        if item.outcome != "invalid"
    )
    if not comparable:
        return None
    base_correct = sum(base.verdict == "correct" for base, _sft in comparable)
    sft_correct = sum(sft.verdict == "correct" for _base, sft in comparable)
    return (sft_correct - base_correct) / len(comparable)


def _conclusion(n_improved: int, n_regressed: int, n_tied: int) -> EvalConclusion:
    return _CONCLUSION_BY_SIGNALS[(bool(n_improved), bool(n_regressed), bool(n_tied))]
