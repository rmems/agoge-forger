"""Deterministic objective scoring for the code-repair held-out canary.

Three scorers are registered. ``exact-match-v1`` is the original, frozen scorer and the
default; it compares raw text, so markdown code fences around a correct answer make it
score as wrong (see #183). The other two are additive and never change v1 results:

* ``exact-match-v2``: v1 after unwrapping a leading markdown code fence (first block wins).
* ``doctest-v1``: runs the candidate module's doctests in a sandboxed subprocess.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from ._artifact_schema import FrozenEvaluationModel
from ._doctest_runner import (
    DOCTEST_TIMEOUT_SECONDS,
    DoctestOutcome,
    reference_examples,
    run_doctests,
)
from ._fence import extract_fenced_code

OBJECTIVE_SCORING_VERSION = "exact-match-v1"
EXACT_MATCH_V2_SCORING_VERSION = "exact-match-v2"
DOCTEST_V1_SCORING_VERSION = "doctest-v1"
SUPPORTED_SCORING_VERSIONS = (
    OBJECTIVE_SCORING_VERSION,
    EXACT_MATCH_V2_SCORING_VERSION,
    DOCTEST_V1_SCORING_VERSION,
)
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
    # Populated by ``doctest-v1`` only; ``None`` for the exact-match scorers.
    n_examples: int | None = None
    n_failed: int | None = None
    error: str | None = None


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


def require_supported_scoring_version(scoring_version: str) -> None:
    if scoring_version not in SUPPORTED_SCORING_VERSIONS:
        raise ValueError(
            f"unsupported objective scoring version {scoring_version!r}; "
            f"supported scorers are {', '.join(SUPPORTED_SCORING_VERSIONS)}"
        )


def score_generation(
    record: GenerationRecord, *, scoring_version: str = OBJECTIVE_SCORING_VERSION
) -> ExampleScore:
    require_supported_scoring_version(scoring_version)
    problem = _unscorable_reason(record)
    if problem is not None:
        return _invalid(record, scoring_version, problem)
    if scoring_version == DOCTEST_V1_SCORING_VERSION:
        return score_doctest(record)
    return _score_exact_match(record, scoring_version)


def _unscorable_reason(record: GenerationRecord) -> str | None:
    if record.status != "ok":
        return record.reason or record.status
    if record.expected_completion is None or record.completion is None:
        return "missing expected or predicted completion"
    if normalize_completion(record.expected_completion) == "":
        return "empty expected completion"
    return None


def _score_exact_match(record: GenerationRecord, scoring_version: str) -> ExampleScore:
    if record.expected_completion is None or record.completion is None:
        raise ValueError("exact-match scoring needs expected and predicted completions")
    expected = normalize_completion(record.expected_completion)
    predicted = normalize_completion(record.completion)
    if scoring_version == EXACT_MATCH_V2_SCORING_VERSION:
        # Both sides are unwrapped so the verdict cannot depend on which one carries a fence.
        expected = normalize_completion(extract_fenced_code(expected))
        if expected == "":
            return _invalid(record, scoring_version, "empty expected completion")
        predicted = normalize_completion(extract_fenced_code(predicted))
    return ExampleScore(
        task_id=record.task_id,
        scoring_version=scoring_version,
        verdict="correct" if predicted == expected else "incorrect",
        expected_completion=record.expected_completion,
        predicted_completion=record.completion,
    )


def _invalid(record: GenerationRecord, scoring_version: str, reason: str) -> ExampleScore:
    return ExampleScore(
        task_id=record.task_id,
        scoring_version=scoring_version,
        verdict="invalid",
        expected_completion=record.expected_completion,
        predicted_completion=record.completion,
        reason=reason,
    )


def score_doctest(
    record: GenerationRecord, *, timeout_seconds: float = DOCTEST_TIMEOUT_SECONDS
) -> ExampleScore:
    """``doctest-v1``: correct iff every doctest in the (unfenced) candidate module passes.

    The reference module's docstring examples are the task's spec, so a candidate that
    deletes or rewrites one of them is ``incorrect`` instead of passing trivially, and a
    candidate with no examples never passes. Everything the candidate can bring about
    (a syntax error, a timeout, killing the worker, tampering with the harness) is
    ``incorrect``; ``invalid`` is reserved for what it cannot: a reference with no usable
    examples, or a worker that never started. ``record`` must already be ``status="ok"``
    with both completions present (``score_generation`` guarantees that).
    """

    reference, completion = record.expected_completion, record.completion
    if reference is None or completion is None:
        raise ValueError("doctest scoring needs expected and predicted completions")
    try:
        wanted = reference_examples(extract_fenced_code(reference))
    except ValueError as exc:
        return _invalid(record, DOCTEST_V1_SCORING_VERSION, str(exc))
    outcome = run_doctests(extract_fenced_code(completion), timeout_seconds=timeout_seconds)
    dropped = wanted - Counter(outcome.examples)
    verdict, reason = _doctest_verdict(outcome, sum(dropped.values()))
    return ExampleScore(
        task_id=record.task_id,
        scoring_version=DOCTEST_V1_SCORING_VERSION,
        verdict=verdict,
        expected_completion=record.expected_completion,
        predicted_completion=record.completion,
        reason=reason,
        n_examples=outcome.n_examples,
        n_failed=outcome.n_failed,
        error=outcome.error,
    )


_FIXED_STATUS_VERDICTS: dict[str, tuple[ExampleVerdict, str]] = {
    "syntax_error": ("incorrect", "candidate module does not parse"),
    "import_error": ("incorrect", "candidate module raised while loading"),
    "malformed_doctest": ("incorrect", "candidate has a malformed doctest"),
    "tampered": ("incorrect", "candidate modified the scoring harness"),
}


def _doctest_verdict(
    outcome: DoctestOutcome, n_dropped_reference_examples: int
) -> tuple[ExampleVerdict, str | None]:
    if outcome.status == "worker_error":  # the scorer, not the candidate, failed
        return "invalid", outcome.error or outcome.status
    if outcome.status in {"timeout", "crashed"}:
        return "incorrect", outcome.error or outcome.status
    if outcome.status in _FIXED_STATUS_VERDICTS:
        return _FIXED_STATUS_VERDICTS[outcome.status]
    return _executed_verdict(outcome, n_dropped_reference_examples)


def _executed_verdict(
    outcome: DoctestOutcome, n_dropped_reference_examples: int
) -> tuple[ExampleVerdict, str | None]:
    if outcome.n_examples == 0:
        return "incorrect", "candidate has no doctest examples"
    if n_dropped_reference_examples:
        return (
            "incorrect",
            f"candidate dropped or rewrote {n_dropped_reference_examples} reference doctest example(s)",
        )
    if outcome.n_failed:
        return "incorrect", f"{outcome.n_failed} of {outcome.n_examples} doctest example(s) failed"
    return "correct", None


def score_arm(
    role: Literal["causal_base", "causal_sft"],
    records: tuple[GenerationRecord, ...],
    *,
    scoring_version: str = OBJECTIVE_SCORING_VERSION,
) -> tuple[tuple[ExampleScore, ...], ArmMetrics]:
    scores = tuple(score_generation(record, scoring_version=scoring_version) for record in records)
    counts = Counter(score.verdict for score in scores)
    n_correct, n_incorrect, n_invalid = counts["correct"], counts["incorrect"], counts["invalid"]
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
