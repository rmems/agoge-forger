import pytest

from agoge_forger.eval.score import (
    OBJECTIVE_SCORING_VERSION,
    GenerationRecord,
    score_arm,
    score_generation,
)


def _record(**overrides) -> GenerationRecord:
    payload = {
        "task_id": "task-001",
        "prompt": "Fix:\n",
        "expected_completion": "answer",
        "completion": "answer",
        "status": "ok",
    }
    payload.update(overrides)
    return GenerationRecord(**payload)


def test_exact_match_accepts_trailing_whitespace_only():
    score = score_generation(_record(completion="answer\n"))
    assert score.verdict == "correct"
    assert score.scoring_version == OBJECTIVE_SCORING_VERSION


def test_exact_match_is_case_sensitive():
    assert score_generation(_record(completion="Answer")).verdict == "incorrect"


def test_unsupported_and_missing_outputs_are_invalid():
    unsupported = score_generation(
        _record(status="unsupported", completion=None, reason="overflow")
    )
    missing = score_generation(_record(completion=None))
    assert unsupported.verdict == "invalid"
    assert missing.verdict == "invalid"


def test_empty_expected_completion_is_invalid():
    assert score_generation(_record(expected_completion="   ")).verdict == "invalid"


def test_score_arm_metrics_exclude_invalid_from_accuracy():
    records = (
        _record(task_id="a", completion="answer"),
        _record(task_id="b", completion="wrong"),
        _record(task_id="c", status="unsupported", completion=None, reason="overflow"),
    )
    _scores, metrics = score_arm("causal_base", records)
    assert metrics.n_correct == 1
    assert metrics.n_incorrect == 1
    assert metrics.n_invalid == 1
    assert metrics.n_scored == 2
    assert metrics.accuracy == pytest.approx(0.5)
