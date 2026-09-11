import pytest

from agoge_forger.eval.compare import compare_arms, compare_example
from agoge_forger.eval.score import ExampleScore


def _score(task_id: str, verdict: str) -> ExampleScore:
    return ExampleScore(
        task_id=task_id,
        scoring_version="exact-match-v1",
        verdict=verdict,
        expected_completion="answer",
        predicted_completion="answer" if verdict == "correct" else "wrong",
    )


def test_compare_example_outcomes():
    assert compare_example(_score("t", "incorrect"), _score("t", "correct")).outcome == "improved"
    assert compare_example(_score("t", "correct"), _score("t", "incorrect")).outcome == "regressed"
    assert compare_example(_score("t", "correct"), _score("t", "correct")).outcome == "tied"
    assert compare_example(_score("t", "incorrect"), _score("t", "incorrect")).outcome == "tied"
    assert compare_example(_score("t", "invalid"), _score("t", "correct")).outcome == "invalid"


def test_compare_arms_conclusions():
    improved = compare_arms((_score("a", "incorrect"),), (_score("a", "correct"),))
    assert improved.conclusion == "improved"
    assert improved.n_improved == 1
    assert improved.delta_accuracy == pytest.approx(1.0)

    regressed = compare_arms((_score("a", "correct"),), (_score("a", "incorrect"),))
    assert regressed.conclusion == "regressed"

    tied = compare_arms((_score("a", "correct"),), (_score("a", "correct"),))
    assert tied.conclusion == "null"

    mixed = compare_arms(
        (_score("a", "incorrect"), _score("b", "correct")),
        (_score("a", "correct"), _score("b", "incorrect")),
    )
    assert mixed.conclusion == "mixed"
    assert mixed.n_improved == 1
    assert mixed.n_regressed == 1

    inconclusive = compare_arms((_score("a", "invalid"),), (_score("a", "invalid"),))
    assert inconclusive.conclusion == "inconclusive"
    assert inconclusive.delta_accuracy is None
