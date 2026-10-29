"""Fence handling: ``extract_fenced_code`` and the ``exact-match-v2`` scorer (#183)."""

import pytest

from agoge_forger.eval._fence import extract_fenced_code
from agoge_forger.eval.score import (
    EXACT_MATCH_V2_SCORING_VERSION,
    OBJECTIVE_SCORING_VERSION,
    GenerationRecord,
    score_arm,
    score_generation,
)

CODE = 'def f(x):\n    """\n    >>> f(1)\n    2\n    """\n    return x + 1'


def _record(completion, expected=CODE, **overrides) -> GenerationRecord:
    payload = {
        "task_id": "task-001",
        "prompt": "Fix:\n",
        "expected_completion": expected,
        "completion": completion,
        "status": "ok",
    }
    payload.update(overrides)
    return GenerationRecord(**payload)


def _v2(completion, expected=CODE):
    return score_generation(
        _record(completion, expected), scoring_version=EXACT_MATCH_V2_SCORING_VERSION
    )


@pytest.mark.parametrize(
    "text",
    [
        CODE,
        CODE + "\n",
        f"```python\n{CODE}\n```",
        f"```python\n{CODE}\n```\n",
        f"```\n{CODE}\n```",
        f"```py\n{CODE}\n```",
        f"```Python  \n{CODE}\n```",
        f"~~~python\n{CODE}\n~~~",
        f"\n\n```python\n{CODE}\n```\n\n",
        f"  ```python\n{CODE}\n  ```   \n",
        f"````python\n{CODE}\n````",
        f"```python\r\n{CODE.replace(chr(10), chr(13) + chr(10))}\r\n```\r\n",
    ],
    ids=[
        "no-fence",
        "no-fence-trailing-newline",
        "python-tag",
        "python-tag-trailing-newline",
        "no-language-tag",
        "short-language-tag",
        "tag-with-trailing-space",
        "tilde-fence",
        "surrounding-blank-lines",
        "indented-fence",
        "four-backtick-fence",
        "crlf",
    ],
)
def test_wrapper_variants_extract_to_the_same_code(text):
    assert extract_fenced_code(text) == CODE


def test_unfenced_replies_are_returned_as_v1_would_see_them():
    # Only CRLF -> LF and trailing whitespace change: leading blank lines and the
    # indentation of the first line survive, so v2 stays "v1 plus fence unwrapping".
    assert extract_fenced_code("\n\n    x = 1\n") == "\n\n    x = 1"
    assert extract_fenced_code("x = 1\r\ny = 2\r\n") == "x = 1\ny = 2"


def test_lone_cr_line_endings_still_find_the_fence():
    assert extract_fenced_code("```python\rx = 1\ry = 2\r```") == "x = 1\ny = 2"


def test_v2_does_not_equate_unfenced_replies_that_differ_by_leading_blank_lines():
    assert _v2(f"\n\n{CODE}").verdict == "incorrect"


def test_longer_fence_can_wrap_a_literal_fence_line():
    inner = 'def f():\n    """\n    ```\n    """\n    return 1'
    assert extract_fenced_code(f"````python\n{inner}\n````") == inner


def test_first_fenced_block_wins_over_later_blocks_and_prose():
    reply = (
        f"```python\n{CODE}\n```\n\n### Explanation\nChanged the return.\n\n```python\nx = 1\n```"
    )
    assert extract_fenced_code(reply) == CODE


def test_unterminated_fence_runs_to_the_end():
    # A reply cut off by max_new_tokens before the closing fence.
    assert extract_fenced_code(f"```python\n{CODE}\n") == CODE


def test_prose_before_the_fence_is_not_unwrapped():
    reply = f"Here is the fix:\n```python\n{CODE}\n```"
    assert extract_fenced_code(reply) == reply


@pytest.mark.parametrize("text", ["", "\n\n", "```\n```", "```python\n\n```"])
def test_empty_replies_extract_to_empty(text):
    assert extract_fenced_code(text) == ""


def test_inline_backticks_are_not_a_fence():
    assert extract_fenced_code("```python```\nx = 1") == "```python```\nx = 1"


def test_v2_scores_fenced_and_unfenced_identically():
    for text in (CODE, f"```python\n{CODE}\n```", f"```\n{CODE}\n```\n"):
        score = _v2(text)
        assert score.verdict == "correct"
        assert score.scoring_version == EXACT_MATCH_V2_SCORING_VERSION
        assert score.predicted_completion == text  # the raw reply is recorded, not the cut


def test_v1_is_unchanged_and_still_confounded_by_fences():
    fenced = _record(f"```python\n{CODE}\n```")
    assert score_generation(fenced).verdict == "incorrect"
    assert score_generation(fenced, scoring_version=OBJECTIVE_SCORING_VERSION).verdict == (
        "incorrect"
    )
    assert score_generation(_record(CODE + "\n")).verdict == "correct"


def test_v2_still_rejects_wrong_code_inside_a_fence():
    wrong = CODE.replace("x + 1", "x + 2")
    assert _v2(f"```python\n{wrong}\n```").verdict == "incorrect"


def test_v2_is_not_lenient_about_code_beyond_the_wrapper():
    assert _v2(f"```python\n{CODE}\nprint('hi')\n```").verdict == "incorrect"
    assert _v2(f"```python\n{CODE.replace('    return', '  return')}\n```").verdict == "incorrect"
    assert _v2(f"```python\n{CODE.upper()}\n```").verdict == "incorrect"


def test_v2_ignores_chatter_after_the_first_block_but_not_a_wrong_first_block():
    wrong = CODE.replace("x + 1", "x + 2")
    after = f"\n\n### Explanation\nI changed things.\n\n```python\n{CODE}\n```"
    assert _v2(f"```python\n{CODE}\n```{after}").verdict == "correct"
    assert _v2(f"```python\n{wrong}\n```{after}").verdict == "incorrect"


def test_v2_unwraps_a_fenced_reference_too():
    fenced_reference = f"```python\n{CODE}\n```"
    assert _v2(CODE, expected=fenced_reference).verdict == "correct"
    assert _v2(f"```python\n{CODE}\n```", expected=fenced_reference).verdict == "correct"


def test_v2_keeps_the_invalid_rules_of_v1():
    assert _v2(None).verdict == "invalid"
    assert _v2(CODE, expected="   ").verdict == "invalid"
    assert _v2(CODE, expected="```python\n```").verdict == "invalid"
    unsupported = score_generation(
        _record(None, status="unsupported", reason="overflow"),
        scoring_version=EXACT_MATCH_V2_SCORING_VERSION,
    )
    assert unsupported.verdict == "invalid"
    assert unsupported.reason == "overflow"


def test_arm_metrics_reflect_fence_invariance():
    records = (
        _record(f"```python\n{CODE}\n```", task_id="a"),
        _record(CODE, task_id="b"),
        _record("```python\nx = 1\n```", task_id="c"),
    )
    _, v1 = score_arm("causal_base", records)
    _, v2 = score_arm("causal_base", records, scoring_version=EXACT_MATCH_V2_SCORING_VERSION)
    assert (v1.n_correct, v1.n_incorrect) == (1, 2)
    assert (v2.n_correct, v2.n_incorrect) == (2, 1)
    assert v2.scoring_version == EXACT_MATCH_V2_SCORING_VERSION


def test_unknown_scoring_version_is_rejected():
    with pytest.raises(ValueError, match="unsupported objective scoring version"):
        score_generation(_record(CODE), scoring_version="heuristic-judge-v1")
