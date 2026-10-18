"""The ``doctest-v1`` scorer: sandboxed execution of the candidate module's doctests."""

import os

import pytest

from agoge_forger.eval._doctest_runner import (
    docstring_examples,
    missing_reference_examples,
    run_doctests,
)
from agoge_forger.eval.score import (
    DOCTEST_V1_SCORING_VERSION,
    GenerationRecord,
    score_arm,
    score_doctest,
    score_generation,
)

REFERENCE = '''def add(a, b):
    """
    >>> add(1, 2)
    3
    >>> add(0, 0)
    0
    """
    return a + b
'''
BUGGY = REFERENCE.replace("a + b", "a - b")


def _record(completion, reference=REFERENCE, **overrides) -> GenerationRecord:
    payload = {
        "task_id": "task-001",
        "prompt": "Fix:\n",
        "expected_completion": reference,
        "completion": completion,
        "status": "ok",
    }
    payload.update(overrides)
    return GenerationRecord(**payload)


def _score(completion, reference=REFERENCE):
    return score_generation(
        _record(completion, reference), scoring_version=DOCTEST_V1_SCORING_VERSION
    )


def test_passing_module_is_correct_with_counts():
    score = _score(REFERENCE)
    assert score.verdict == "correct"
    assert score.scoring_version == DOCTEST_V1_SCORING_VERSION
    assert (score.n_examples, score.n_failed, score.error) == (2, 0, None)


def test_fenced_and_unfenced_candidates_score_identically():
    fenced = _score(f"```python\n{REFERENCE}```")
    plain = _score(REFERENCE)
    assert fenced.verdict == plain.verdict == "correct"
    assert (fenced.n_examples, fenced.n_failed) == (plain.n_examples, plain.n_failed)


def test_a_different_correct_fix_passes_even_though_exact_match_would_not():
    alternative = REFERENCE.replace("return a + b", "return b + a")
    assert _score(alternative).verdict == "correct"


def test_module_failing_one_doctest_is_incorrect():
    flawed = REFERENCE.replace("return a + b", "return a + b if a else 99")
    score = _score(flawed)
    assert score.verdict == "incorrect"
    assert (score.n_examples, score.n_failed) == (2, 1)
    assert score.error is not None
    assert "add(0, 0)" in score.error
    assert "1 of 2" in (score.reason or "")


def test_unfixed_module_is_incorrect():
    score = _score(BUGGY)
    assert (score.verdict, score.n_examples, score.n_failed) == ("incorrect", 2, 1)


def test_syntax_error_is_invalid():
    score = _score("def add(a, b:\n    return a + b\n")
    assert score.verdict == "invalid"
    assert score.reason == "candidate module does not parse"
    assert "SyntaxError" in (score.error or "")


def test_unclosed_docstring_from_a_truncated_reply_is_invalid():
    assert _score('```python\ndef add(a, b):\n    """\n    >>> add(1, 2)\n').verdict == "invalid"


def test_timeout_is_invalid():
    looping = REFERENCE.replace("return a + b", "while True:\n        pass")
    score = score_doctest(_record(looping), timeout_seconds=1)
    assert score.verdict == "invalid"
    assert "timed out after 1s" in (score.reason or "")


def test_infinite_loop_at_import_is_invalid_not_a_hang():
    outcome = run_doctests("while True:\n    pass\n", timeout_seconds=1)
    assert outcome.status == "timeout"


def test_exception_while_loading_the_module_is_incorrect():
    score = _score("raise RuntimeError('boom')\n" + REFERENCE)
    assert score.verdict == "incorrect"
    assert "boom" in (score.error or "")


def test_sys_exit_while_loading_is_contained():
    assert _score("import sys\nsys.exit(0)\n" + REFERENCE).verdict == "incorrect"


def test_deleting_the_doctests_does_not_pass():
    stripped = 'def add(a, b):\n    """Adds."""\n    return a + b\n'
    score = _score(stripped)
    assert score.verdict == "incorrect"
    assert score.n_examples == 0
    assert score.reason == "candidate has no doctest examples"


def test_rewriting_an_example_to_match_the_bug_does_not_pass():
    rewritten = BUGGY.replace(">>> add(1, 2)\n    3", ">>> add(1, 2)\n    -1")
    score = _score(rewritten)
    assert score.verdict == "incorrect"
    assert "dropped or rewrote 1 reference doctest example" in (score.reason or "")


def test_extra_examples_are_allowed():
    extended = REFERENCE.replace(
        ">>> add(0, 0)\n    0", ">>> add(0, 0)\n    0\n    >>> add(2, 2)\n    4"
    )
    score = _score(extended)
    assert (score.verdict, score.n_examples) == ("correct", 3)


def test_status_and_missing_text_follow_the_other_scorers():
    unsupported = _record(None, status="unsupported", reason="overflow")
    assert score_generation(unsupported, scoring_version=DOCTEST_V1_SCORING_VERSION).verdict == (
        "invalid"
    )
    assert _score(REFERENCE, reference="  ").verdict == "invalid"


def test_arm_metrics_count_invalid_separately():
    records = (
        _record(REFERENCE, task_id="a"),
        _record(BUGGY, task_id="b"),
        _record("def (:", task_id="c"),
    )
    _, metrics = score_arm("causal_sft", records, scoring_version=DOCTEST_V1_SCORING_VERSION)
    assert (metrics.n_correct, metrics.n_incorrect, metrics.n_invalid) == (1, 1, 1)
    assert metrics.accuracy == pytest.approx(0.5)


def test_candidate_cannot_open_a_network_connection():
    module = (
        'def f():\n    """\n    >>> import socket\n'
        '    >>> socket.create_connection(("127.0.0.1", 9), timeout=1)\n    """\n'
    )
    score = score_doctest(_record(module, reference=module), timeout_seconds=10)
    assert score.verdict == "incorrect"
    assert "sandbox blocked" in (score.error or "")


def test_candidate_cannot_spawn_processes():
    module = 'def f():\n    """\n    >>> import os\n    >>> os.system("echo hi")\n    """\n'
    score = score_doctest(_record(module, reference=module))
    assert score.verdict == "incorrect"
    assert "sandbox blocked" in (score.error or "")


def test_candidate_cannot_write_outside_its_directory(tmp_path):
    target = tmp_path / "escaped.txt"
    module = f'def f():\n    """\n    >>> open({str(target)!r}, "w").write("x")\n    """\n'
    score = score_doctest(_record(module, reference=module))
    assert score.verdict == "incorrect"
    assert "sandbox blocked" in (score.error or "")
    assert not target.exists()


def test_candidate_does_not_inherit_the_callers_environment(monkeypatch):
    monkeypatch.setenv("AGOGE_TEST_SECRET", "hunter2")
    module = 'def f():\n    """\n    >>> import os\n    >>> os.environ.get("AGOGE_TEST_SECRET")\n    """\n'
    assert score_doctest(_record(module, reference=module)).verdict == "correct"


def test_noisy_modules_do_not_corrupt_the_result():
    noisy = 'print("x" * 100000)\nimport sys\nprint("y", file=sys.stderr)\n' + REFERENCE
    assert _score(noisy).verdict == "correct"


def test_runs_do_not_leave_files_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _score(REFERENCE)
    assert list(tmp_path.iterdir()) == []


def test_example_extraction_is_static_and_tolerates_bad_source():
    assert docstring_examples("def (:") is None
    counted = docstring_examples(REFERENCE)
    assert counted is not None
    assert sum(counted.values()) == 2
    assert not missing_reference_examples(REFERENCE, REFERENCE)
    assert sum(missing_reference_examples(REFERENCE, "x = 1").values()) == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX resource limits")
def test_memory_bomb_is_contained():
    bomb = REFERENCE.replace("return a + b", "return bytearray(8 * 1024**3)")
    score = score_doctest(_record(bomb), timeout_seconds=10)
    assert score.verdict == "incorrect"


def test_worker_runs_with_safe_path_so_cwd_is_not_importable():
    module = (
        'def f():\n    """\n    >>> import sys\n    >>> sys.flags.safe_path\n    True\n    """\n'
    )
    assert score_doctest(_record(module, reference=module)).verdict == "correct"
