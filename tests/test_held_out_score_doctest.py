"""The ``doctest-v1`` scorer: sandboxed execution of the candidate module's doctests."""

import doctest
import importlib.util
import json
import os
import signal
import subprocess  # nosec B404 - the tests launch only fixed argv they control
import sys
import time
from pathlib import Path

import pytest

from agoge_forger.eval import _doctest_runner
from agoge_forger.eval._doctest_runner import (
    _kill,
    _read_result,
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

_spec = importlib.util.spec_from_file_location(
    "_doctest_worker_under_test", Path(_doctest_runner.__file__).with_name("_doctest_worker.py")
)
assert _spec is not None and _spec.loader is not None
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)  # only defines functions unless run as a script

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


def test_syntax_error_is_incorrect():
    score = _score("def add(a, b:\n    return a + b\n")
    assert score.verdict == "incorrect"
    assert score.reason == "candidate module does not parse"
    assert "SyntaxError" in (score.error or "")


def test_unclosed_docstring_from_a_truncated_reply_is_incorrect():
    assert _score('```python\ndef add(a, b):\n    """\n    >>> add(1, 2)\n').verdict == "incorrect"


def test_timeout_is_incorrect():
    looping = REFERENCE.replace("return a + b", "while True:\n        pass")
    score = score_doctest(_record(looping), timeout_seconds=1)
    assert score.verdict == "incorrect"
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
        _record(None, task_id="c", status="unsupported", reason="overflow"),
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
    assert not missing_reference_examples(REFERENCE, counted)
    assert sum(missing_reference_examples(REFERENCE, ()).values()) == 2


def test_malformed_doctest_directive_is_a_per_task_verdict_not_an_abort():
    malformed = REFERENCE.replace(">>> add(1, 2)", ">>> add(1, 2)  # doctest: +NOPE")
    assert docstring_examples(malformed) is None  # the static reference path tolerates it too
    score = _score(malformed)
    assert score.verdict == "incorrect"
    assert score.reason == "candidate has a malformed doctest"
    assert "+NOPE" in (score.error or "")


def test_examples_kept_only_in_a_nested_function_do_not_count():
    decoy = '''def add(a, b):
    """
    >>> 1 + 1
    2
    """
    def inner():
        """
        >>> add(1, 2)
        3
        >>> add(0, 0)
        0
        """
    return a - b
'''
    score = _score(decoy)
    assert score.verdict == "incorrect"
    assert score.reason == "candidate dropped or rewrote 2 reference doctest example(s)"


def test_expected_output_whitespace_is_part_of_the_preserved_example():
    reference = 'def f():\n    """\n    >>> print("a  b")\n    a  b\n    """\n    print("a  b")\n'
    squeezed = reference.replace('    a  b\n    """', '    a b\n    """')
    assert squeezed != reference
    assert _score(reference, reference).verdict == "correct"
    assert _score(squeezed, reference).verdict == "incorrect"


def test_reindenting_a_docstring_is_still_the_same_example():
    reindented = (
        REFERENCE.replace("\n    >>>", "\n        >>>")
        .replace("\n    3", "\n        3")
        .replace("\n    0\n", "\n        0\n")
    )
    assert reindented != REFERENCE
    assert _score(reindented).verdict == "correct"


def test_text_that_cannot_be_written_as_utf8_is_incorrect_not_an_abort():
    assert _score(REFERENCE + "\n# \ud800\n").verdict == "incorrect"


_FORGED = (
    "import json, os\n"
    'json.dump({"status": "ok", "n_examples": 5, "n_failed": 0, "error": None}, '
    'open("result.json", "w"))\n'
    "os._exit(0)\n"
)


def test_candidate_cannot_forge_the_result_in_its_working_directory():
    assert run_doctests(BUGGY + _FORGED).status == "crashed"
    assert _score(BUGGY + _FORGED).verdict == "incorrect"


def test_candidate_cannot_write_the_result_file_or_guess_a_token():
    path_forgery = BUGGY + (
        "import json, os, sys\n"
        'forged = {"status": "ok", "n_examples": 5, "n_failed": 0, "error": None, "token": ""}\n'
        "try:\n"
        "    json.dump(forged, open(sys.argv[2], 'w'))\n"
        "except PermissionError:\n"
        "    pass\n"
        "os._exit(0)\n"
    )
    assert run_doctests(path_forgery).status == "crashed"


def test_outcomes_are_not_accepted_without_the_matching_token(tmp_path):
    result = tmp_path / "result.json"
    payload = {"status": "ok", "n_examples": 1, "n_failed": 0, "error": None}
    result.write_text(json.dumps({**payload, "token": "other"}), encoding="utf-8")
    assert _read_result(result, "expected").status == "crashed"
    result.write_text(json.dumps({**payload, "token": "expected"}), encoding="utf-8")
    assert _read_result(result, "expected").status == "ok"
    result.write_text(json.dumps({**payload, "status": "timeout", "token": "expected"}))
    assert _read_result(result, "expected").status == "crashed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file semantics")
def test_descriptor_and_dir_fd_tricks_cannot_touch_files_outside(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    victim.chmod(0o640)
    os.utime(victim, (1_000_000, 1_000_000))
    before = victim.stat()
    source = f"""
import os
VICTIM = {str(victim)!r}
def attempt(action):
    try:
        action()
    except OSError:
        pass
read_fd = os.open(VICTIM, os.O_RDONLY)
dir_fd = os.open({str(tmp_path)!r}, os.O_RDONLY)
attempt(lambda: os.ftruncate(read_fd, 0))
attempt(lambda: os.fchmod(read_fd, 0o777))
attempt(lambda: os.chmod(read_fd, 0o777))
attempt(lambda: os.utime(read_fd, (1, 1)))
attempt(lambda: os.utime(VICTIM, (1, 1)))
attempt(lambda: os.chmod("victim.txt", 0o777, dir_fd=dir_fd))
attempt(lambda: os.truncate(VICTIM, 0))
attempt(lambda: os.setxattr(VICTIM, "user.x", b"1"))
attempt(lambda: os.unlink("victim.txt", dir_fd=dir_fd))
def f():
    '''
    >>> 1 + 1
    2
    '''
"""
    assert run_doctests(source).status == "ok"
    after = victim.stat()
    assert victim.read_text(encoding="utf-8") == "precious"
    assert (after.st_mode, after.st_mtime) == (before.st_mode, before.st_mtime)


def test_the_audit_hook_fails_closed_on_unexpected_event_shapes():
    policy = worker._PathPolicy("/sandbox-root")
    assert worker._denial("os.remove", (), policy) is not None
    assert worker._denial("os.remove", ("/sandbox-root/x", 7), policy) is not None
    assert worker._denial("os.remove", ("/sandbox-root/x", -1), policy) is None
    assert worker._denial("os.chmod", (3, 0o777, -1), policy) is not None
    assert worker._denial("open", (), policy) is None  # no write flags, nothing to refuse
    assert worker._denial("socket.connect", (object(), ("h", 1)), policy) is not None


def test_a_missing_standard_stream_original_does_not_break_the_restore(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "_install_audit_hook", lambda _sandbox: None)
    monkeypatch.setattr(sys, "__stdout__", None)
    monkeypatch.setattr(sys, "__stderr__", None)
    monkeypatch.setitem(sys.modules, "candidate", sys.modules.get("candidate", worker))
    stdout, stderr = sys.stdout, sys.stderr
    result = worker._run("x = 1\n", str(tmp_path), worker._Harness(str(tmp_path / "result")))
    assert result["status"] == "ok"
    assert (sys.stdout, sys.stderr) == (stdout, stderr)


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_kill_falls_back_to_the_worker_when_the_group_cannot_be_signalled(monkeypatch):
    def refuse(_pid, _signal):
        raise PermissionError

    monkeypatch.setattr(os, "killpg", refuse)
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    process = subprocess.Popen(argv, start_new_session=True)  # nosec B603 - fixed argv, no shell
    _kill(process)
    assert process.returncode == -signal.SIGKILL


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_kill_takes_down_the_whole_process_group(tmp_path):
    marker = tmp_path / "child.pid"
    parent = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    argv = [sys.executable, "-c", parent]
    process = subprocess.Popen(argv, start_new_session=True)  # nosec B603 - fixed argv, no shell
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    child_pid = int(marker.read_text(encoding="utf-8"))
    _kill(process)
    time.sleep(0.2)
    assert not _is_running(child_pid)


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # an unreaped (zombie) orphan is dead; /proc tells the two apart on Linux
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state != "Z"


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


# --- Bypasses an independent review executed against the first sandbox (PR #192) ---

_SELF_TEST = 'def f():\n    """\n    >>> 1 + 1\n    2\n    """\n'


def _blocked(module: str, *, outside: Path | None = None) -> None:
    """Score ``module`` against itself; the sandbox must refuse it and keep its verdict."""

    score = score_doctest(_record(module, reference=module))
    assert score.verdict != "correct"
    error = score.error or ""
    assert "sandbox blocked" in error or "halted; None in sys.modules" in error
    if outside is not None:
        assert list(outside.iterdir()) == []


def test_candidate_cannot_forge_the_result_by_reassigning_json_dump():
    forged = BUGGY + (
        "import json\n"
        "json.dump = lambda obj, fp: fp.write(json.dumps({**obj, 'status': 'ok', "
        "'n_examples': 2, 'n_failed': 0, "
        "'examples': [['add(1, 2)\\n', '3\\n'], ['add(0, 0)\\n', '0\\n']]}))\n"
    )
    score = _score(forged)
    assert score.verdict == "incorrect"
    assert score.reason == "candidate modified the scoring harness"


def test_candidate_cannot_forge_the_result_by_patching_the_doctest_runner():
    forged = BUGGY + (
        "import doctest\n"
        "doctest.DocTestRunner.run = "
        "lambda self, test, **kw: setattr(self, 'tries', len(test.examples))\n"
    )
    score = _score(forged)
    assert score.verdict == "incorrect"
    assert score.reason == "candidate modified the scoring harness"


def test_candidate_cannot_forge_the_result_by_patching_the_json_encoder_internals():
    forged = BUGGY + (
        "import json.encoder\n"
        "json.encoder.c_make_encoder = None\n"
        "json.encoder._make_iterencode = lambda *a, **k: (lambda obj, level: "
        "[json.dumps({**obj, 'n_failed': 0})])\n"
    )
    assert _score(forged).verdict == "incorrect"


def test_patching_the_harness_from_inside_an_example_does_not_pass():
    rewritten = REFERENCE.replace(
        ">>> add(0, 0)\n    0",
        ">>> add(0, 0)\n    0\n    >>> import doctest, json\n"
        ">>> doctest.DocTestRunner.failures = property(lambda self: 0)\n",
    )
    score = _score(rewritten.replace("a + b", "a - b"))
    assert score.verdict == "incorrect"


def test_candidate_cannot_disable_the_hook_through_the_worker_module():
    patched = BUGGY + "import __main__\n__main__._denial = lambda *args: None\n"
    score = _score(patched)
    assert score.verdict == "incorrect"
    assert score.reason == "candidate modified the scoring harness"


def test_candidate_cannot_send_udp_datagrams():
    _blocked(
        'def f():\n    """\n    >>> import socket\n'
        "    >>> socket.socket(socket.AF_INET, socket.SOCK_DGRAM)"
        '.sendto(b"x", ("127.0.0.1", 9))\n    1\n    """\n'
    )


def test_candidate_cannot_create_any_socket():
    _blocked('def f():\n    """\n    >>> import socket\n    >>> s = socket.socket()\n    """\n')


def test_candidate_cannot_reach_a_cached_socket_module_through_another_module():
    _blocked(
        'def f():\n    """\n    >>> import pdb, sys\n'
        '    >>> s = sys.modules["pdb"].socket.socket()\n    """\n'
    )


@pytest.mark.parametrize(
    "module",
    ["_posixsubprocess", "subprocess", "socket", "_socket", "ctypes", "sqlite3", "multiprocessing"],
)
def test_candidate_cannot_import_process_network_or_native_modules(module):
    _blocked(f'def f():\n    """\n    >>> import {module}\n    """\n')
    _blocked(
        f'def f():\n    """\n    >>> import importlib\n    >>> importlib.import_module("{module}")\n    """\n'
    )


@pytest.mark.skipif(os.name != "posix", reason="fork_exec is POSIX only")
def test_candidate_cannot_fork_exec_a_shell(tmp_path):
    marker = tmp_path / "pwned"
    module = (
        'def f():\n    """\n    >>> import _posixsubprocess, os\n'
        f'    >>> _posixsubprocess.fork_exec([b"/bin/sh", b"-c", b"touch {marker}"], '
        "[b'/bin/sh'], True, (), None, None, -1, -1, -1, -1, -1, -1, "
        '*os.pipe(), False, False, -1, None, None, None, -1, None)\n    """\n'
    )
    _blocked(module, outside=tmp_path)


def test_candidate_cannot_write_a_database_outside_its_directory(tmp_path):
    module = (
        'def f():\n    """\n    >>> import sqlite3\n'
        f'    >>> sqlite3.connect({str(tmp_path / "x.db")!r}).execute("create table t(x)")\n'
        '    """\n'
    )
    _blocked(module, outside=tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is POSIX only")
def test_candidate_cannot_create_fifos_or_device_nodes_outside_its_directory(tmp_path):
    module = (
        'def f():\n    """\n    >>> import os, posix\n'
        f"    >>> os.mkfifo({str(tmp_path / 'fifo')!r})\n"
        f'    >>> posix.mknod({str(tmp_path / "node")!r})\n    """\n'
    )
    score = score_doctest(_record(module, reference=module))
    assert score.verdict == "incorrect"
    assert "AttributeError" in (score.error or "")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="dir_fd-relative open is POSIX only")
def test_candidate_cannot_write_through_a_dir_fd_relative_open(tmp_path):
    # Open the target directory read-only (allowed), then try to create a file relative to
    # that descriptor. The ``open`` audit event omits ``dir_fd``, so without the wrapper the
    # path resolves against the sandbox cwd and the write escapes into ``tmp_path``.
    marker = "escaped_x"
    module = (
        'def f():\n    """\n    >>> import os\n'
        f"    >>> fd = os.open({str(tmp_path)!r}, os.O_RDONLY)\n"
        f'    >>> os.open({marker!r}, os.O_WRONLY | os.O_CREAT, dir_fd=fd)\n    0\n    """\n'
    )
    score = score_doctest(_record(module, reference=module))
    assert score.verdict != "correct"
    assert "sandbox blocked" in (score.error or "")
    assert not (tmp_path / marker).exists()


def test_legitimate_standard_library_imports_still_score_correct():
    # Blocked imports raise ImportError, so stdlib optional-import probes (pathlib -> ntpath
    # -> _winapi, uuid -> ctypes) fall back instead of failing a valid candidate.
    imports = "pathlib, uuid, unittest.mock, json, re, collections, dataclasses, typing, functools"
    module = f'def f():\n    """\n    >>> import {imports}\n    >>> 1 + 1\n    2\n    """\n'
    assert _score(module, reference=module).verdict == "correct"


def test_a_reference_that_does_not_parse_is_invalid_not_a_disabled_check():
    for reference in ("def add(a, b:\n", 'def add(a, b):\n    """Adds."""\n'):
        score = _score(_SELF_TEST, reference=reference)
        assert score.verdict == "invalid"
        assert "reference" in (score.reason or "")
        assert score.n_examples is None  # the candidate was never run


def test_a_reference_with_a_malformed_directive_is_invalid():
    malformed = REFERENCE.replace(">>> add(1, 2)", ">>> add(1, 2)  # doctest: +NOPE")
    assert _score(REFERENCE, reference=malformed).verdict == "invalid"


def test_killing_the_worker_from_an_example_is_incorrect_not_invalid():
    exiting = 'def f():\n    """\n    >>> import os\n    >>> os._exit(0)\n    """\n'
    score = _score(exiting)
    assert score.verdict == "incorrect"
    assert "without a result" in (score.reason or "")


def test_a_worker_that_never_started_is_invalid(monkeypatch):
    monkeypatch.setattr(_doctest_runner, "_WORKER", Path("/nonexistent/_doctest_worker.py"))
    outcome = run_doctests(REFERENCE)
    assert outcome.status == "worker_error"
    assert _score(REFERENCE).verdict == "invalid"


def test_result_channel_ignores_json_patched_after_it_was_bound(tmp_path, monkeypatch):
    channel = worker._ResultChannel(str(tmp_path / "result.json"))
    forged = {"status": "ok", "n_failed": 0}
    monkeypatch.setattr(json, "dump", lambda obj, fp, **kw: fp.write(json.dumps(forged)))
    monkeypatch.setattr(json, "dumps", lambda obj, **kw: "forged")
    monkeypatch.setattr(json.encoder, "c_make_encoder", None)
    monkeypatch.setattr(json.encoder, "_make_iterencode", lambda *a, **k: lambda o, level: ["{}"])
    channel.publish({"status": "import_error", "n_failed": 1, "error": None, "ok": [1, "x"]})
    written = (tmp_path / "result.json").read_bytes()
    assert written == b'{"status":"import_error","n_failed":1,"error":null,"ok":[1,"x"]}'


def test_harness_snapshot_reports_rebound_and_removed_names_only(monkeypatch):
    snapshot = worker._HarnessSnapshot()
    assert snapshot.changes() == []
    monkeypatch.setattr(json, "brand_new_name", 1, raising=False)  # additions are allowed
    assert snapshot.changes() == []
    monkeypatch.setattr(doctest.DocTestRunner, "run", lambda *a, **k: None)
    monkeypatch.delattr(json, "dumps")
    assert sorted(snapshot.changes()) == ["DocTestRunner.run", "json.dumps"]
