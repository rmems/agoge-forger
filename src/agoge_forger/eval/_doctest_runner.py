"""Parent side of the ``doctest-v1`` scorer: run one candidate module's doctests safely.

The candidate is model-generated and therefore untrusted. It is never imported or
executed in this process: it is parsed with :mod:`ast` here, then run by
``_doctest_worker.py`` in a separate interpreter with a wall-clock timeout, a
scrubbed environment, a throwaway working directory, resource limits, and an audit
hook that denies socket use, process creation, and the imports that reach them (see
the worker docstring for what that does and does not guarantee).
"""

from __future__ import annotations

import ast
import doctest
import json
import os
import secrets
import signal
import subprocess  # nosec B404 - launches only this package's own worker script
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DOCTEST_TIMEOUT_SECONDS = 10.0
_WORKER = Path(__file__).with_name("_doctest_worker.py")
_ERROR_CHARS = 500
_WINDOWS_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

DoctestStatus = Literal[
    "ok",
    "syntax_error",
    "import_error",
    "malformed_doctest",
    "tampered",
    "timeout",
    "crashed",
    "worker_error",
]
# Statuses the worker itself may report. ``timeout`` and ``crashed`` (the worker started
# and then died or left an unusable result) and ``worker_error`` (it never started, so
# the candidate was not at fault) are decided here.
_WORKER_STATUSES = frozenset(
    {"ok", "syntax_error", "import_error", "malformed_doctest", "tampered"}
)
ExamplePair = tuple[str, str]


@dataclass(frozen=True)
class DoctestOutcome:
    status: DoctestStatus
    n_examples: int = 0
    n_failed: int = 0
    error: str | None = None
    # The ``(source, want)`` pairs the worker's runner actually executed.
    examples: tuple[ExamplePair, ...] = ()


def _documented_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Yield the nodes whose docstrings ``doctest.DocTestFinder`` would discover.

    The module, its top-level functions and classes, and the members of those classes
    (recursively). Functions nested inside a function body are never discovered, so a
    docstring there is not an example anyone runs.
    """

    yield node
    if isinstance(node, ast.Module | ast.ClassDef):
        for child in node.body:
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                yield from _documented_nodes(child)


def _node_examples(node: ast.AST, parser: doctest.DocTestParser) -> list[doctest.Example]:
    """The doctest examples in ``node``'s own docstring; empty when it has none.

    Raises ``ValueError`` for a malformed ``# doctest:`` directive, exactly as
    ``DocTestParser.get_examples`` does; the caller turns that into a rejected module.
    """

    if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return []
    docstring = ast.get_docstring(node, clean=False)
    if docstring is None:
        return []
    return parser.get_examples(docstring)


def docstring_examples(source: str) -> Counter[ExamplePair] | None:
    """Count the ``(source, want)`` doctest examples ``doctest`` would discover in a module.

    Purely static (no code runs), so it is meant for trusted text such as the reference.
    Values are compared exactly as the doctest parser yields them (already dedented), so
    whitespace in the expected output counts. Returns ``None`` when ``source`` does not
    parse or its doctests are malformed.
    """

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    parser = doctest.DocTestParser()
    examples: Counter[ExamplePair] = Counter()
    try:
        for node in _documented_nodes(tree):
            for example in _node_examples(node, parser):
                # A ``# doctest: +SKIP`` example never runs, so ``doctest`` never reports it
                # and it is not part of the must-run spec: requiring it would mark every
                # candidate (even one identical to the reference) as having dropped it.
                if example.options.get(doctest.SKIP):
                    continue
                examples[(example.source, example.want)] += 1
    except ValueError:  # a malformed ``# doctest:`` directive anywhere in the module
        return None
    return examples


def reference_examples(reference: str) -> Counter[ExamplePair]:
    """The doctest examples a reference module specifies; the task's spec.

    Raises ``ValueError`` when the reference does not parse, has a malformed doctest, or
    has no examples at all: a reference like that cannot check a candidate, and silently
    checking nothing would let any candidate with one passing self-written example pass.
    """

    examples = docstring_examples(reference)
    if examples is None:
        raise ValueError("reference module does not parse or has a malformed doctest")
    if not examples:
        raise ValueError("reference module has no doctest examples")
    return examples


def missing_reference_examples(reference: str, ran: Iterable[ExamplePair]) -> Counter[ExamplePair]:
    """Reference doctest examples that the candidate did not actually run.

    ``ran`` is what the sandboxed worker's runner executed in the candidate
    (``DoctestOutcome.examples``), so an example that was dropped, rewritten, or moved
    somewhere ``doctest`` never looks (a nested function, dead code) counts as missing.
    Raises ``ValueError`` for a reference ``reference_examples`` rejects.
    """

    return reference_examples(reference) - Counter(ran)


def run_doctests(
    source: str, *, timeout_seconds: float = DOCTEST_TIMEOUT_SECONDS
) -> DoctestOutcome:
    """Run ``doctest`` over ``source`` in a sandboxed subprocess."""

    try:
        source.encode("utf-8")
    except UnicodeEncodeError as exc:  # e.g. a lone surrogate from a bad JSON escape
        return DoctestOutcome("syntax_error", error=f"not valid UTF-8 text: {exc.reason}")
    # The result goes through ``channel``, a directory the candidate may read but whose
    # contents the sandbox never lets it write; ``sandbox`` is the only writable place.
    with (
        tempfile.TemporaryDirectory(prefix="agoge-doctest-") as sandbox,
        tempfile.TemporaryDirectory(prefix="agoge-doctest-result-") as channel,
    ):
        launch = _WorkerLaunch(
            sandbox=sandbox,
            source_path=Path(sandbox) / "candidate.py",
            result_path=Path(channel) / "result.json",
            token=secrets.token_hex(16),
        )
        launch.source_path.write_text(source, encoding="utf-8")
        completed = _run_worker(launch, timeout_seconds)
        if completed is not None:
            return completed
        return _read_result(launch.result_path, launch.token)


def _worker_environment() -> dict[str, str]:
    # Nothing from the caller's environment (tokens, proxies, PYTHON* switches) is passed on.
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",  # set/dict ordering must not make a doctest flaky
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    for name in ("SYSTEMROOT", "SYSTEMDRIVE"):  # required for the interpreter to start on Windows
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


@dataclass(frozen=True)
class _WorkerLaunch:
    """Where one worker run reads its source, writes its result, and what token it echoes."""

    sandbox: str
    source_path: Path
    result_path: Path
    token: str


def _run_worker(launch: _WorkerLaunch, timeout_seconds: float) -> DoctestOutcome | None:
    """Launch the worker; return an outcome only when it timed out or crashed."""

    command = [
        sys.executable,
        "-P",
        "-s",
        str(_WORKER),
        str(launch.source_path),
        str(launch.result_path),
        str(int(timeout_seconds) + 2),
    ]
    with subprocess.Popen(  # nosec B603  # nosemgrep - fixed argv, no shell, scrubbed environment
        command,
        cwd=launch.sandbox,
        env=_worker_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix",
        creationflags=_WINDOWS_PROCESS_GROUP if os.name != "posix" else 0,
    ) as process:
        try:
            # The token travels on stdin so it is not visible in argv, environ, or /proc.
            process.communicate(input=launch.token.encode("ascii"), timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill(process)
            return DoctestOutcome("timeout", error=f"timed out after {timeout_seconds:g}s")
    # ``Popen.__exit__`` has waited for the worker, so ``returncode`` is set here. The
    # worker creates the result file before the candidate runs, so a missing file means it
    # never got that far; an empty or unreadable one means the candidate took it down.
    if not launch.result_path.is_file():
        return DoctestOutcome(
            "worker_error",
            error=f"doctest worker exited with code {process.returncode} before it started",
        )
    if launch.result_path.stat().st_size == 0:
        return DoctestOutcome(
            "crashed",
            error=f"doctest worker exited with code {process.returncode} without a result",
        )
    return None


def _kill(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL the worker's whole process group (it leads its own session), then reap it."""

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except OSError:  # the group is gone or not ours to signal: take down the worker itself
        process.kill()
    process.wait()


def _read_result(result_path: Path, token: str) -> DoctestOutcome:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload["token"] != token or payload["status"] not in _WORKER_STATUSES:
            raise ValueError("result was not written by the worker for this run")
        return DoctestOutcome(
            status=payload["status"],
            n_examples=int(payload["n_examples"]),
            n_failed=int(payload["n_failed"]),
            error=payload.get("error"),
            examples=tuple(
                (str(source), str(want)) for source, want in payload.get("examples", ())
            ),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return DoctestOutcome(
            "crashed", error=f"unreadable doctest worker result: {exc}"[:_ERROR_CHARS]
        )
