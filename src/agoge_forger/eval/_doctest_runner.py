"""Parent side of the ``doctest-v1`` scorer: run one candidate module's doctests safely.

The candidate is model-generated and therefore untrusted. It is never imported or
executed in this process: it is parsed with :mod:`ast` here, then run by
``_doctest_worker.py`` in a separate interpreter with a wall-clock timeout, a
scrubbed environment, a throwaway working directory, resource limits, and an audit
hook that denies network access and process creation (see the worker docstring for
what that does and does not guarantee).
"""

from __future__ import annotations

import ast
import doctest
import json
import os
import signal
import subprocess  # nosec B404 - launches only this package's own worker script
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DOCTEST_TIMEOUT_SECONDS = 10.0
_WORKER = Path(__file__).with_name("_doctest_worker.py")
_ERROR_CHARS = 500
_WINDOWS_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

DoctestStatus = Literal["ok", "syntax_error", "import_error", "timeout", "crashed"]
ExamplePair = tuple[str, str]


@dataclass(frozen=True)
class DoctestOutcome:
    status: DoctestStatus
    n_examples: int = 0
    n_failed: int = 0
    error: str | None = None


def docstring_examples(source: str) -> Counter[ExamplePair] | None:
    """Count the ``(source, want)`` doctest examples in a module's docstrings.

    Purely static (no code runs). Returns ``None`` when ``source`` does not parse.
    Every module, class, and function docstring is included, nested ones too.
    """

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    parser = doctest.DocTestParser()
    examples: Counter[ExamplePair] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring is None:
            continue
        for example in parser.get_examples(docstring):
            examples[(example.source.strip(), example.want.strip())] += 1
    return examples


def missing_reference_examples(reference: str, candidate: str) -> Counter[ExamplePair]:
    """Reference doctest examples that the candidate dropped or rewrote."""

    wanted = docstring_examples(reference) or Counter()
    present = docstring_examples(candidate) or Counter()
    return wanted - present


def run_doctests(
    source: str, *, timeout_seconds: float = DOCTEST_TIMEOUT_SECONDS
) -> DoctestOutcome:
    """Run ``doctest`` over ``source`` in a sandboxed subprocess."""

    with tempfile.TemporaryDirectory(prefix="agoge-doctest-") as sandbox:
        source_path = Path(sandbox) / "candidate.py"
        result_path = Path(sandbox) / "result.json"
        source_path.write_text(source, encoding="utf-8")
        completed = _run_worker(sandbox, source_path, result_path, timeout_seconds)
        if completed is not None:
            return completed
        return _read_result(result_path)


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


def _run_worker(
    sandbox: str, source_path: Path, result_path: Path, timeout_seconds: float
) -> DoctestOutcome | None:
    """Launch the worker; return an outcome only when it timed out or crashed."""

    command = [
        sys.executable,
        "-P",
        "-s",
        str(_WORKER),
        str(source_path),
        str(result_path),
        str(int(timeout_seconds) + 2),
    ]
    with subprocess.Popen(  # nosec B603 - fixed argv, no shell, scrubbed environment
        command,
        cwd=sandbox,
        env=_worker_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix",
        creationflags=_WINDOWS_PROCESS_GROUP if os.name != "posix" else 0,
    ) as process:
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill(process)
            return DoctestOutcome("timeout", error=f"timed out after {timeout_seconds:g}s")
    if not result_path.is_file():
        return DoctestOutcome(
            "crashed",
            error=f"doctest worker exited with code {process.returncode} without a result",
        )
    return None


def _kill(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    process.wait()


def _read_result(result_path: Path) -> DoctestOutcome:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return DoctestOutcome(
            status=payload["status"],
            n_examples=int(payload["n_examples"]),
            n_failed=int(payload["n_failed"]),
            error=payload.get("error"),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return DoctestOutcome(
            "crashed", error=f"unreadable doctest worker result: {exc}"[:_ERROR_CHARS]
        )
