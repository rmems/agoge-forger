"""Subprocess entry point for the ``doctest-v1`` scorer.

Run as ``python -P -s _doctest_worker.py SOURCE RESULT CPU_SECONDS`` with a one-time
token on stdin. It is standalone (standard library only) and is never imported by the
parent process, because it executes model-generated code. The parent supplies the
timeout, a scrubbed environment, and a throwaway working directory; this process adds
best-effort containment on top of that:

* POSIX resource limits (CPU time, address space, file size, core dumps);
* an audit hook that refuses network access, process creation, ``ctypes`` library
  loading, and filesystem mutation outside the working directory. Importing a compiled
  extension module is **not** intercepted (CPython raises no dedicated audit event for
  it), so code that already has a ``.so`` on ``sys.path`` can still load it.

The result is reported through ``RESULT``, a file in a *parent-owned directory outside
the working directory*, opened before the candidate runs and stamped with the token the
parent sent on stdin. The audit hook refuses writes outside the working directory, so
candidate code cannot create or replace that file by path, and a result without the
right token is discarded by the parent. What this does not stop: candidate code that
deliberately digs the already-open file object or the token out of this process's frames
or descriptors. That is the same class of attack as escaping the hook (see below).

This is a guard against accidents (a generated module that opens a socket, spawns a
process, or writes outside its directory), **not** a security boundary against code
written to escape it. Run the scorer inside a container or VM if the generations
are adversarial.
"""

from __future__ import annotations

import doctest
import json
import os
import sys
import types

_MEMORY_LIMIT_BYTES = 2 * 1024**3
_FILE_SIZE_LIMIT_BYTES = 4 * 1024**2
_ERROR_CHARS = 500

_BLOCKED_EVENTS = frozenset(
    {
        "ctypes.dlopen",
        "ctypes.dlsym",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.startfile",
        "os.system",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.getnameinfo",
        "socket.sendmsg",
        "subprocess.Popen",
    }
)
# Events whose leading arguments are filesystem paths that must stay in the sandbox:
# event -> (number of leading path arguments, positions of ``dir_fd`` arguments). A path
# given as a bare descriptor (``os.fchmod``) or relative to another directory descriptor
# (``dir_fd=``) cannot be vetted against the sandbox root, so both are refused.
_PATH_EVENTS: dict[str, tuple[int, tuple[int, ...]]] = {
    "os.chmod": (1, (2,)),
    "os.chown": (1, (3,)),
    "os.link": (2, (2, 3)),
    "os.mkdir": (1, (2,)),
    "os.remove": (1, (1,)),
    "os.removexattr": (1, ()),
    "os.rename": (2, (2, 3)),
    "os.rmdir": (1, (1,)),
    "os.setxattr": (1, ()),
    "os.symlink": (2, (2,)),
    "os.truncate": (1, ()),
    "os.utime": (1, (3,)),
    "shutil.copyfile": (2, ()),
    "shutil.move": (2, ()),
    "shutil.rmtree": (1, (2,)),
}
_NO_DIR_FD = (None, -1)
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _apply_resource_limits(cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:  # Windows has no ``resource``; the parent timeout still applies.
        return
    limits = [
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_AS, _MEMORY_LIMIT_BYTES),
        (resource.RLIMIT_FSIZE, _FILE_SIZE_LIMIT_BYTES),
        (resource.RLIMIT_CORE, 0),
    ]
    for kind, value in limits:
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError):  # e.g. a hard limit already below the request
            continue


class _PathPolicy:
    """Decides whether a filesystem path stays inside the sandbox directory."""

    def __init__(self, sandbox: str) -> None:
        self._root = os.path.realpath(sandbox)
        self._devnull = {os.path.realpath(os.devnull), os.devnull}

    def allows(self, path: object, *, allow_descriptor: bool = False) -> bool:
        if isinstance(path, int):
            # ``open(fd)`` re-wraps a descriptor that was vetted when it was opened; a
            # descriptor handed to a mutating call cannot be tied back to a path.
            return allow_descriptor
        real = _real_path(path)
        return real is not None and self._contains(real)

    def _contains(self, real: str) -> bool:
        return real in self._devnull or real == self._root or real.startswith(self._root + os.sep)


def _real_path(path: object) -> str | None:
    if not isinstance(path, str | bytes | os.PathLike):
        return None
    try:
        return os.path.realpath(os.fsdecode(path))
    except ValueError:
        return None


def _open_flags(args: tuple[object, ...]) -> int:
    flags = args[2] if len(args) > 2 else 0
    return flags if isinstance(flags, int) else 0


def _open_denial(args: tuple[object, ...], policy: _PathPolicy) -> str | None:
    path = args[0] if args else None  # no recognisable path: treated as outside the sandbox
    if _open_flags(args) & _WRITE_FLAGS and not policy.allows(path, allow_descriptor=True):
        return "doctest-v1 sandbox blocked a write outside its directory"
    return None


def _path_event_denial(event: str, args: tuple[object, ...], policy: _PathPolicy) -> str | None:
    path_count, dir_fd_positions = _PATH_EVENTS[event]
    if len(args) < path_count:  # an argument shape we do not recognise fails closed
        return f"doctest-v1 sandbox blocked {event} outside its directory"
    uses_dir_fd = any(
        position < len(args) and args[position] not in _NO_DIR_FD for position in dir_fd_positions
    )
    if not uses_dir_fd and all(policy.allows(path) for path in args[:path_count]):
        return None
    return f"doctest-v1 sandbox blocked {event} outside its directory"


def _denial(event: str, args: tuple[object, ...], policy: _PathPolicy) -> str | None:
    """Return why the sandbox refuses ``event``, or ``None`` when it is allowed."""

    if event in _BLOCKED_EVENTS:
        return f"doctest-v1 sandbox blocked {event}"
    if event == "open":
        return _open_denial(args, policy)
    if event in _PATH_EVENTS:
        return _path_event_denial(event, args, policy)
    return None


def _install_audit_hook(sandbox: str) -> None:
    policy = _PathPolicy(sandbox)

    def hook(event: str, args: tuple[object, ...]) -> None:
        denial = _denial(event, args, policy)
        if denial is not None:
            raise PermissionError(denial)

    sys.addaudithook(hook)


class _RecordingRunner(doctest.DocTestRunner):
    """Runs examples quietly and keeps the first few failure descriptions."""

    def __init__(self) -> None:
        super().__init__(verbose=False)
        self.messages: list[str] = []

    def report_start(self, out, test, example):
        """Intentionally silent: doctest's per-example progress output is not wanted."""

    def report_success(self, out, test, example, got):
        """Intentionally silent: only failures are recorded."""

    def report_failure(self, out, test, example, got):
        self._note(
            f"{test.name}: {example.source.strip()!r} expected {example.want.strip()!r}, "
            f"got {got.strip()!r}"
        )

    def report_unexpected_exception(self, out, test, example, exc_info):
        self._note(
            f"{test.name}: {example.source.strip()!r} raised {exc_info[0].__name__}: {exc_info[1]}"
        )

    def _note(self, message: str) -> None:
        if len(self.messages) < 3:
            self.messages.append(message)


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _ERROR_CHARS else text[: _ERROR_CHARS - 3] + "..."


def _failure(status: str, error: str) -> dict[str, object]:
    return {"status": status, "n_examples": 0, "n_failed": 0, "error": _short(error)}


def _load_candidate(code: types.CodeType, module: types.ModuleType) -> dict[str, object] | None:
    """Execute the candidate's top level; return a failure result, or ``None`` on success."""

    try:
        exec(code, module.__dict__)  # noqa: S102  # nosec B102  # nosemgrep  # pylint: disable=exec-used
    except BaseException as exc:  # noqa: BLE001  # NOSONAR
        # Deliberately broad: the candidate is untrusted, so *anything* it raises while
        # loading (SystemExit, KeyboardInterrupt, a RecursionError...) is a scoring result for
        # this worker, never a reason to die without writing one. Nothing is swallowed
        # silently: the type and message are recorded as an ``import_error``.
        return _failure("import_error", f"{type(exc).__name__}: {exc}")
    return None


def _run_examples(module: types.ModuleType) -> dict[str, object]:
    try:
        tests = doctest.DocTestFinder(recurse=True).find(module, "candidate")
    except ValueError as exc:  # e.g. a malformed ``# doctest: +NOPE`` directive
        return _failure("malformed_doctest", str(exc))
    runner = _RecordingRunner()
    for test in tests:
        runner.run(test, out=lambda _text: None, clear_globs=True)
    error = " | ".join(runner.messages) or None
    return {
        "status": "ok",
        "n_examples": runner.tries,
        "n_failed": runner.failures,
        "error": None if error is None else _short(error),
        # The examples the finder actually discovered and ran, so the parent can check that
        # the reference's examples were kept where they execute (not in dead or nested code).
        "examples": [[e.source, e.want] for test in tests for e in test.examples],
    }


def _run(source: str, sandbox: str) -> dict[str, object]:
    try:
        code = compile(source, "candidate.py", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return _failure("syntax_error", repr(exc))

    module = types.ModuleType("candidate")
    module.__file__ = os.path.join(sandbox, "candidate.py")
    sys.modules["candidate"] = module
    # ``sys.__stdout__`` / ``sys.__stderr__`` may be ``None``; restore what was really there.
    original_stdout, original_stderr = sys.stdout, sys.stderr
    quiet = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - closed with the process
    sys.stdout = sys.stderr = quiet
    try:
        _install_audit_hook(sandbox)
        return _load_candidate(code, module) or _run_examples(module)
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def main(argv: list[str]) -> int:
    source_path, result_path, cpu_seconds = argv[1], argv[2], int(argv[3])
    token = sys.stdin.read()  # sent once by the parent; not in argv, environ, or /proc cmdline
    sandbox = os.path.dirname(os.path.abspath(source_path))
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    sys.dont_write_bytecode = True
    sys.setrecursionlimit(2000)
    _apply_resource_limits(cpu_seconds)
    # Opened before the audit hook exists: afterwards nothing outside the sandbox directory
    # can be opened for writing, and this file lives outside it.
    with open(result_path, "w", encoding="utf-8") as handle:
        result = _run(source, sandbox)
        result["token"] = token
        json.dump(result, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
