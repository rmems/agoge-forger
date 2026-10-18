"""Subprocess entry point for the ``doctest-v1`` scorer.

Run as ``python -P -s _doctest_worker.py SOURCE RESULT CPU_SECONDS``. It is standalone
(standard library only) and is never imported by the parent process, because it
executes model-generated code. The parent supplies the timeout, a scrubbed
environment, and a throwaway working directory; this process adds best-effort
containment on top of that:

* POSIX resource limits (CPU time, address space, file size, core dumps);
* an audit hook that refuses network access, process creation, native library
  loading, and filesystem mutation outside the working directory.

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
# Events whose leading arguments are filesystem paths that must stay in the sandbox.
_PATH_EVENTS = {
    "os.chmod": 1,
    "os.link": 2,
    "os.mkdir": 1,
    "os.remove": 1,
    "os.rename": 2,
    "os.rmdir": 1,
    "os.symlink": 2,
    "os.truncate": 1,
    "shutil.copyfile": 2,
    "shutil.move": 2,
    "shutil.rmtree": 1,
}
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


def _install_audit_hook(sandbox: str) -> None:
    root = os.path.realpath(sandbox)
    allowed_devnull = {os.path.realpath(os.devnull), os.devnull}

    def inside(path: object) -> bool:
        if isinstance(path, int):
            return True  # an already-open descriptor was vetted when it was opened
        if not isinstance(path, str | bytes | os.PathLike):
            return False
        try:
            real = os.path.realpath(os.fsdecode(path))
        except ValueError:
            return False
        return real in allowed_devnull or real == root or real.startswith(root + os.sep)

    def hook(event: str, args: tuple[object, ...]) -> None:
        if event in _BLOCKED_EVENTS:
            raise PermissionError(f"doctest-v1 sandbox blocked {event}")
        if event == "open":
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            if flags & _WRITE_FLAGS and not inside(args[0]):
                raise PermissionError("doctest-v1 sandbox blocked a write outside its directory")
        elif event in _PATH_EVENTS:
            if not all(inside(path) for path in args[: _PATH_EVENTS[event]]):
                raise PermissionError(f"doctest-v1 sandbox blocked {event} outside its directory")

    sys.addaudithook(hook)


class _RecordingRunner(doctest.DocTestRunner):
    """Runs examples quietly and keeps the first few failure descriptions."""

    def __init__(self) -> None:
        super().__init__(verbose=False)
        self.messages: list[str] = []

    def report_start(self, out, test, example):
        pass

    def report_success(self, out, test, example, got):
        pass

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


def _run(source: str, sandbox: str) -> dict[str, object]:
    try:
        code = compile(source, "candidate.py", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return {
            "status": "syntax_error",
            "n_examples": 0,
            "n_failed": 0,
            "error": _short(repr(exc)),
        }

    module = types.ModuleType("candidate")
    module.__file__ = os.path.join(sandbox, "candidate.py")
    sys.modules["candidate"] = module
    quiet = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - closed with the process
    sys.stdout = sys.stderr = quiet
    try:
        _install_audit_hook(sandbox)
        try:
            exec(code, module.__dict__)  # noqa: S102  # nosec B102 - sandboxed worker
        except BaseException as exc:  # noqa: BLE001 - SystemExit from the module must not escape
            return {
                "status": "import_error",
                "n_examples": 0,
                "n_failed": 0,
                "error": _short(f"{type(exc).__name__}: {exc}"),
            }
        runner = _RecordingRunner()
        for test in doctest.DocTestFinder(recurse=True).find(module, "candidate"):
            runner.run(test, out=lambda _text: None, clear_globs=True)
        error = " | ".join(runner.messages) or None
        return {
            "status": "ok",
            "n_examples": runner.tries,
            "n_failed": runner.failures,
            "error": None if error is None else _short(error),
        }
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__


def main(argv: list[str]) -> int:
    source_path, result_path, cpu_seconds = argv[1], argv[2], int(argv[3])
    sandbox = os.path.dirname(os.path.abspath(source_path))
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    sys.dont_write_bytecode = True
    sys.setrecursionlimit(2000)
    _apply_resource_limits(cpu_seconds)
    result = _run(source, sandbox)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
