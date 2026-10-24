"""Subprocess entry point for the ``doctest-v1`` scorer.

Run as ``python -P -s _doctest_worker.py SOURCE RESULT CPU_SECONDS`` with a one-time
token on stdin. It is standalone (standard library only) and is never imported by the
parent process, because it executes model-generated code. The parent supplies the
timeout, a scrubbed environment, and a throwaway working directory; this process adds
best-effort containment on top of that:

* POSIX resource limits (CPU time, address space, file size, core dumps);
* an audit hook that refuses socket creation and use, process creation and signalling,
  ``ctypes`` library loading, filesystem mutation outside the working directory, and the
  import of the modules that reach those facilities without an audit event of their own
  (``_posixsubprocess``, ``socket``, ``subprocess``, ``ctypes``, ``sqlite3``,
  ``multiprocessing``, ...; see ``_BLOCKED_IMPORTS``). Any dotted component of the
  import name counts, so ``pkg._posixsubprocess`` is refused as well as the top-level
  module. ``_imp.create_builtin`` and ``_imp.create_dynamic`` apply that same denylist
  and then call the original loader, so a denylisted spec cannot revive ``fork_exec``
  while every other extension (stdlib ``lib-dynload`` or a third-party speedup) still
  loads. A blocked import is refused as an
  ``ImportError`` so a stdlib optional-import probe (``try: import _winapi``, reached by
  ``pathlib``) falls back instead of crashing a valid candidate. Copies of those modules
  that the standard library imported before the candidate ran are dropped from
  ``sys.modules`` so the denylist bites a cached import too; ``os.mkfifo`` / ``os.mknod``
  (which raise no audit event) are removed from ``os`` and ``posix``; and ``os.open`` is
  wrapped to refuse a ``dir_fd`` (its ``open`` audit event omits ``dir_fd``, so a write
  relative to a descriptor opened outside the sandbox would otherwise escape).

The result is reported through ``RESULT``, a file in a *parent-owned directory outside
the working directory*. It is opened before the candidate runs, and the bytes are
written with bindings (``os.write``, the file descriptor, a JSON encoder built from the
C accelerator) captured at the same time, so reassigning ``json.dump`` or anything else
in ``json``'s namespace cannot change what is written. The result is stamped with the
token the parent sent on stdin; the audit hook refuses writes outside the working
directory, so candidate code cannot create or replace the file by path, and a result
without the right token is discarded by the parent. Finally the namespaces the scoring
path relies on (``doctest``, ``json``, ``os``, ``builtins``, this module, and the
``doctest`` classes) are snapshotted before the candidate runs and compared afterwards;
a candidate that rebinds any of them is reported as ``tampered`` instead of scored.

What this does **not** defend: code that digs the open descriptor, the token, or the
harness bindings out of this process's frames (``sys._getframe``, ``gc.get_referrers``);
in-process patching outside the snapshotted set (``linecache``, ``pdb``, ``sys``,
``importlib`` internals, ``sys.modules`` itself); and compiled extension modules other
than those in the import denylist, because CPython raises no dedicated audit event for
loading one.

This is a guard against accidents (a generated module that opens a socket, spawns a
process, or writes outside its directory), **not** a security boundary against code
written to escape it. Run the scorer inside a container or VM if the generations
are adversarial.
"""

from __future__ import annotations

import _imp
import builtins
import doctest
import json.encoder
import os
import sys
import types
from collections.abc import Callable
from typing import Any

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
        "os.kill",
        "os.killpg",
        "os.posix_spawn",
        "os.spawn",
        "os.startfile",
        "os.system",
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.getnameinfo",
        "socket.sendmsg",
        "socket.sendto",
        "subprocess.Popen",
    }
)
# Top-level modules the candidate may not import: each reaches process creation, the
# network, native code, or a database file through calls that raise no audit event.
_BLOCKED_IMPORTS = frozenset(
    {
        "_ctypes",
        "_multiprocessing",
        "_posixsubprocess",
        "_socket",
        "_sqlite3",
        "_winapi",
        "ctypes",
        "multiprocessing",
        "socket",
        "sqlite3",
        "subprocess",
    }
)
# Blocked modules (and their submodules) are dropped from ``sys.modules`` so a cached copy
# cannot be handed back, then set to ``None`` so a fresh import is refused. ``asyncio`` /
# ``ssl`` are deliberately *not* purged: legitimate modules import them (``unittest.mock``
# pulls in ``asyncio``), and the only unaudited facility they expose indirectly — the
# ``fork_exec`` process primitive — is revoked outright by ``_revoke_fork_exec``.
_PURGED_IMPORTS = _BLOCKED_IMPORTS
# ``os`` functions that mutate the filesystem without raising an audit event.
_UNAUDITED_OS_FUNCTIONS = ("mkfifo", "mknod")
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
    "sqlite3.connect": (1, ()),
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


def _path_text(path: object) -> str | bytes | os.PathLike[str] | None:
    """A path whose string methods are the real ``str`` / ``bytes`` ones.

    A ``str`` subclass can override ``split`` or ``__fspath__`` and still pass
    ``isinstance(..., str)``. Only an exact ``str`` or ``bytes``, or some other
    ``os.PathLike``, is accepted.
    """

    if type(path) is str or type(path) is bytes:
        return path
    if isinstance(path, os.PathLike) and not isinstance(path, str | bytes):
        return path
    return None


def _real_path(path: object) -> str | None:
    plain = _path_text(path)
    if plain is None:
        return None
    try:
        return os.path.realpath(os.fsdecode(plain))
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


def _blocked_import_name(name: str) -> bool:
    """True when any dotted component is denylisted (``pkg._posixsubprocess`` included)."""

    return any(part in _BLOCKED_IMPORTS for part in name.split("."))


def _import_denial(args: tuple[object, ...]) -> str | None:
    name = args[0] if args else None
    if type(name) is not str:  # a str subclass can override split; that fails closed
        return "doctest-v1 sandbox blocked an import it could not identify"
    if _blocked_import_name(name):
        return f"doctest-v1 sandbox blocked import of {name}"
    return None


# Originals of ``_imp.create_builtin`` / ``create_dynamic``, filled in before those
# attributes are replaced. Kept on the module rather than in a ``__closure__`` cell.
# Reading this global from the candidate is the same in-process access the sandbox
# does not claim to stop; the functions installed on ``_imp`` close over nothing.
_SAVED_IMP_LOADERS: dict[str, Callable[..., Any]] = {}


def _loader_name(spec: object) -> str | None:
    name = getattr(spec, "name", None)
    # ``isinstance`` would accept a ``str`` subclass whose ``split`` hides the real name.
    if type(name) is not str or name == "":
        return None
    return name


def _call_saved_loader(
    kind: str, spec: object, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    """Run a saved ``_imp`` loader unless the spec name is denylisted.

    ``ModuleNotFoundError`` (an ``ImportError``) so an optional speedup import falls
    back, matching the import hook. A name we cannot read is refused.
    """

    name = _loader_name(spec)
    if name is None or _blocked_import_name(name):
        raise ModuleNotFoundError(
            "doctest-v1 sandbox blocked loading a denylisted module without an import",
            name=name,
        )
    return _SAVED_IMP_LOADERS[kind](spec, *args, **kwargs)


def _create_builtin(spec: object, *args: object, **kwargs: object) -> object:
    return _call_saved_loader("create_builtin", spec, args, kwargs)


def _create_dynamic(spec: object, *args: object, **kwargs: object) -> object:
    return _call_saved_loader("create_dynamic", spec, args, kwargs)


def _fence_extension_loaders() -> None:
    """Point ``_imp``'s loaders at the denylist. Importlib looks them up on each call."""

    for name, replacement in (
        ("create_builtin", _create_builtin),
        ("create_dynamic", _create_dynamic),
    ):
        load = getattr(_imp, name, None)
        if not callable(load):  # fail closed: never run a candidate with an unguarded loader
            raise TypeError(f"doctest-v1 worker cannot guard non-callable _imp.{name}")
        _SAVED_IMP_LOADERS[name] = load
        setattr(_imp, name, replacement)


def _denial(event: str, args: tuple[object, ...], policy: _PathPolicy) -> str | None:
    """Return why the sandbox refuses ``event``, or ``None`` when it is allowed."""

    if event in _BLOCKED_EVENTS:
        return f"doctest-v1 sandbox blocked {event}"
    if event == "open":
        return _open_denial(args, policy)
    if event == "import":
        return _import_denial(args)
    if event in _PATH_EVENTS:
        return _path_event_denial(event, args, policy)
    return None


def _install_audit_hook(sandbox: str) -> None:
    policy = _PathPolicy(sandbox)
    denial_of = _denial  # bound here so rebinding this module's global cannot disable it

    def hook(event: str, args: tuple[object, ...]) -> None:
        denial = denial_of(event, args, policy)
        if denial is None:
            return
        if event == "import":
            # Raise an ``ImportError`` (not ``PermissionError``) so a stdlib optional-import
            # probe such as ``try: import _winapi`` (reached by ``pathlib`` -> ``ntpath``)
            # falls back cleanly; a candidate that actually needs the module still fails to
            # import it, and the dangerous calls stay blocked by the other branches.
            name = args[0] if args and isinstance(args[0], str) else None
            raise ModuleNotFoundError(denial, name=name)
        raise PermissionError(denial)

    sys.addaudithook(hook)


def _strip_module_attrs(module: types.ModuleType | None, names: tuple[str, ...]) -> None:
    """Delete each of ``names`` from ``module`` when present; a no-op if the module is absent.

    Used for calls that mutate the filesystem without raising an audit event
    (``os.mkfifo``, ``os.mknod``), so making them unavailable is the only defence.
    """

    if module is None:
        return
    for name in names:
        if hasattr(module, name):
            delattr(module, name)


def _inode_identity(path: str) -> tuple[int, int] | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return info.st_dev, info.st_ino


def _dir_fd_write_allowed(dir_fd: object, flags: object, path: object, root: str) -> bool:
    """Whether an ``os.open`` through ``dir_fd`` may proceed.

    Reads are always fine. A write is allowed only when ``dir_fd`` is the sandbox root
    itself (same device and inode) and the relative ``path`` resolves back inside it, so a
    descriptor opened on any directory outside the sandbox is refused.
    """

    if not (isinstance(flags, int) and flags & _WRITE_FLAGS):
        return True
    root_id = _inode_identity(root)
    plain = _path_text(path)
    if root_id is None or plain is None:
        return False
    try:
        info = os.fstat(dir_fd)  # type: ignore[arg-type]
    except (OSError, TypeError, ValueError):
        return False
    if (info.st_dev, info.st_ino) != root_id:
        return False
    resolved = os.path.realpath(os.path.join(root, os.fsdecode(plain)))
    return resolved == root or resolved.startswith(root + os.sep)


# Real ``os.open`` / ``posix.open`` and the sandbox root they are checked against.
# Same rule as ``_SAVED_IMP_LOADERS``: the installed functions close over nothing.
_SANDBOX_ROOT = ""
_SAVED_OPENS: dict[str, Callable[..., Any]] = {}


def _open_with_dir_fd_policy(
    module_name: str, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    """``os.open`` / ``posix.open`` with the sandbox-root ``dir_fd`` check."""

    if kwargs.get("dir_fd") is not None:
        path = args[0] if args else kwargs.get("path")
        flags = args[1] if len(args) > 1 else kwargs.get("flags", 0)
        if not _dir_fd_write_allowed(kwargs["dir_fd"], flags, path, _SANDBOX_ROOT):
            raise PermissionError(
                "doctest-v1 sandbox blocked a dir_fd-relative open outside its directory"
            )
    return _SAVED_OPENS[module_name](*args, **kwargs)


def _guarded_os_open(*args: object, **kwargs: object) -> object:
    return _open_with_dir_fd_policy("os", args, kwargs)


def _guarded_posix_open(*args: object, **kwargs: object) -> object:
    return _open_with_dir_fd_policy("posix", args, kwargs)


_DIR_FD_GUARDS = {"os": _guarded_os_open, "posix": _guarded_posix_open}


def _fence_dir_fd(module: types.ModuleType | None, name: str, sandbox: str) -> None:
    """Wrap ``module.name`` so a ``dir_fd`` write that leaves the sandbox is refused.

    ``os.open``'s ``open`` audit event carries only the path and flags, not ``dir_fd``, so
    the path-based hook cannot tell that ``os.open("x", O_WRONLY, dir_fd=outside_fd)`` writes
    outside the sandbox. The ``*at`` events (``os.mkdir`` and friends) do carry ``dir_fd``
    and are refused in ``_PATH_EVENTS``; this closes the one write path that does not, while
    still allowing a ``dir_fd`` that points at the sandbox root (``os.open(".")``). The
    wrapper is a module-level function, not a closure over the real ``open``.
    """

    global _SANDBOX_ROOT
    if module is None:
        return
    opener = getattr(module, name, None)
    if not callable(opener):  # fail closed: never run a candidate with an unguarded open
        raise TypeError(f"doctest-v1 worker cannot guard non-callable {module.__name__}.{name}")
    guard = _DIR_FD_GUARDS.get(module.__name__)
    if guard is None:
        raise TypeError(f"doctest-v1 worker has no dir_fd guard for {module.__name__}")
    _SANDBOX_ROOT = os.path.realpath(sandbox)
    _SAVED_OPENS[module.__name__] = opener
    setattr(module, name, guard)


def _revoke_fork_exec() -> None:
    """Rebind every live reference to ``_posixsubprocess.fork_exec`` to a refusing stub.

    ``fork_exec`` spawns a process but raises no audit event, so blocking the import of
    ``_posixsubprocess`` is not enough: a cached ``subprocess`` (reachable through, e.g.,
    ``asyncio.unix_events``) holds the callable directly as ``subprocess._fork_exec``.
    Called before the modules are purged, while the holders are still reachable.
    """

    original = getattr(sys.modules.get("_posixsubprocess"), "fork_exec", None)
    if original is None:
        return

    def refused(*_args: object, **_kwargs: object) -> object:
        raise PermissionError("doctest-v1 sandbox blocked process creation via fork_exec")

    for module in list(sys.modules.values()):
        _rebind_attr(module, original, refused)


def _rebind_attr(module: types.ModuleType | None, original: object, replacement: object) -> None:
    """Replace any attribute of ``module`` that *is* ``original`` with ``replacement``."""

    if module is None:
        return
    try:
        names = [name for name, value in vars(module).items() if value is original]
    except TypeError:  # a module without an inspectable ``__dict__``
        return
    for name in names:
        try:
            setattr(module, name, replacement)
        except (AttributeError, TypeError):  # a read-only attribute: leave it
            continue


def _remove_unaudited_facilities(sandbox: str) -> None:
    """Take the modules and calls the audit hook cannot see out of the candidate's reach."""

    _revoke_fork_exec()
    _fence_extension_loaders()
    for module in (os, sys.modules.get("posix")):
        _strip_module_attrs(module, _UNAUDITED_OS_FUNCTIONS)
        _fence_dir_fd(module, "open", sandbox)
    for name in [n for n in sys.modules if n.partition(".")[0] in _PURGED_IMPORTS]:
        del sys.modules[name]
    for name in _BLOCKED_IMPORTS:
        # ``None`` makes every import path (including ``importlib.import_module``, which
        # raises no audit event) fail with ``ModuleNotFoundError``.
        sys.modules[name] = None  # type: ignore[assignment]


class _RecordingRunner(doctest.DocTestRunner):
    """Runs examples quietly; records what ran and the first few failure descriptions.

    The counts and example pairs come from this class's own report callbacks rather than
    from ``DocTestRunner``'s bookkeeping, which a candidate could reach through the class.
    """

    def __init__(self) -> None:
        super().__init__(verbose=False)
        self.messages: list[str] = []
        self.ran: list[list[str]] = []
        self.n_failed = 0

    def report_start(self, out, test, example):
        """Silent except for recording the example ``doctest`` is about to run."""
        self.ran.append([example.source, example.want])

    # ``report_success`` is not overridden: the base method only writes when ``verbose`` is
    # set, and this runner is quiet, so a passing example needs no handling here.

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
        self.n_failed += 1
        if len(self.messages) < 3:
            self.messages.append(message)


_HARNESS_CLASSES: tuple[type, ...] = (
    doctest.DocTest,
    doctest.DocTestFinder,
    doctest.DocTestParser,
    doctest.DocTestRunner,
    doctest.Example,
    doctest.OutputChecker,
    _RecordingRunner,
)
_HARNESS_MODULES: tuple[types.ModuleType, ...] = (
    builtins,
    doctest,
    json,
    json.encoder,
    os,
    os.path,
)
# ``sys.displayhook`` legitimately rebinds ``builtins._`` while examples run.
_VOLATILE_NAMES = frozenset({"_"})


def _harness_namespaces() -> dict[str, dict[str, object]]:
    """The live module namespaces the scoring path relies on, including this module's own."""

    namespaces = {module.__name__: vars(module) for module in _HARNESS_MODULES}
    namespaces["worker"] = globals()
    return namespaces


class _HarnessSnapshot:
    """Identity snapshot of the namespaces the scoring path relies on."""

    def __init__(self) -> None:
        self._modules = {name: dict(live) for name, live in _harness_namespaces().items()}
        self._classes = {cls: dict(vars(cls)) for cls in _HARNESS_CLASSES}

    def changes(self) -> list[str]:
        """Names rebound or removed since the snapshot (modules may gain names freely)."""

        live = _harness_namespaces()
        changed: list[str] = []
        for name, saved in self._modules.items():
            changed.extend(_rebound(name, saved, live[name], additions=False))
        for cls, saved in self._classes.items():
            changed.extend(_rebound(cls.__qualname__, saved, dict(vars(cls)), additions=True))
        return changed


def _rebound(
    owner: str, saved: dict[str, object], current: dict[str, object], *, additions: bool
) -> list[str]:
    names = set(saved) | (set(current) if additions else set())
    return sorted(
        f"{owner}.{name}"
        for name in names - _VOLATILE_NAMES
        if name not in current or name not in saved or current[name] is not saved[name]
    )


def _bind_json_encoder() -> Callable[[object], str]:
    """Return a JSON serializer whose call path no longer consults ``json``'s globals.

    ``json.dumps`` looks up ``json.encoder.c_make_encoder`` and friends at call time, so a
    candidate could redirect it; the C encoder instance built here does not.
    """

    # Imported straight from the C accelerator rather than read off ``json.encoder``, where
    # ``c_make_encoder`` is ``None`` on a pure-Python build and is a name a candidate could
    # rebind. Without it the worker fails closed: this runs before the result file exists,
    # so the parent sees a worker that never started (``invalid``), not a scored result.
    try:
        from _json import encode_basestring_ascii, make_encoder
    except ImportError as exc:  # pragma: no cover - CPython always ships _json
        raise RuntimeError("doctest-v1 worker needs the C json accelerator (_json)") from exc

    def unsupported(value: object) -> object:
        raise TypeError(f"result contains an unserialisable {type(value).__name__}")

    chunks = make_encoder(
        None, unsupported, encode_basestring_ascii, None, ":", ",", False, False, False
    )
    return lambda value: "".join(chunks(value, 0))


class _ResultChannel:
    """The result file, opened and bound before the candidate runs."""

    def __init__(self, path: str) -> None:
        # Bound first: if the encoder is unavailable the worker stops before the result file
        # exists, which the parent reports as a worker that never started.
        self._encode = _bind_json_encoder()
        self._write = os.write
        self._close = os.close
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)

    def publish(self, result: dict[str, object], *, _len: Callable[[bytes], int] = len) -> None:
        """Write ``result`` through the bindings taken at construction only."""

        data = self._encode(result).encode("utf-8")
        written = 0
        while written < _len(data):
            written += self._write(self._fd, data[written:])
        self._close(self._fd)


class _Harness:
    """Everything the worker needs after the candidate has run, bound before it runs."""

    def __init__(self, result_path: str) -> None:
        self.channel = _ResultChannel(result_path)
        self.find = doctest.DocTestFinder.find
        self.run = doctest.DocTestRunner.run
        self.snapshot = _HarnessSnapshot()


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


def _run_examples(module: types.ModuleType, harness: _Harness) -> dict[str, object]:
    try:
        tests = harness.find(doctest.DocTestFinder(recurse=True), module, "candidate")
    except ValueError as exc:  # e.g. a malformed ``# doctest: +NOPE`` directive
        return _failure("malformed_doctest", str(exc))
    runner = _RecordingRunner()
    for test in tests:
        harness.run(runner, test, out=lambda _text: None, clear_globs=True)
    error = " | ".join(runner.messages) or None
    return {
        "status": "ok",
        "n_examples": len(runner.ran),
        "n_failed": runner.n_failed,
        "error": None if error is None else _short(error),
        # The examples the runner actually ran, so the parent can check that the
        # reference's examples were kept where they execute (not in dead or nested code).
        "examples": runner.ran,
    }


def _run(source: str, sandbox: str, harness: _Harness) -> dict[str, object]:
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
        result = _load_candidate(code, module) or _run_examples(module, harness)
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
    return _check_harness(result, harness)


def _check_harness(result: dict[str, object], harness: _Harness) -> dict[str, object]:
    """Replace ``result`` with a ``tampered`` failure if the candidate rebound the harness."""

    changes = harness.snapshot.changes()
    if not changes:
        return result
    return _failure("tampered", "candidate rebound " + ", ".join(changes))


def main(argv: list[str]) -> int:
    source_path, result_path, cpu_seconds = argv[1], argv[2], int(argv[3])
    token = sys.stdin.read()  # sent once by the parent; not in argv, environ, or /proc cmdline
    sandbox = os.path.dirname(os.path.abspath(source_path))
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    sys.dont_write_bytecode = True
    sys.setrecursionlimit(2000)
    _apply_resource_limits(cpu_seconds)
    _remove_unaudited_facilities(sandbox)
    # Opened and bound before the audit hook exists: afterwards nothing outside the sandbox
    # directory can be opened for writing, and this file lives outside it.
    harness = _Harness(result_path)
    result = _run(source, sandbox, harness)
    result["token"] = token
    harness.channel.publish(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
