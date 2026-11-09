"""Denylist on ``_imp`` loaders: block ``_posixsubprocess``, keep other extensions."""

import os
from pathlib import Path

import pytest

from agoge_forger.eval.score import (
    DOCTEST_V1_SCORING_VERSION,
    GenerationRecord,
    score_doctest,
    score_generation,
)


def _score(completion: str, reference: str | None = None):
    reference = completion if reference is None else reference
    record = GenerationRecord(
        task_id="task-001",
        prompt="Fix:\n",
        expected_completion=reference,
        completion=completion,
        status="ok",
    )
    return score_generation(record, scoring_version=DOCTEST_V1_SCORING_VERSION)


@pytest.mark.skipif(os.name != "posix", reason="fork_exec is POSIX only")
def test_extension_loader_original_is_not_recovered_from_a_closure(tmp_path: Path):
    # A wrapper that closed over the real ``create_builtin`` let the candidate read it
    # back from ``__closure__`` and spawn a process. The installed function has no closure.
    marker = tmp_path / "pwned"
    module = (
        'def f():\n    """\n    >>> import _imp, importlib.machinery, os\n'
        "    >>> load = _imp.create_builtin.__closure__[0].cell_contents\n"
        '    >>> spec = importlib.machinery.ModuleSpec("_posixsubprocess", '
        "importlib.machinery.BuiltinImporter)\n"
        "    >>> m = load(spec)\n"
        f'    >>> m.fork_exec([b"/bin/sh", b"-c", b"touch {marker}"], '
        "[b'/bin/sh'], True, (), None, None, -1, -1, -1, -1, -1, -1, "
        "*os.pipe(), False, False, -1, None, None, None, -1, None)\n"
        '    """\n'
    )
    score = score_doctest(
        GenerationRecord(
            task_id="task-001",
            prompt="Fix:\n",
            expected_completion=module,
            completion=module,
            status="ok",
        )
    )
    assert score.verdict != "correct"
    assert not marker.exists()


def test_installed_loader_has_no_closure_and_denies_a_blocked_name():
    module = (
        'def f():\n    """\n    >>> import _imp\n'
        "    >>> _imp.create_builtin.__closure__ is None\n"
        "    True\n"
        "    >>> _imp.create_dynamic.__closure__ is None\n"
        "    True\n"
        '    """\n'
    )
    assert _score(module).verdict == "correct"


@pytest.mark.skipif(os.name != "posix", reason="dir_fd-relative open is POSIX only")
def test_real_open_is_not_recovered_from_a_closure(tmp_path: Path):
    marker = "escaped_x"
    module = (
        'def f():\n    """\n    >>> import os\n'
        "    >>> real = os.open.__closure__[0].cell_contents\n"
        f"    >>> fd = real({str(tmp_path)!r}, os.O_RDONLY)\n"
        f"    >>> real({marker!r}, os.O_WRONLY | os.O_CREAT, dir_fd=fd)\n"
        '    """\n'
    )
    score = score_doctest(
        GenerationRecord(
            task_id="task-001",
            prompt="Fix:\n",
            expected_completion=module,
            completion=module,
            status="ok",
        )
    )
    assert score.verdict != "correct"
    assert not (tmp_path / marker).exists()


@pytest.mark.skipif(os.name != "posix", reason="fork_exec is POSIX only")
def test_str_subclass_cannot_smuggle_a_denylisted_loader_name(tmp_path: Path):
    # ``isinstance(name, str)`` is true for a subclass that overrides ``split``, so the
    # denylist used to see ``"ok"`` while the C loader read ``_posixsubprocess``.
    marker = tmp_path / "pwned_sub"
    module = (
        'def f():\n    """\n    >>> import _imp, importlib.machinery, os\n'
        "    >>> class S(str):\n"
        "    ...     def split(self, sep=None, maxsplit=-1):\n"
        '    ...         return ["ok"]\n'
        '    >>> spec = importlib.machinery.ModuleSpec(S("_posixsubprocess"), '
        "importlib.machinery.BuiltinImporter)\n"
        "    >>> m = _imp.create_builtin(spec)\n"
        f'    >>> m.fork_exec([b"/bin/sh", b"-c", b"touch {marker}"], '
        "[b'/bin/sh'], True, (), None, None, -1, -1, -1, -1, -1, -1, "
        "*os.pipe(), False, False, -1, None, None, None, -1, None)\n"
        '    """\n'
    )
    score = score_doctest(
        GenerationRecord(
            task_id="task-001",
            prompt="Fix:\n",
            expected_completion=module,
            completion=module,
            status="ok",
        )
    )
    assert score.verdict != "correct"
    assert not marker.exists()


def test_third_party_extension_module_still_imports():
    # ``markupsafe._speedups`` is a shared library. Refusing every ``create_dynamic``
    # call turns this into a ``PermissionError`` that the package does not catch.
    module = (
        'def f():\n    """\n    >>> import markupsafe\n'
        "    >>> markupsafe.escape('<tag>')\n"
        "    Markup('&lt;tag&gt;')\n"
        '    """\n'
    )
    score = _score(module)
    assert score.verdict == "correct", score.error
