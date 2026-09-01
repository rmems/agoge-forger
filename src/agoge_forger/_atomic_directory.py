"""Atomic, no-clobber directory publication on Linux."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename *source* while refusing every existing destination."""

    encoded_source = _encoded_path(source)
    encoded_destination = _encoded_path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported on this platform",
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        encoded_source,
        _AT_FDCWD,
        encoded_destination,
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(
            error,
            f"refusing silent regeneration because output path already exists: {destination}",
            destination,
        )
    raise OSError(error, os.strerror(error), destination)


def _encoded_path(path: Path) -> bytes:
    encoded = os.fsencode(path)
    if b"\0" in encoded:
        raise ValueError(f"filesystem path contains an embedded NUL: {path!r}")
    return encoded
