"""Atomic, no-clobber filesystem publication on Linux."""

from __future__ import annotations

import ctypes
import errno
import os
import tempfile
from pathlib import Path

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def require_rename_noreplace_support(staging_parent: Path) -> None:
    """Probe no-replace support on the destination filesystem."""

    if not staging_parent.is_dir():
        raise ValueError(f"atomic publication staging parent must be a directory: {staging_parent}")
    with tempfile.TemporaryDirectory(prefix=".agoge-rename-probe-", dir=staging_parent) as root:
        probe_root = Path(root)
        _probe_rename_noreplace(probe_root, staging_parent)


def _probe_rename_noreplace(probe_root: Path, staging_parent: Path) -> None:
    source = probe_root / "source"
    occupied = probe_root / "occupied"
    source.mkdir()
    occupied.mkdir()
    try:
        rename_noreplace(source, occupied)
    except FileExistsError:
        return
    except OSError as exc:
        _raise_probe_error(exc, staging_parent)
    _raise_unsupported_filesystem(staging_parent)


def _raise_probe_error(error: OSError, staging_parent: Path) -> None:
    unsupported = {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}
    if error.errno not in unsupported:
        raise error
    _raise_unsupported_filesystem(staging_parent, cause=error)


def _raise_unsupported_filesystem(staging_parent: Path, *, cause: OSError | None = None) -> None:
    error = OSError(
        errno.ENOTSUP,
        "atomic no-replace publication is unsupported on the destination filesystem",
        staging_parent,
    )
    if cause is None:
        raise error
    raise error from cause


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename *source* while refusing every existing destination."""

    encoded_source = _encoded_path(source)
    encoded_destination = _encoded_path(destination)
    renameat2 = _renameat2()
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


def _renameat2():
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        return libc.renameat2
    except AttributeError as exc:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace publication is unsupported on this platform",
        ) from exc


def _encoded_path(path: Path) -> bytes:
    encoded = os.fsencode(path)
    if b"\0" in encoded:
        raise ValueError(f"filesystem path contains an embedded NUL: {path!r}")
    return encoded
