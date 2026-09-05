"""Shared staging policy for immutable validation inputs."""

import errno
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def validation_staging_dir(preferred: Path) -> Path:
    configured = os.environ.get("AGOGE_VALIDATION_STAGING_DIR")
    if configured is None or not configured.strip():
        return preferred
    try:
        path = Path(configured).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("validation staging path must be an existing writable directory") from exc
    if not path.is_dir():
        raise ValueError(f"validation staging path must be a directory: {path}")
    return path


def unwritable_staging_error() -> ValueError:
    return ValueError(
        "validation staging is not writable; set AGOGE_VALIDATION_STAGING_DIR "
        "to an existing writable directory"
    )


@contextmanager
def validation_directory(prefix: str, preferred: Path) -> Iterator[Path]:
    try:
        staging = tempfile.TemporaryDirectory(
            prefix=prefix,
            dir=validation_staging_dir(preferred),
        )
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
            raise
        raise unwritable_staging_error() from exc
    with staging as path:
        yield Path(path)
