"""Non-executing structural checks for current PyTorch ZIP serialization."""

from __future__ import annotations

import io
import pickletools  # nosec B403 - disassembles bytes; never executes or loads pickle
import zipfile
from pathlib import Path

_MAX_PICKLE_METADATA_BYTES = 64 * 1024 * 1024


class _NullWriter(io.StringIO):
    def write(self, text: str) -> int:
        """Discard pickle disassembly output."""
        return len(text)


def _archive_root(names: list[str]) -> str | None:
    roots = {name.partition("/")[0] for name in names}
    return roots.pop() if len(roots) == 1 else None


def _pickle_metadata_usable(archive: zipfile.ZipFile, name: str) -> bool:
    with archive.open(name) as stream:
        payload = stream.read(_MAX_PICKLE_METADATA_BYTES + 1)
    if not payload or len(payload) > _MAX_PICKLE_METADATA_BYTES:
        return False
    try:
        pickletools.dis(payload, out=_NullWriter())
    except (ValueError, EOFError):
        return False
    return True


def torch_zip_usable(path: Path) -> bool:
    """Validate a PyTorch ZIP and its pickle syntax without deserializing it."""
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with path.open("rb") as handle, zipfile.ZipFile(handle) as archive:
            names = archive.namelist()
            root = _archive_root(names)
            if root is None:
                return False
            data_name = f"{root}/data.pkl"
            required = {data_name, f"{root}/version", f"{root}/.data/serialization_id"}
            return bool(
                required.issubset(names)
                and archive.testzip() is None
                and _pickle_metadata_usable(archive, data_name)
            )
    except (RuntimeError, zipfile.BadZipFile):
        return False
