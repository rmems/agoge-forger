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


def _pickle_metadata_strings(archive: zipfile.ZipFile, name: str) -> set[str] | None:
    with archive.open(name) as stream:
        payload = stream.read(_MAX_PICKLE_METADATA_BYTES + 1)
    if not payload or len(payload) > _MAX_PICKLE_METADATA_BYTES:
        return None
    try:
        pickletools.dis(payload, out=_NullWriter())
    except (ValueError, EOFError):
        return None
    return {argument for _, argument, _ in pickletools.genops(payload) if isinstance(argument, str)}


def torch_zip_metadata(path: Path, *, require_data_record: bool = False) -> set[str] | None:
    """Return static pickle strings from a structurally valid PyTorch ZIP."""
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("rb") as handle, zipfile.ZipFile(handle) as archive:
            names = archive.namelist()
            root = _archive_root(names)
            if root is None:
                return None
            data_name = f"{root}/data.pkl"
            required = {data_name, f"{root}/version", f"{root}/.data/serialization_id"}
            data_prefix = f"{root}/data/"
            if not required.issubset(names) or archive.testzip() is not None:
                return None
            if require_data_record and not any(name.startswith(data_prefix) for name in names):
                return None
            return _pickle_metadata_strings(archive, data_name)
    except (RuntimeError, zipfile.BadZipFile):
        return None
