"""Restricted, memory-mapped validation for current PyTorch ZIP state."""

from __future__ import annotations

import struct
import zipfile
import zlib
from contextlib import nullcontext
from pathlib import Path
from pickle import UnpicklingError  # nosec B403 - exception type only; no pickle loading
from typing import Any, BinaryIO

import torch
from transformers.trainer_pt_utils import safe_globals

_MAX_PICKLE_METADATA_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 65_536
_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024 * 1024
_MAX_EOCD_TAIL_BYTES = 22 + 65_535
_EOCD = struct.Struct("<4s4H2IH")
_ZIP64_EOCD = struct.Struct("<4sQ2H2I4Q")
_ZIP64_LOCATOR = struct.Struct("<4sIQI")


def _directory_limits_usable(entries: int, size: int) -> bool:
    return 0 < entries <= _MAX_ARCHIVE_MEMBERS and size <= _MAX_CENTRAL_DIRECTORY_BYTES


def _classic_eocd(
    tail: bytes,
) -> tuple[int, tuple[bytes, int, int, int, int, int, int, int]] | None:
    position = tail.rfind(b"PK\x05\x06")
    while position >= 0:
        record = tail[position : position + _EOCD.size]
        if len(record) == _EOCD.size:
            fields = _EOCD.unpack(record)
            if position + _EOCD.size + fields[-1] == len(tail):
                return position, fields
        position = tail.rfind(b"PK\x05\x06", 0, position)
    return None


def _standard_directory_usable(
    fields: tuple[bytes, int, int, int, int, int, int, int],
    eocd_offset: int,
) -> bool:
    _, disk, directory_disk, disk_entries, entries, size, offset, _ = fields
    return bool(
        disk == directory_disk == 0
        and disk_entries == entries
        and _directory_limits_usable(entries, size)
        and offset + size <= eocd_offset
    )


def _zip64_structure_usable(
    record_fields: tuple[int | bytes, ...],
    locator_fields: tuple[int | bytes, ...],
    record_offset: int,
) -> bool:
    signature, record_size, *_ = record_fields
    locator_signature, locator_disk, declared_offset, disks = locator_fields
    return bool(
        (signature, record_size, locator_signature) == (b"PK\x06\x06", 44, b"PK\x06\x07")
        and (locator_disk, declared_offset, disks) == (0, record_offset, 1)
    )


def _zip64_values_usable(record_fields: tuple[int | bytes, ...], record_offset: int) -> bool:
    _, _, _, _, disk, directory_disk, disk_entries, entries, size, offset = record_fields
    return bool(
        disk == directory_disk == 0
        and disk_entries == entries
        and _directory_limits_usable(int(entries), int(size))
        and int(offset) + int(size) <= record_offset
    )


def _zip64_directory_usable(handle: BinaryIO, eocd_offset: int) -> bool:
    locator_offset = eocd_offset - _ZIP64_LOCATOR.size
    record_offset = locator_offset - _ZIP64_EOCD.size
    if record_offset < 0:
        return False
    handle.seek(record_offset)
    record = handle.read(_ZIP64_EOCD.size)
    locator = handle.read(_ZIP64_LOCATOR.size)
    if len(record) != _ZIP64_EOCD.size or len(locator) != _ZIP64_LOCATOR.size:
        return False
    record_fields = _ZIP64_EOCD.unpack(record)
    locator_fields = _ZIP64_LOCATOR.unpack(locator)
    return _zip64_structure_usable(
        record_fields, locator_fields, record_offset
    ) and _zip64_values_usable(
        record_fields,
        record_offset,
    )


def _declared_directory_usable(handle: BinaryIO) -> bool:
    handle.seek(0, 2)
    file_size = handle.tell()
    tail_size = min(file_size, _MAX_EOCD_TAIL_BYTES)
    handle.seek(file_size - tail_size)
    tail = handle.read(tail_size)
    result = _classic_eocd(tail)
    if result is None:
        return False
    position, fields = result
    eocd_offset = file_size - tail_size + position
    _, _, _, disk_entries, entries, size, offset, _ = fields
    zip64 = disk_entries == entries == 0xFFFF or size == 0xFFFFFFFF or offset == 0xFFFFFFFF
    if zip64:
        return _zip64_directory_usable(handle, eocd_offset)
    return _standard_directory_usable(fields, eocd_offset)


def _archive_members(
    archive: zipfile.ZipFile,
) -> tuple[str, dict[str, zipfile.ZipInfo]] | None:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    roots = {name.partition("/")[0] for name in names if "/" in name}
    members = dict(zip(names, infos, strict=True))
    if not infos or len(infos) > _MAX_ARCHIVE_MEMBERS:
        return None
    if len(members) != len(infos) or len(roots) != 1:
        return None
    root = next(iter(roots))
    return (root, members) if root and all(name.startswith(f"{root}/") for name in names) else None


def _storage_record(info: zipfile.ZipInfo, root: str) -> bool:
    prefix = f"{root}/data/"
    suffix = info.filename.removeprefix(prefix)
    numbered_name = info.filename.startswith(prefix) and suffix.isascii() and suffix.isdigit()
    stored_file = not info.is_dir() and info.compress_type == zipfile.ZIP_STORED
    return numbered_name and stored_file


def _metadata_crc_usable(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> bool:
    if any(info.is_dir() or info.file_size < 0 for info in infos):
        return False
    if sum(info.file_size for info in infos) > _MAX_PICKLE_METADATA_BYTES:
        return False
    return all(len(archive.read(info)) == info.file_size for info in infos)


def _archive_preflight(path: Path, *, require_data_record: bool) -> bool:
    with path.open("rb") as handle:
        if not _declared_directory_usable(handle):
            return False
        handle.seek(0)
        with zipfile.ZipFile(handle) as archive:
            archive_parts = _archive_members(archive)
            if archive_parts is None:
                return False
            root, members = archive_parts
            required = {
                f"{root}/data.pkl",
                f"{root}/version",
                f"{root}/.data/serialization_id",
            }
            storage = [info for info in members.values() if _storage_record(info, root)]
            data_prefix = f"{root}/data/"
            storage_names_usable = all(
                not info.filename.startswith(data_prefix) or info in storage
                for info in members.values()
            )
            metadata = [info for info in members.values() if info not in storage]
            return bool(
                required.issubset(members)
                and storage_names_usable
                and (not require_data_record or storage)
                and _metadata_crc_usable(archive, metadata)
            )


def torch_mapping(
    path: Path,
    *,
    allow_numpy: bool = False,
    require_data_record: bool = False,
) -> dict[str, Any] | None:
    """Restricted-deserialize bounded metadata without paging in tensor data."""
    if path.is_symlink() or not path.is_file():
        return None
    try:
        if not _archive_preflight(path, require_data_record=require_data_record):
            return None
        context = safe_globals() if allow_numpy else nullcontext()
        with context:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
    except (
        UnpicklingError,
        EOFError,
        IndexError,
        KeyError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        zlib.error,
    ):
        return None
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        return None
    return payload
