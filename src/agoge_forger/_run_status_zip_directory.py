"""Bounded ZIP central-directory validation for restricted state loading."""

from __future__ import annotations

import struct
from typing import BinaryIO

_MAX_ARCHIVE_MEMBERS = 65_536
_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024 * 1024
_MAX_EOCD_TAIL_BYTES = 22 + 65_535
_EOCD = struct.Struct("<4s4H2IH")
_ZIP64_EOCD = struct.Struct("<4sQ2H2I4Q")
_ZIP64_LOCATOR = struct.Struct("<4sIQI")


def _directory_limits_usable(entries: int, size: int) -> bool:
    return all(
        (
            0 < entries <= _MAX_ARCHIVE_MEMBERS,
            size <= _MAX_CENTRAL_DIRECTORY_BYTES,
        )
    )


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
    return all(
        (
            (disk, directory_disk, disk_entries) == (0, 0, entries),
            _directory_limits_usable(entries, size),
            offset + size <= eocd_offset,
        )
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
    return all(
        (
            (disk, directory_disk, disk_entries) == (0, 0, entries),
            _directory_limits_usable(int(entries), int(size)),
            int(offset) + int(size) <= record_offset,
        )
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
    ) and _zip64_values_usable(record_fields, record_offset)


def declared_directory_usable(handle: BinaryIO) -> bool:
    """Return whether the ZIP directory is single-disk and resource-bounded."""
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
