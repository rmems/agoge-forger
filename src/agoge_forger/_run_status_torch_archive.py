"""Restricted, memory-mapped validation for current PyTorch ZIP state."""

from __future__ import annotations

import inspect
import zipfile
import zlib
from contextlib import nullcontext
from pathlib import Path
from pickle import UnpicklingError  # nosec B403 - exception type only; no pickle loading
from typing import Any

import torch
from transformers.trainer_pt_utils import safe_globals

from agoge_forger._run_status_zip_directory import declared_directory_usable

_MAX_PICKLE_METADATA_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 65_536
_CRC_CHUNK_BYTES = 1024 * 1024
# Without mmap, torch.load materializes every storage. One GiB accommodates
# large LoRA optimizer states while bounding legacy-Torch resident memory.
_MAX_MATERIALIZED_STORAGE_BYTES = 1024**3


def _bounded_archive_infos(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo] | None:
    infos = archive.infolist()
    return infos if infos and len(infos) <= _MAX_ARCHIVE_MEMBERS else None


def _single_archive_root(names: list[str]) -> str | None:
    roots = {name.partition("/")[0] for name in names}
    if len(roots) != 1:
        return None
    root = next(iter(roots))
    names_are_rooted = all(name.startswith(f"{root}/") for name in names)
    return root if all((bool(root), names_are_rooted)) else None


def _archive_members(
    archive: zipfile.ZipFile,
) -> tuple[str, dict[str, zipfile.ZipInfo]] | None:
    infos = _bounded_archive_infos(archive)
    if infos is None:
        return None
    names = [info.filename for info in infos]
    members = dict(zip(names, infos, strict=True))
    root = _single_archive_root(names)
    if len(members) != len(infos) or root is None:
        return None
    return root, members


def _storage_record(info: zipfile.ZipInfo, root: str) -> bool:
    prefix = f"{root}/data/"
    suffix = info.filename.removeprefix(prefix)
    return (
        info.filename.startswith(prefix),
        suffix.isascii(),
        suffix.isdigit(),
        info.is_dir(),
        info.compress_type,
        info.compress_size == info.file_size,
    ) == (True, True, True, False, zipfile.ZIP_STORED, True)


def _metadata_record_usable(info: zipfile.ZipInfo) -> bool:
    return not info.is_dir() and min(info.file_size, info.compress_size) >= 0


def _stream_crc_usable(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_bytes: int,
) -> bool:
    total = 0
    with archive.open(info) as member:
        while block := member.read(min(_CRC_CHUNK_BYTES, max_bytes - total + 1)):
            total += len(block)
            if total > info.file_size or total > max_bytes:
                return False
    return total == info.file_size


def _metadata_crc_usable(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> bool:
    if not all(_metadata_record_usable(info) for info in infos):
        return False
    if (
        max(
            sum(info.file_size for info in infos),
            sum(info.compress_size for info in infos),
        )
        > _MAX_PICKLE_METADATA_BYTES
    ):
        return False
    return all(_stream_crc_usable(archive, info, max_bytes=info.file_size) for info in infos)


def _storage_inventory(
    root: str,
    members: dict[str, zipfile.ZipInfo],
) -> tuple[list[zipfile.ZipInfo], list[zipfile.ZipInfo]]:
    storage = [info for info in members.values() if _storage_record(info, root)]
    storage_names = {info.filename for info in storage}
    metadata = [info for name, info in members.items() if name not in storage_names]
    return storage, metadata


def _storage_names_usable(
    root: str,
    members: dict[str, zipfile.ZipInfo],
    storage: list[zipfile.ZipInfo],
) -> bool:
    data_prefix = f"{root}/data/"
    data_names = {name for name in members if name.startswith(data_prefix)}
    return data_names == {info.filename for info in storage}


def _required_metadata_present(root: str, members: dict[str, zipfile.ZipInfo]) -> bool:
    required = {
        f"{root}/data.pkl",
        f"{root}/version",
        f"{root}/.data/serialization_id",
    }
    return required.issubset(members)


def _storage_size_usable(storage: list[zipfile.ZipInfo], max_storage_bytes: int | None) -> bool:
    return max_storage_bytes is None or sum(info.file_size for info in storage) <= max_storage_bytes


def _storage_inventory_usable(
    storage: list[zipfile.ZipInfo],
    *,
    require_data_record: bool,
    max_storage_bytes: int | None,
) -> bool:
    if require_data_record and not storage:
        return False
    return _storage_size_usable(storage, max_storage_bytes)


def _archive_contents_usable(
    archive: zipfile.ZipFile,
    *,
    require_data_record: bool,
    max_storage_bytes: int | None,
) -> bool:
    archive_parts = _archive_members(archive)
    if archive_parts is None:
        return False
    root, members = archive_parts
    storage, metadata = _storage_inventory(root, members)
    if not _required_metadata_present(root, members):
        return False
    if not _storage_names_usable(root, members, storage):
        return False
    if not _storage_inventory_usable(
        storage,
        require_data_record=require_data_record,
        max_storage_bytes=max_storage_bytes,
    ):
        return False
    return all(
        _stream_crc_usable(archive, info, max_bytes=info.file_size) for info in storage
    ) and _metadata_crc_usable(archive, metadata)


def _archive_preflight(
    path: Path,
    *,
    require_data_record: bool,
    max_storage_bytes: int | None,
) -> bool:
    with path.open("rb") as handle:
        if not declared_directory_usable(handle):
            return False
        handle.seek(0)
        with zipfile.ZipFile(handle) as archive:
            return _archive_contents_usable(
                archive,
                require_data_record=require_data_record,
                max_storage_bytes=max_storage_bytes,
            )


def _string_mapping(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        return None
    return payload


def _restricted_load_kwargs(loader: Any, *, allow_mmap: bool) -> dict[str, Any]:
    try:
        parameters = inspect.signature(loader).parameters
    except (TypeError, ValueError) as error:
        raise RuntimeError("cannot verify the restricted torch.load interface") from error
    if "weights_only" not in parameters:
        raise RuntimeError("torch.load does not support restricted weights-only loading")
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": True,
    }
    if allow_mmap and "mmap" in parameters:
        kwargs["mmap"] = True
    return kwargs


def _legacy_file_usable(path: Path) -> bool:
    return path.stat().st_size <= _MAX_MATERIALIZED_STORAGE_BYTES


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
        loader = torch.load
        is_zip = zipfile.is_zipfile(path)
        load_kwargs = _restricted_load_kwargs(loader, allow_mmap=is_zip)
        max_storage_bytes = (
            None if load_kwargs.get("mmap") is True else _MAX_MATERIALIZED_STORAGE_BYTES
        )
        if is_zip:
            if not _archive_preflight(
                path,
                require_data_record=require_data_record,
                max_storage_bytes=max_storage_bytes,
            ):
                return None
        elif not _legacy_file_usable(path):
            return None
        context = safe_globals() if allow_numpy else nullcontext()
        with context:
            payload = loader(path, **load_kwargs)
    except (
        UnpicklingError,
        EOFError,
        IndexError,
        KeyError,
        RuntimeError,  # Includes unsupported ZIP compression's NotImplementedError.
        ValueError,
        zipfile.BadZipFile,
        zlib.error,
    ):
        return None
    return _string_mapping(payload)
