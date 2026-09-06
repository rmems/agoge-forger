"""Resource-boundary tests for restricted PyTorch archive inspection."""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import cast

import pytest
import torch

from agoge_forger._run_status_torch_archive import (
    _metadata_crc_usable,
    _storage_record,
    _stream_crc_usable,
    torch_mapping,
)


def _write_zip(
    path: Path,
    *,
    pickle_payload: bytes,
    directory_data: bool = False,
    compress_storage: bool = False,
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("archive/data.pkl", pickle_payload)
        storage_compression = zipfile.ZIP_DEFLATED if compress_storage else zipfile.ZIP_STORED
        archive.writestr(
            "archive/data/" if directory_data else "archive/data/0",
            b"storage",
            compress_type=storage_compression,
        )
        archive.writestr("archive/version", b"3\n")
        archive.writestr("archive/.data/serialization_id", b"0")


def _oversized_eocd(*, zip64: bool) -> bytes:
    if not zip64:
        return struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 1, 1, 64 * 1024 * 1024 + 1, 0, 0)
    zip64_record = struct.pack(
        "<4sQ2H2I4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        65_537,
        65_537,
        0,
        0,
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, 0, 1)
    classic = struct.pack(
        "<4s4H2IH",
        b"PK\x05\x06",
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    return zip64_record + locator + classic


def _declare_unsupported_compression(path: Path, member: str) -> None:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    payload = bytearray(path.read_bytes())
    unsupported = (99).to_bytes(2, "little")
    payload[info.header_offset + 8 : info.header_offset + 10] = unsupported
    central_offset = payload.index(b"PK\x01\x02")
    encoded_name = member.encode()
    assert payload[central_offset + 46 : central_offset + 46 + len(encoded_name)] == encoded_name
    payload[central_offset + 10 : central_offset + 12] = unsupported
    path.write_bytes(payload)


def _declare_storage_size(path: Path, member: str, size: int) -> None:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    payload = bytearray(path.read_bytes())
    encoded_size = size.to_bytes(4, "little")
    payload[info.header_offset + 18 : info.header_offset + 26] = encoded_size * 2
    central_offset = payload.index(b"PK\x01\x02")
    encoded_name = member.encode()
    while payload[central_offset + 46 : central_offset + 46 + len(encoded_name)] != encoded_name:
        name_size = int.from_bytes(payload[central_offset + 28 : central_offset + 30], "little")
        extra_size = int.from_bytes(payload[central_offset + 30 : central_offset + 32], "little")
        comment_size = int.from_bytes(payload[central_offset + 32 : central_offset + 34], "little")
        central_offset += 46 + name_size + extra_size + comment_size
    payload[central_offset + 20 : central_offset + 28] = encoded_size * 2
    path.write_bytes(payload)


def _declare_storage_compressed_size(path: Path, member: str, size: int) -> None:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    payload = bytearray(path.read_bytes())
    encoded_size = size.to_bytes(4, "little")
    payload[info.header_offset + 18 : info.header_offset + 22] = encoded_size
    central_offset = payload.index(b"PK\x01\x02")
    encoded_name = member.encode()
    while payload[central_offset + 46 : central_offset + 46 + len(encoded_name)] != encoded_name:
        name_size = int.from_bytes(payload[central_offset + 28 : central_offset + 30], "little")
        extra_size = int.from_bytes(payload[central_offset + 30 : central_offset + 32], "little")
        comment_size = int.from_bytes(payload[central_offset + 32 : central_offset + 34], "little")
        central_offset += 46 + name_size + extra_size + comment_size
    payload[central_offset + 20 : central_offset + 24] = encoded_size
    path.write_bytes(payload)


def _corrupt_stored_member(path: Path, member: str) -> None:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    with path.open("r+b") as handle:
        handle.seek(info.header_offset + 26)
        name_size = int.from_bytes(handle.read(2), "little")
        extra_size = int.from_bytes(handle.read(2), "little")
        handle.seek(info.header_offset + 30 + name_size + extra_size)
        first_byte = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([first_byte[0] ^ 0xFF]))


@pytest.mark.parametrize("zip64", [False, True], ids=["classic-size", "zip64-count"])
def test_oversized_central_directory_is_rejected_before_zipfile_parsing(
    tmp_path, monkeypatch, zip64
):
    state = tmp_path / "optimizer.pt"
    state.write_bytes(_oversized_eocd(zip64=zip64))

    def fail_zipfile(*args, **kwargs):
        raise AssertionError("oversized central directory reached ZipFile")

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.zipfile.ZipFile", fail_zipfile)

    assert torch_mapping(state) is None


@pytest.mark.parametrize(
    ("zip_options", "require_data_record"),
    [
        ({"pickle_payload": b"x" * (64 * 1024 * 1024 + 1)}, False),
        ({"pickle_payload": b"\x80\x02}q\x00.", "directory_data": True}, True),
        ({"pickle_payload": b"\x80\x02}q\x00.", "compress_storage": True}, True),
    ],
    ids=["oversized-metadata", "directory-storage", "compressed-storage"],
)
def test_invalid_archive_layout_is_rejected_before_deserialization(
    tmp_path,
    monkeypatch,
    zip_options,
    require_data_record,
):
    state = tmp_path / "optimizer.pt"
    _write_zip(state, **zip_options)

    def fail_load(*args, **kwargs):
        raise AssertionError("invalid archive layout reached torch.load")

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", fail_load)

    assert torch_mapping(state, require_data_record=require_data_record) is None


@pytest.mark.parametrize("member", ["archive/data.pkl", "archive/version"])
def test_corrupt_compressed_metadata_fails_closed(tmp_path, monkeypatch, member):
    state = tmp_path / "optimizer.pt"
    _write_zip(state, pickle_payload=b"\x80\x02}q\x00.")
    with zipfile.ZipFile(state) as archive:
        info = archive.getinfo(member)
    with state.open("r+b") as handle:
        handle.seek(info.header_offset + 26)
        name_size = int.from_bytes(handle.read(2), "little")
        extra_size = int.from_bytes(handle.read(2), "little")
        handle.seek(info.header_offset + 30 + name_size + extra_size)
        first_byte = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([first_byte[0] ^ 0xFF]))

    def fail_load(*args, **kwargs):
        return {"unexpected": "deserialized"}

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", fail_load)

    assert torch_mapping(state) is None


def test_corrupt_tensor_storage_fails_before_deserialization(tmp_path, monkeypatch):
    state = tmp_path / "optimizer.pt"
    torch.save({"value": torch.arange(16)}, state)
    with zipfile.ZipFile(state) as archive:
        storage = next(info.filename for info in archive.infolist() if "/data/" in info.filename)
    _corrupt_stored_member(state, storage)

    def fail_load(path, *, map_location, weights_only, mmap):
        raise AssertionError("corrupt tensor storage reached torch.load")

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", fail_load)

    assert torch_mapping(state, require_data_record=True) is None


def test_stored_tensor_requires_equal_compressed_and_uncompressed_sizes(tmp_path, monkeypatch):
    state = tmp_path / "optimizer.pt"
    _write_zip(state, pickle_payload=b"\x80\x02}q\x00.")
    _declare_storage_compressed_size(state, "archive/data/0", 1)
    with zipfile.ZipFile(state) as archive:
        forged = archive.getinfo("archive/data/0")

    assert _storage_record(forged, "archive") is False

    def fail_load(path, *, map_location, weights_only, mmap):
        raise AssertionError("forged storage sizes reached torch.load")

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", fail_load)

    assert torch_mapping(state, require_data_record=True) is None


def test_crc_stream_stops_at_declared_or_budgeted_size():
    class Member:
        reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            self.reads += 1
            if self.reads > 2:
                raise AssertionError("CRC stream read beyond its byte budget")
            return b"1234"

    member = Member()

    class Archive:
        def open(self, info):
            return member

    info = zipfile.ZipInfo("archive/data/0")
    info.file_size = 4

    archive = cast(zipfile.ZipFile, Archive())
    assert _stream_crc_usable(archive, info, max_bytes=4) is False
    assert member.reads == 2


def test_metadata_compressed_and_uncompressed_totals_are_bounded_before_streaming():
    info = zipfile.ZipInfo("archive/data.pkl")
    info.file_size = 1
    info.compress_size = 64 * 1024 * 1024 + 1

    class Archive:
        def open(self, member):
            raise AssertionError("oversized compressed metadata reached streaming")

    archive = cast(zipfile.ZipFile, Archive())
    assert _metadata_crc_usable(archive, [info]) is False


def test_legacy_torch_serialization_is_restricted_and_supported(tmp_path, monkeypatch):
    state = tmp_path / "adapter_model.bin"
    payload = {"value": torch.tensor([1, 2, 3])}
    torch.save(payload, state, _use_new_zipfile_serialization=False)
    real_load = torch.load
    calls = []

    def restricted_load(path, *, map_location, weights_only):
        calls.append((map_location, weights_only))
        return real_load(path, map_location=map_location, weights_only=weights_only)

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", restricted_load)

    loaded = torch_mapping(state, require_data_record=True)

    assert loaded is not None
    assert loaded["value"].tolist() == [1, 2, 3]
    assert calls == [("cpu", True)]


def test_oversized_legacy_serialization_is_rejected_before_loading(tmp_path, monkeypatch):
    state = tmp_path / "adapter_model.bin"
    state.write_bytes(b"not-a-zip")
    state.touch()
    with state.open("r+b") as handle:
        handle.truncate(1024**3 + 1)

    def fail_load(*args, **kwargs):
        raise AssertionError("oversized legacy serialization reached torch.load")

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", fail_load)

    assert torch_mapping(state, require_data_record=True) is None


def test_unsupported_metadata_compression_fails_closed(tmp_path):
    state = tmp_path / "optimizer.pt"
    _write_zip(state, pickle_payload=b"\x80\x02}q\x00.")
    _declare_unsupported_compression(state, "archive/data.pkl")

    assert torch_mapping(state) is None


def test_torch_load_without_mmap_keeps_restricted_loading(tmp_path, monkeypatch):
    state = tmp_path / "optimizer.pt"
    torch.save({"value": torch.tensor([1])}, state)
    real_load = torch.load
    calls = []

    def load_without_mmap(path, *, map_location, weights_only):
        calls.append((map_location, weights_only))
        return real_load(path, map_location=map_location, weights_only=weights_only)

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", load_without_mmap)

    payload = torch_mapping(state, require_data_record=True)

    assert payload is not None
    assert payload["value"].tolist() == [1]
    assert calls == [("cpu", True)]


def test_oversized_storage_is_rejected_before_non_mmap_loader(tmp_path, monkeypatch):
    state = tmp_path / "optimizer.pt"
    _write_zip(state, pickle_payload=b"\x80\x02}q\x00.")
    _declare_storage_size(state, "archive/data/0", 1024**3 + 1)

    def legacy_load(path, *, map_location, weights_only):
        raise AssertionError("oversized storage reached the non-mmap loader")

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", legacy_load)
    assert torch_mapping(state, require_data_record=True) is None


def test_mmap_loader_keeps_non_materializing_storage_size_contract(tmp_path, monkeypatch):
    state = tmp_path / "optimizer.pt"
    state.write_bytes(b"PK placeholder")
    preflight_limits = []

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.zipfile.is_zipfile", lambda p: True)

    def preflight(path, *, require_data_record, max_storage_bytes):
        preflight_limits.append(max_storage_bytes)
        return True

    monkeypatch.setattr("agoge_forger._run_status_torch_archive._archive_preflight", preflight)

    def mmap_load(path, *, map_location, weights_only, mmap):
        return {}

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", mmap_load)

    assert torch_mapping(state, require_data_record=True) == {}
    assert preflight_limits == [None]
