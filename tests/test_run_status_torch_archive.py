"""Resource-boundary tests for restricted PyTorch archive inspection."""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pytest

from agoge_forger._run_status_torch_archive import torch_mapping


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


def test_oversized_pickle_metadata_is_rejected_before_deserialization(tmp_path, monkeypatch):
    state = tmp_path / "optimizer.pt"
    _write_zip(state, pickle_payload=b"x" * (64 * 1024 * 1024 + 1))

    def fail_load(*args, **kwargs):
        raise AssertionError("oversized pickle metadata reached torch.load")

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", fail_load)

    assert torch_mapping(state) is None


def test_data_directory_is_not_a_tensor_storage_record(tmp_path, monkeypatch):
    state = tmp_path / "optimizer.pt"
    _write_zip(state, pickle_payload=b"\x80\x02}q\x00.", directory_data=True)

    def fail_load(*args, **kwargs):
        raise AssertionError("directory-only storage reached torch.load")

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", fail_load)

    assert torch_mapping(state, require_data_record=True) is None


def test_compressed_tensor_storage_is_rejected_before_deserialization(tmp_path, monkeypatch):
    state = tmp_path / "optimizer.pt"
    _write_zip(
        state,
        pickle_payload=b"\x80\x02}q\x00.",
        compress_storage=True,
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("compressed tensor storage reached torch.load")

    monkeypatch.setattr("agoge_forger._run_status_torch_archive.torch.load", fail_load)

    assert torch_mapping(state, require_data_record=True) is None


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
