"""Host and GPU identity checks shared by marker and profile-window joins."""

from __future__ import annotations

from typing import Any


def same_host(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _hostname(left) == _hostname(right) and _hostname(left) is not None


def gpu_match(marker_gpu: Any, sample_gpu: Any) -> bool:
    if not isinstance(marker_gpu, dict) or not isinstance(sample_gpu, dict):
        return False
    marker_uuid = identity_text(marker_gpu.get("uuid"))
    sample_uuid = identity_text(sample_gpu.get("uuid"))
    if marker_uuid is not None and sample_uuid is not None:
        return marker_uuid == sample_uuid
    marker_pci = identity_text(marker_gpu.get("pci_bus_id"))
    sample_pci = identity_text(sample_gpu.get("pci_bus_id"))
    if marker_pci is not None and sample_pci is not None:
        return marker_pci == sample_pci
    return False


def identity_text(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _hostname(record: dict[str, Any]) -> str | None:
    host = record.get("host")
    if not isinstance(host, dict):
        return None
    return identity_text(host.get("hostname"))
