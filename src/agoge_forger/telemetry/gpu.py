"""GPU identity for correlation envelopes. Missing fields stay null, never empty strings."""

from __future__ import annotations

from typing import Any

import torch


def gpu_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _empty_gpu()
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", None)
    pci = getattr(props, "pci_bus_id", None)
    return {
        "pci_bus_id": str(pci) if pci else None,
        "uuid": str(uuid) if uuid else None,
        "name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
    }


def _empty_gpu() -> dict[str, Any]:
    return {
        "pci_bus_id": None,
        "uuid": None,
        "name": None,
        "compute_capability": None,
    }
