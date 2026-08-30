"""Lazy diagnostics for the vLLM runtime."""

from __future__ import annotations

import importlib.util
from typing import Any

_vllm_module: Any = None  # pylint: disable=invalid-name


def _import_vllm() -> Any | None:  # pylint: disable=import-outside-toplevel
    global _vllm_module  # pylint: disable=global-statement
    if _vllm_module is None and importlib.util.find_spec("vllm") is not None:
        try:
            import vllm  # pylint: disable=import-error,import-outside-toplevel

            _vllm_module = vllm
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            _vllm_module = None
    return _vllm_module


def get_vllm_version() -> str | None:
    """Return the installed vLLM version, or None if vLLM is not installed."""
    vllm = _import_vllm()
    if vllm is None:
        return None
    return getattr(vllm, "__version__", None)


def get_cuda_version() -> str:
    """Return the PyTorch-detected CUDA version, or 'None' if not available."""
    try:
        import torch  # pylint: disable=import-error,import-outside-toplevel
    except ImportError:
        return "None"
    return torch.version.cuda or "None"


def get_gpu_name() -> str:
    """Return the name of the current CUDA device, or 'None' on CPU."""
    try:
        import torch  # pylint: disable=import-error,import-outside-toplevel
    except ImportError:
        return "None"
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0) or "Unknown"
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return "Unknown"
    return "None"
