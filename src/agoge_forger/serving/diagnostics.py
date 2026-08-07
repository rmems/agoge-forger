"""Lazy diagnostics for the vLLM runtime and Rust frontend support."""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
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


@contextmanager
def _rust_frontend_env(enabled: bool = True):
    key = "VLLM_USE_RUST_FRONTEND"
    original = os.environ.get(key)
    os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


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


def has_rust_frontend_support() -> tuple[bool, str]:
    """Check whether the installed vLLM build supports the Rust frontend.

    Returns (True, path) when supported, otherwise (False, diagnostic).
    """
    result: tuple[bool, str] = (False, "vLLM is not installed")
    vllm = _import_vllm()
    if vllm is None:
        return result

    with _rust_frontend_env(True):
        resolver = getattr(vllm.envs, "_resolve_rust_frontend_path", None)
        if resolver is not None:
            try:
                path = resolver()
                result = (
                    (True, path)
                    if path
                    else (
                        False,
                        "VLLM_USE_RUST_FRONTEND=1 but no Rust frontend path was resolved",
                    )
                )
            except FileNotFoundError as exc:
                result = (False, f"Rust frontend binary not found: {exc}")
            except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                result = (
                    False,
                    f"Could not resolve Rust frontend path: {type(exc).__name__}: {exc}",
                )
        else:
            # Fallback for older vLLM versions that do not expose the resolver.
            candidate = os.path.join(os.path.dirname(vllm.__file__), "vllm-rs")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                result = (True, candidate)
            else:
                result = (
                    False,
                    f"Rust frontend binary not found at {candidate}; "
                    + "vLLM version does not expose _resolve_rust_frontend_path",
                )

    return result


def get_environment() -> dict[str, str]:
    """Return vLLM-relevant environment variables for reproducibility.

    Deliberately excludes potential secrets such as HF_TOKEN.
    """
    include_keys = {"CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"}
    excluded_vllm = {"VLLM_API_KEY"}
    return {
        k: v
        for k, v in os.environ.items()
        if (k.startswith("VLLM_") and k not in excluded_vllm) or k in include_keys
    }
