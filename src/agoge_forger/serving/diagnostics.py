from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from typing import Any

_vllm_module: Any = None


def _import_vllm() -> Any | None:
    global _vllm_module
    if _vllm_module is None and importlib.util.find_spec("vllm") is not None:
        try:
            import vllm

            _vllm_module = vllm
        except Exception:  # noqa: BLE001
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
    import torch

    return torch.version.cuda or "None"


def get_gpu_name() -> str:
    """Return the name of the current CUDA device, or 'None' on CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0) or "Unknown"
    return "None"


def has_rust_frontend_support() -> tuple[bool, str]:
    """Check whether the installed vLLM build supports the Rust frontend.

    Returns (True, path) when supported, otherwise (False, diagnostic).
    """
    vllm = _import_vllm()
    if vllm is None:
        return False, "vLLM is not installed"

    with _rust_frontend_env(True):
        resolver = getattr(vllm.envs, "_resolve_rust_frontend_path", None)
        if resolver is not None:
            try:
                path = resolver()
                if path:
                    return True, path
                return False, "VLLM_USE_RUST_FRONTEND=1 but no Rust frontend path was resolved"
            except FileNotFoundError as exc:
                return False, f"Rust frontend binary not found: {exc}"
            except Exception as exc:  # noqa: BLE001
                return False, f"Could not resolve Rust frontend path: {type(exc).__name__}: {exc}"

        # Fallback for older vLLM versions that do not expose the resolver.
        candidate = os.path.join(os.path.dirname(vllm.__file__), "vllm-rs")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return True, candidate
        return (
            False,
            (
                f"Rust frontend binary not found at {candidate}; "
                "vLLM version does not expose _resolve_rust_frontend_path"
            ),
        )


def get_environment() -> dict[str, str]:
    """Return vLLM-relevant environment variables for reproducibility.

    Deliberately excludes potential secrets such as HF_TOKEN.
    """
    include_keys = {"CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"}
    return {k: v for k, v in os.environ.items() if k.startswith("VLLM_") or k in include_keys}
