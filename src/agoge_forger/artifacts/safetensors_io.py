from __future__ import annotations

import glob
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._atomic_file import publish_bytes_replace
from ..logging import logger

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None  # type: ignore[misc, assignment]

# Pickle-based model weight files only — not Trainer optimizer/RNG state (*.pt under checkpoint-*).
UNSAFE_WEIGHT_PATTERNS = [
    "pytorch_model.bin",
    "adapter_model.bin",
    "*.ckpt",
]


def inspect_safetensors_file(path: str) -> dict[str, Any]:
    if safe_open is None:
        logger.warning("safetensors library not installed.")
        return {}

    info: dict[str, Any] = {"tensors": {}, "metadata": {}}
    try:
        with safe_open(path, framework="pt") as f:
            info["metadata"] = f.metadata()
            # `safe_open` exposes keys() but is not itself iterable, so `for key
            # in f` raised TypeError on every valid file and escaped the handler
            # below, which does not catch TypeError. SIM118 assumes a dict here
            # and its suggested rewrite is exactly the bug, so it stays silenced.
            for key in f.keys():  # noqa: SIM118
                tensor = f.get_slice(key)
                info["tensors"][key] = {
                    "shape": tensor.get_shape(),
                    "dtype": str(tensor.get_dtype()),
                }
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        # Do not hand back a half-filled result: the caller cannot tell it from a
        # file that genuinely has no tensors, and the CLI would print `{}` and
        # exit 0 on an unreadable file. Log for context, then let the caller's
        # boundary turn it into a reported failure.
        logger.error(f"Failed to inspect safetensors file {path}: {e}")
        raise
    return info


def find_safetensors_files(path: str) -> list[str]:
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return [path]
    if os.path.isdir(path):
        return glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True)
    return []


def assert_no_unsafe_weight_bins(path: str, *, recursive: bool = True) -> None:
    found_unsafe = []

    if os.path.isdir(path):
        for pattern in UNSAFE_WEIGHT_PATTERNS:
            search_root = (
                os.path.join(path, "**", pattern) if recursive else os.path.join(path, pattern)
            )
            matches = glob.glob(search_root, recursive=recursive)
            found_unsafe.extend(matches)

    if found_unsafe:
        raise RuntimeError(
            f"Unsafe weight binaries found in {path}: {found_unsafe}. Safe serialization is required."
        )


def sha256_file(path: str) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute hash for {path}: {e}")
        raise


def write_artifact_index(
    output_dir: str,
    producer_provenance: object | None = None,
) -> str:
    index_path = os.path.join(output_dir, "artifact_index.json")
    artifacts = []

    for root, _, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.abspath(filepath) == os.path.abspath(index_path):
                continue
            rel_path = os.path.relpath(filepath, output_dir)
            size = os.path.getsize(filepath)
            checksum = sha256_file(filepath)
            artifacts.append({"file": rel_path, "size_bytes": size, "sha256": checksum})

    index: dict[str, Any] = {"output_dir": output_dir, "artifacts": artifacts}
    provenance = _producer_provenance_payload(producer_provenance)
    if provenance is not None:
        index["producer_provenance"] = provenance

    publish_bytes_replace(Path(index_path), json.dumps(index, indent=2).encode())
    return index_path


def _producer_provenance_payload(producer_provenance: object | None) -> dict[str, object] | None:
    if producer_provenance is None:
        return None
    dumped = getattr(producer_provenance, "model_dump", None)
    if callable(dumped):
        payload = dumped(mode="json")
    elif isinstance(producer_provenance, Mapping):
        payload = dict(producer_provenance)
    else:
        raise TypeError("producer_provenance must be a mapping or a dumpable model")
    if not isinstance(payload, dict):
        raise TypeError("producer_provenance payload must be a mapping")
    return payload
