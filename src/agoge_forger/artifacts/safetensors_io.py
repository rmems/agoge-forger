from __future__ import annotations

import glob
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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
    try:
        return _inspect_open_safetensors(path)
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        # Do not hand back a half-filled result: the caller cannot tell it from a
        # file that genuinely has no tensors, and the CLI would print `{}` and
        # exit 0 on an unreadable file. Log for context, then let the caller's
        # boundary turn it into a reported failure.
        logger.error(f"Failed to inspect safetensors file {path}: {e}")
        raise


def _inspect_open_safetensors(path: str) -> dict[str, Any]:
    opener = safe_open
    if opener is None:
        return {}
    with opener(path, framework="pt") as handle:
        return _tensors_from_handle(handle)


def _tensors_from_handle(handle: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"tensors": {}, "metadata": handle.metadata()}
    # `safe_open` exposes keys() but is not itself iterable, so `for key
    # in f` raised TypeError on every valid file and escaped the handler
    # below, which does not catch TypeError. SIM118 assumes a dict here
    # and its suggested rewrite is exactly the bug, so it stays silenced.
    for key in handle.keys():  # noqa: SIM118
        tensor = handle.get_slice(key)
        info["tensors"][key] = {
            "shape": tensor.get_shape(),
            "dtype": str(tensor.get_dtype()),
        }
    return info


def find_safetensors_files(path: str) -> list[str]:
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return [path]
    if os.path.isdir(path):
        return glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True)
    return []


def assert_no_unsafe_weight_bins(path: str, *, recursive: bool = True) -> None:
    found_unsafe = _unsafe_weight_hits(path, recursive=recursive)
    if found_unsafe:
        raise RuntimeError(
            f"Unsafe weight binaries found in {path}: {found_unsafe}. Safe serialization is required."
        )


def _unsafe_weight_hits(path: str, *, recursive: bool) -> list[str]:
    if not os.path.isdir(path):
        return []
    found: list[str] = []
    for pattern in UNSAFE_WEIGHT_PATTERNS:
        found.extend(_glob_weight_pattern(path, pattern, recursive))
    return found


def _glob_weight_pattern(path: str, pattern: str, recursive: bool) -> list[str]:
    search_root = os.path.join(path, "**", pattern) if recursive else os.path.join(path, pattern)
    return glob.glob(search_root, recursive=recursive)


def sha256_file(path: str) -> str:
    try:
        return _sha256_bytes(path)
    except Exception as e:
        logger.error(f"Failed to compute hash for {path}: {e}")
        raise


def _sha256_bytes(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for byte_block in iter(lambda: handle.read(4096), b""):
            digest.update(byte_block)
    return digest.hexdigest()


def write_artifact_index(
    output_dir: str,
    producer_provenance: object | None = None,
) -> str:
    index_path = os.path.join(output_dir, "artifact_index.json")
    index: dict[str, Any] = {
        "output_dir": output_dir,
        "artifacts": _listed_artifacts(output_dir, index_path),
    }
    _attach_producer_provenance(index, producer_provenance)
    publish_bytes_replace(Path(index_path), json.dumps(index, indent=2).encode())
    return index_path


def _listed_artifacts(output_dir: str, index_path: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for root, _, files in os.walk(output_dir):
        artifacts.extend(_artifacts_in_dir(root, files, output_dir, index_path))
    return artifacts


def _artifacts_in_dir(
    root: str, files: list[str], output_dir: str, index_path: str
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for file in files:
        entry = _artifact_entry(root, file, output_dir, index_path)
        if entry is not None:
            entries.append(entry)
    return entries


def _artifact_entry(
    root: str, file: str, output_dir: str, index_path: str
) -> dict[str, Any] | None:
    filepath = os.path.join(root, file)
    if os.path.abspath(filepath) == os.path.abspath(index_path):
        return None
    return {
        "file": os.path.relpath(filepath, output_dir),
        "size_bytes": os.path.getsize(filepath),
        "sha256": sha256_file(filepath),
    }


def _attach_producer_provenance(index: dict[str, Any], producer_provenance: object | None) -> None:
    provenance = _producer_provenance_payload(producer_provenance)
    if provenance is not None:
        index["producer_provenance"] = provenance


@runtime_checkable
class _JsonDumpable(Protocol):
    def model_dump(self, *, mode: str) -> object: ...


def _producer_provenance_payload(producer_provenance: object | None) -> dict[str, object] | None:
    if producer_provenance is None:
        return None
    return _require_mapping_payload(_dump_producer_provenance(producer_provenance))


def _dump_producer_provenance(producer_provenance: object) -> object:
    if isinstance(producer_provenance, Mapping):
        return dict(producer_provenance)
    if isinstance(producer_provenance, _JsonDumpable):
        return producer_provenance.model_dump(mode="json")
    raise TypeError("producer_provenance must be a mapping or a dumpable model")


def _require_mapping_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("producer_provenance payload must be a mapping")
    return payload
