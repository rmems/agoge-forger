import glob
import hashlib
import json
import os
from typing import Any

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
            for key in f:
                tensor = f.get_slice(key)
                info["tensors"][key] = {
                    "shape": tensor.get_shape(),
                    "dtype": str(tensor.get_dtype()),
                }
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        logger.error(f"Failed to inspect safetensors file {path}: {e}")
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


def write_artifact_index(output_dir: str) -> str:
    index_path = os.path.join(output_dir, "artifact_index.json")
    artifacts = []

    for root, _, files in os.walk(output_dir):
        for file in files:
            if file == "artifact_index.json":
                continue
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, output_dir)
            size = os.path.getsize(filepath)
            checksum = sha256_file(filepath)
            artifacts.append({"file": rel_path, "size_bytes": size, "sha256": checksum})

    index = {"output_dir": output_dir, "artifacts": artifacts}

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return index_path
