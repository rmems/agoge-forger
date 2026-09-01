"""Race-safe safetensor header inspection."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

from safetensors import SafetensorError, safe_open

from ._artifact_schema import IndexedArtifacts


class TensorSchemaEntry(NamedTuple):
    shape: tuple[int, ...]
    dtype: str


_TORCH_TO_SAFETENSORS_DTYPE = {
    "torch.float64": "F64",
    "torch.float32": "F32",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.int64": "I64",
    "torch.int32": "I32",
    "torch.int16": "I16",
    "torch.int8": "I8",
    "torch.uint64": "U64",
    "torch.uint32": "U32",
    "torch.uint16": "U16",
    "torch.uint8": "U8",
    "torch.bool": "BOOL",
    "torch.float8_e4m3fn": "F8_E4M3",
    "torch.float8_e5m2": "F8_E5M2",
    "torch.float8_e8m0fnu": "F8_E8M0",
    "torch.float4_e2m1fn_x2": "F4",
    "torch.complex64": "C64",
}


def safetensors_dtype(dtype: Any) -> str:
    try:
        return _TORCH_TO_SAFETENSORS_DTYPE[str(dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype for safetensors: {dtype}") from exc


def torch_tensor_schema_entry(tensor: Any) -> TensorSchemaEntry:
    dtype = safetensors_dtype(tensor.dtype)
    shape = tuple(tensor.shape)
    if dtype == "F4" and shape:
        shape = (*shape[:-1], shape[-1] * 2)
    return TensorSchemaEntry(shape, dtype)


def require_matching_tensor_schema(
    actual: dict[str, TensorSchemaEntry],
    expected: dict[str, TensorSchemaEntry],
    *,
    label: str,
) -> None:
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    shared = expected.keys() & actual.keys()
    wrong_shapes = sorted(key for key in shared if expected[key].shape != actual[key].shape)
    wrong_dtypes = sorted(key for key in shared if expected[key].dtype != actual[key].dtype)
    if any((missing, unexpected, wrong_shapes, wrong_dtypes)):
        raise ValueError(
            f"{label}: missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"wrong_shapes={wrong_shapes[:5]}, wrong_dtypes={wrong_dtypes[:5]}"
        )


def read_verified_tensor_schema(
    indexed: IndexedArtifacts,
    weights: set[PurePosixPath],
) -> dict[str, TensorSchemaEntry]:
    schema: dict[str, TensorSchemaEntry] = {}
    for portable in sorted(weights):
        _, path = indexed[portable]
        collect_tensor_schema(path, portable, schema)
    return schema


def collect_tensor_schema(
    snapshot: Path,
    portable: PurePosixPath,
    schema: dict[str, TensorSchemaEntry],
) -> None:
    try:
        with safe_open(snapshot, framework="pt") as handle:
            tensor_keys = handle.keys()
            for key in tensor_keys:
                if key in schema:
                    raise ValueError(f"tensor key occurs in multiple model shards: {key}")
                tensor = handle.get_slice(key)
                schema[key] = TensorSchemaEntry(
                    shape=tuple(tensor.get_shape()),
                    dtype=tensor.get_dtype(),
                )
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"invalid safetensors artifact: {portable}") from exc
