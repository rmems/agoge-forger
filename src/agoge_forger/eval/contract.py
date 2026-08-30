"""Architecture-neutral validation for paired held-out evaluation contracts.

This module freezes comparability metadata only. It intentionally performs no
model loading, inference, scoring, or result generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from safetensors import SafetensorError, safe_open

from ..split_contract import (
    SplitManifest,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_split_manifest,
)

EVALUATION_CONTRACT_VERSION: Literal["agoge.evaluation-contract.v1"] = (
    "agoge.evaluation-contract.v1"
)
ArtifactKind = Literal["peft_adapter", "merged_model"]

_ADAPTER_CONFIG_PATH = PurePosixPath("adapter_config.json")
_ADAPTER_WEIGHTS_PATH = PurePosixPath("adapter_model.safetensors")
_MERGED_CONFIG_PATH = PurePosixPath("config.json")
_MERGED_WEIGHTS_PATH = PurePosixPath("model.safetensors")
_MERGED_WEIGHTS_INDEX_PATH = PurePosixPath("model.safetensors.index.json")
_MODEL_SHARD_PATTERN = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")

COMPARABLE_ARM_FIELDS = (
    "model_repository",
    "model_revision",
    "tokenizer_repository",
    "tokenizer_revision",
    "serializer_id",
    "serializer_version",
    "serializer_sha256",
    "logical_task_set_sha256",
    "context_window",
    "truncation_policy",
    "decoding",
    "scoring_version",
)


class FrozenEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecodingContract(FrozenEvaluationModel):
    do_sample: bool = False
    seed: int
    max_new_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)


class ArtifactIndexEntry(FrozenEvaluationModel):
    file: str = Field(min_length=1)
    size_bytes: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactIndex(FrozenEvaluationModel):
    output_dir: str = Field(min_length=1)
    artifacts: tuple[ArtifactIndexEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_paths(self) -> ArtifactIndex:
        paths = tuple(entry.file for entry in self.artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("artifact index contains duplicate file paths")
        return self


class ArtifactIndexReference(FrozenEvaluationModel):
    kind: ArtifactKind
    artifact_index_path: str = Field(min_length=1)
    artifact_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationArm(FrozenEvaluationModel):
    role: Literal["causal_base", "causal_sft"]
    model_repository: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    artifact: ArtifactIndexReference | None = None
    tokenizer_repository: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    serializer_id: str = Field(min_length=1)
    serializer_version: str = Field(min_length=1)
    serializer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_task_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_window: int = Field(ge=1)
    truncation_policy: Literal["reject", "mark_unsupported", "left", "right"]
    decoding: DecodingContract
    scoring_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_sft_artifact(self) -> EvaluationArm:
        if self.role == "causal_sft" and self.artifact is None:
            raise ValueError("causal_sft arm requires a verified artifact-index reference")
        if self.role == "causal_base" and self.artifact is not None:
            raise ValueError("causal_base arm cannot reference a trained artifact")
        return self


class PairedEvaluationContract(FrozenEvaluationModel):
    schema_version: Literal["agoge.evaluation-contract.v1"] = EVALUATION_CONTRACT_VERSION
    split_manifest_path: str = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    held_out_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_task_ids: tuple[str, ...] = Field(min_length=1)
    logical_task_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base: EvaluationArm
    sft: EvaluationArm

    @model_validator(mode="after")
    def require_pair_comparability(self) -> PairedEvaluationContract:
        _require_arm_roles(self.base, self.sft)
        _require_task_identity(self)
        _require_no_arm_drift(self.base, self.sft)
        return self


def _require_arm_roles(base: EvaluationArm, sft: EvaluationArm) -> None:
    if base.role != "causal_base" or sft.role != "causal_sft":
        raise ValueError("paired contract requires causal_base and causal_sft arms")


def _require_task_identity(contract: PairedEvaluationContract) -> None:
    if len(set(contract.logical_task_ids)) != len(contract.logical_task_ids):
        raise ValueError("logical_task_ids must be unique")
    expected_digest = logical_task_set_sha256(contract.logical_task_ids)
    if contract.logical_task_set_sha256 != expected_digest:
        raise ValueError("logical task-set digest does not match logical_task_ids")
    if contract.base.logical_task_set_sha256 != contract.logical_task_set_sha256:
        raise ValueError("paired arms do not reference the contract logical task set")


def _require_no_arm_drift(base: EvaluationArm, sft: EvaluationArm) -> None:
    drift = [
        field for field in COMPARABLE_ARM_FIELDS if getattr(base, field) != getattr(sft, field)
    ]
    if drift:
        raise ValueError(f"paired evaluation arms are non-comparable; drift in: {drift}")


def logical_task_set_sha256(task_ids: tuple[str, ...] | list[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(task_ids)))


def held_out_task_ids(manifest: SplitManifest) -> tuple[str, ...]:
    return tuple(member.canonical_id for member in manifest.splits["held_out"].members)


def load_evaluation_contract(contract_path: str | Path) -> PairedEvaluationContract:
    path = Path(contract_path).expanduser().resolve(strict=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid evaluation contract JSON: {path}") from exc
    return PairedEvaluationContract.model_validate(value)


def validate_evaluation_contract(contract_path: str | Path) -> PairedEvaluationContract:
    """Validate pair comparability and its immutable #99 held-out reference."""

    path = Path(contract_path).expanduser().resolve(strict=True)
    contract = load_evaluation_contract(path)
    manifest = _validate_manifest_reference(path, contract)
    _validate_held_out_reference(contract, manifest)
    _validate_sft_artifact(path, contract.sft)
    return contract


def _validate_manifest_reference(
    contract_path: Path, contract: PairedEvaluationContract
) -> SplitManifest:
    manifest_path = (contract_path.parent / contract.split_manifest_path).resolve(strict=True)
    if sha256_file(manifest_path) != contract.split_manifest_sha256:
        raise ValueError("evaluation contract split-manifest SHA-256 mismatch")
    return validate_split_manifest(manifest_path)


def _validate_held_out_reference(
    contract: PairedEvaluationContract, manifest: SplitManifest
) -> None:
    if manifest.splits["held_out"].sha256 != contract.held_out_split_sha256:
        raise ValueError("evaluation contract held-out split SHA-256 mismatch")
    expected_ids = held_out_task_ids(manifest)
    if contract.logical_task_ids != expected_ids:
        raise ValueError("evaluation contract task IDs differ from the frozen held-out manifest")
    if logical_task_set_sha256(expected_ids) != contract.logical_task_set_sha256:
        raise ValueError("evaluation contract task-set digest differs from held-out membership")


def build_evaluation_contract(
    *, manifest_path: str | Path, contract_path: str | Path, base: EvaluationArm, sft: EvaluationArm
) -> PairedEvaluationContract:
    """Build and exclusively write a paired contract without evaluating models."""

    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    destination = Path(contract_path).expanduser()
    validated_base = EvaluationArm.model_validate(base.model_dump(mode="json"))
    validated_sft = EvaluationArm.model_validate(sft.model_dump(mode="json"))
    manifest = validate_split_manifest(manifest_file)
    task_ids = held_out_task_ids(manifest)
    task_digest = logical_task_set_sha256(task_ids)
    normalized_sft = _normalize_sft_artifact(validated_sft, destination)
    contract = PairedEvaluationContract(
        split_manifest_path=str(Path(os.path.relpath(manifest_file, destination.parent.resolve()))),
        split_manifest_sha256=sha256_file(manifest_file),
        held_out_split_sha256=manifest.splits["held_out"].sha256,
        logical_task_ids=task_ids,
        logical_task_set_sha256=task_digest,
        base=validated_base,
        sft=normalized_sft,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(canonical_json_bytes(contract.model_dump(mode="json")) + b"\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite evaluation contract: {destination}") from exc
    return contract


def _normalize_sft_artifact(sft: EvaluationArm, destination: Path) -> EvaluationArm:
    if sft.artifact is None:
        raise ValueError("causal_sft arm requires a verified artifact-index reference")
    anchor = destination.parent.resolve()
    supplied = Path(sft.artifact.artifact_index_path).expanduser()
    artifact_path = (
        supplied.resolve(strict=True)
        if supplied.is_absolute()
        else (anchor / supplied).resolve(strict=True)
    )
    _require_artifact_index(
        artifact_path,
        sft.artifact.artifact_index_sha256,
        kind=sft.artifact.kind,
        model_repository=sft.model_repository,
        model_revision=sft.model_revision,
    )
    normalized = ArtifactIndexReference.model_validate(
        {
            **sft.artifact.model_dump(mode="json"),
            "artifact_index_path": str(Path(os.path.relpath(artifact_path, anchor))),
        }
    )
    return EvaluationArm.model_validate(
        {**sft.model_dump(mode="json"), "artifact": normalized.model_dump(mode="json")}
    )


def _validate_sft_artifact(contract_path: Path, sft: EvaluationArm) -> None:
    if sft.artifact is None:
        raise ValueError("causal_sft arm requires a verified artifact-index reference")
    artifact_path = (contract_path.parent / sft.artifact.artifact_index_path).resolve(strict=True)
    _require_artifact_index(
        artifact_path,
        sft.artifact.artifact_index_sha256,
        kind=sft.artifact.kind,
        model_repository=sft.model_repository,
        model_revision=sft.model_revision,
    )


IndexedArtifacts = dict[PurePosixPath, tuple[ArtifactIndexEntry, Path]]


def _require_artifact_index(
    path: Path,
    expected: str,
    *,
    kind: ArtifactKind,
    model_repository: str,
    model_revision: str,
) -> None:
    index = _load_artifact_index(path, expected)
    artifacts = tuple(
        (
            _portable_artifact_path(entry.file),
            entry,
            _resolve_indexed_target(path, entry.file),
        )
        for entry in index.artifacts
    )
    targets = tuple(target for _, _, target in artifacts)
    _require_unique_artifact_targets(targets)
    for _, entry, target in artifacts:
        _require_matching_artifact(entry, target)
    indexed = {portable: (entry, target) for portable, entry, target in artifacts}
    _require_complete_artifact_bundle(path, targets)
    _require_artifact_layout(
        kind,
        indexed,
        model_repository=model_repository,
        model_revision=model_revision,
    )


def _require_complete_artifact_bundle(index_path: Path, targets: tuple[Path, ...]) -> None:
    bundle_files = _enumerate_artifact_bundle(index_path.parent, index_path)
    omitted = bundle_files.difference(targets)
    if omitted:
        relative = sorted(str(path.relative_to(index_path.parent)) for path in omitted)
        raise ValueError(f"artifact bundle contains files omitted from artifact index: {relative}")


def _enumerate_artifact_bundle(root: Path, index_path: Path) -> set[Path]:
    files: set[Path] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError(f"artifact bundle is not readable: {directory}") from exc
        for entry in entries:
            candidate = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"artifact bundle entry is not readable: {candidate}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"artifact bundle cannot contain symlinks: {candidate}")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(candidate)
            elif stat.S_ISREG(metadata.st_mode):
                resolved = candidate.resolve(strict=True)
                if resolved != index_path.resolve(strict=True):
                    files.add(resolved)
            else:
                raise ValueError(f"artifact bundle entry is not a regular file: {candidate}")
    return files


def _require_artifact_layout(
    kind: ArtifactKind,
    indexed: IndexedArtifacts,
    *,
    model_repository: str,
    model_revision: str,
) -> None:
    unsafe = sorted(str(path) for path in indexed if _is_unsafe_weight_path(path))
    if unsafe:
        raise ValueError(f"artifact bundle contains unsafe serialized weights: {unsafe}")
    _require_valid_safetensors(indexed)
    if kind == "peft_adapter":
        _require_peft_adapter_layout(
            indexed,
            model_repository=model_repository,
            model_revision=model_revision,
        )
    else:
        _require_merged_model_layout(indexed)


def _require_peft_adapter_layout(
    indexed: IndexedArtifacts,
    *,
    model_repository: str,
    model_revision: str,
) -> None:
    _require_indexed_paths(
        indexed,
        {_ADAPTER_CONFIG_PATH, _ADAPTER_WEIGHTS_PATH},
        kind="peft_adapter",
    )
    if _MERGED_CONFIG_PATH in indexed:
        raise ValueError("peft_adapter artifact cannot contain merged-model config.json")
    root_safetensors = {
        path for path in indexed if len(path.parts) == 1 and path.suffix == ".safetensors"
    }
    if root_safetensors != {_ADAPTER_WEIGHTS_PATH}:
        raise ValueError(
            "peft_adapter artifact must contain only adapter_model.safetensors weights"
        )
    merged_weights = {
        path
        for path in indexed
        if path == _MERGED_WEIGHTS_INDEX_PATH
        or (path.name.startswith("model") and path.suffix == ".safetensors")
    }
    if merged_weights:
        raise ValueError("peft_adapter artifact cannot contain merged-model weights")
    unexpected_safetensors = sorted(
        str(path) for path in indexed if _is_unexpected_peft_safetensors(path)
    )
    if unexpected_safetensors:
        raise ValueError(
            f"peft_adapter artifact contains unexpected safetensors: {unexpected_safetensors}"
        )
    entry, config_path = indexed[_ADAPTER_CONFIG_PATH]
    config = _load_verified_json(entry, config_path, label="PEFT adapter config")
    _require_nonempty_string(config, "peft_type", label="PEFT adapter config")
    _require_adapter_provenance(
        config,
        model_repository=model_repository,
        model_revision=model_revision,
    )


def _require_adapter_provenance(
    config: dict[str, object],
    *,
    model_repository: str,
    model_revision: str,
) -> None:
    expected = {
        "base_model_name_or_path": model_repository,
        "revision": model_revision,
    }
    for field, expected_value in expected.items():
        if field not in config or not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"PEFT adapter config is missing required {field}")
        if config[field] != expected_value:
            raise ValueError(
                f"PEFT adapter config {field} does not match the contracted SFT arm: "
                f"expected {expected_value!r}, found {config[field]!r}"
            )


def _require_merged_model_layout(indexed: IndexedArtifacts) -> None:
    _require_indexed_paths(indexed, {_MERGED_CONFIG_PATH}, kind="merged_model")
    if _ADAPTER_CONFIG_PATH in indexed or any(
        path.name.startswith("adapter_model") for path in indexed
    ):
        raise ValueError("merged_model artifact cannot contain PEFT adapter files")
    config_entry, config_path = indexed[_MERGED_CONFIG_PATH]
    config = _load_verified_json(config_entry, config_path, label="merged-model config")
    _require_nonempty_string(config, "model_type", label="merged-model config")
    has_single = _MERGED_WEIGHTS_PATH in indexed
    has_sharded = _MERGED_WEIGHTS_INDEX_PATH in indexed
    if has_single == has_sharded:
        raise ValueError(
            "merged_model artifact must contain exactly one of model.safetensors or "
            "model.safetensors.index.json"
        )
    model_weights = {path for path in indexed if path.suffix == ".safetensors"}
    if has_single:
        if model_weights != {_MERGED_WEIGHTS_PATH}:
            raise ValueError("unsharded merged_model artifact contains unexpected model shards")
        return
    entry, shard_index_path = indexed[_MERGED_WEIGHTS_INDEX_PATH]
    shard_index = _load_verified_json(entry, shard_index_path, label="merged-model shard index")
    tensor_shards = _model_tensor_shards(shard_index)
    referenced_shards = set(tensor_shards.values())
    missing = sorted(str(path) for path in referenced_shards if path not in indexed)
    if missing:
        raise ValueError(f"merged-model shard index references unindexed shards: {missing}")
    if referenced_shards != model_weights:
        unexpected = sorted(str(path) for path in model_weights.difference(referenced_shards))
        raise ValueError(
            f"merged_model artifact contains shards absent from weight_map: {unexpected}"
        )
    _require_exact_shard_tensor_map(indexed, tensor_shards)


def _model_tensor_shards(shard_index: dict[str, object]) -> dict[str, PurePosixPath]:
    weight_map = shard_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("merged-model shard index requires a non-empty weight_map")
    metadata = shard_index.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("merged-model shard index requires a metadata object")
    tensor_shards: dict[str, PurePosixPath] = {}
    for tensor_name, value in weight_map.items():
        if not tensor_name:
            raise ValueError("merged-model weight_map tensor names must be non-empty")
        if not isinstance(value, str):
            raise TypeError("merged-model weight_map shard paths must be strings")
        path = _portable_artifact_path(value)
        if value != path.as_posix():
            raise ValueError("merged-model weight_map shard paths must be canonical")
        if path.suffix != ".safetensors":
            raise ValueError("merged-model weight_map must reference safetensors shards only")
        tensor_shards[tensor_name] = path
    _require_canonical_model_shard_set(set(tensor_shards.values()))
    return tensor_shards


def _require_exact_shard_tensor_map(
    indexed: IndexedArtifacts, expected: dict[str, PurePosixPath]
) -> None:
    actual: dict[str, PurePosixPath] = {}
    for shard in set(expected.values()):
        keys = _read_safetensor_keys(indexed[shard][1], shard)
        duplicate = sorted(key for key in keys if key in actual)
        if duplicate:
            raise ValueError(f"tensor keys occur in multiple merged-model shards: {duplicate}")
        actual.update(dict.fromkeys(keys, shard))
    missing = sorted(set(expected).difference(actual))
    if missing:
        raise ValueError(f"merged-model weight_map names tensors absent from shards: {missing}")
    extra = sorted(set(actual).difference(expected))
    if extra:
        raise ValueError(f"merged-model shards contain tensors absent from weight_map: {extra}")
    misplaced = sorted(key for key in expected if expected[key] != actual[key])
    if misplaced:
        raise ValueError(f"merged-model weight_map assigns tensors to wrong shards: {misplaced}")


def _require_canonical_model_shard_set(shards: set[PurePosixPath]) -> None:
    parsed: list[tuple[int, int]] = []
    for shard in shards:
        match = _MODEL_SHARD_PATTERN.fullmatch(shard.as_posix())
        if match is None:
            raise ValueError("merged-model shard names must use model-NNNNN-of-NNNNN.safetensors")
        parsed.append((int(match.group(1)), int(match.group(2))))
    totals = {total for _, total in parsed}
    if len(totals) != 1:
        raise ValueError("merged-model shard names disagree on their total shard count")
    total = totals.pop()
    if total < 1 or {number for number, _ in parsed} != set(range(1, total + 1)):
        raise ValueError("merged-model shard set must be complete and contiguous")


def _is_unsafe_weight_path(path: PurePosixPath) -> bool:
    name = path.name.lower()
    suffixes = {suffix.lower() for suffix in path.suffixes}
    pickle_family = name.startswith(("adapter_model", "pytorch_model")) and ".bin" in suffixes
    return pickle_family or path.suffix.lower() == ".ckpt"


def _is_unexpected_peft_safetensors(path: PurePosixPath) -> bool:
    if path.suffix != ".safetensors" or path == _ADAPTER_WEIGHTS_PATH:
        return False
    return not (
        len(path.parts) == 2
        and re.fullmatch(r"checkpoint-\d+", path.parts[0]) is not None
        and path.name == _ADAPTER_WEIGHTS_PATH.name
    )


def _require_valid_safetensors(indexed: IndexedArtifacts) -> None:
    for portable, (_, path) in indexed.items():
        if portable.suffix != ".safetensors":
            continue
        _read_safetensor_keys(path, portable)


def _read_safetensor_keys(path: Path, portable: PurePosixPath) -> set[str]:
    try:
        with safe_open(path, framework="pt") as handle:
            keys = set(handle.keys())
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"invalid safetensors artifact: {portable}") from exc
    if not keys:
        raise ValueError(f"safetensors file contains no tensors: {portable}")
    return keys


def _require_nonempty_string(config: dict[str, object], field: str, *, label: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} requires non-empty {field}")
    return value


def _require_indexed_paths(
    indexed: IndexedArtifacts, required: set[PurePosixPath], *, kind: ArtifactKind
) -> None:
    missing = sorted(str(path) for path in required.difference(indexed))
    if missing:
        raise ValueError(f"{kind} artifact is missing required indexed files: {missing}")


def _load_verified_json(entry: ArtifactIndexEntry, path: Path, *, label: str) -> dict[str, object]:
    payload = _read_regular_file(path, label=f"invalid {label}")
    if len(payload) != entry.size_bytes or sha256_bytes(payload) != entry.sha256:
        raise ValueError(f"{label} changed after artifact validation: {path}")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"invalid {label}: expected a JSON object")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_artifact_index(path: Path, expected: str) -> ArtifactIndex:
    payload = _read_artifact_index(path)
    actual = sha256_bytes(payload)
    if actual != expected:
        raise ValueError(
            f"SFT artifact-index SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return _parse_artifact_index(path, payload)


def _read_artifact_index(path: Path) -> bytes:
    return _read_regular_file(path, label="invalid SFT artifact index")


def _read_regular_file(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label}: not a regular file: {path}")
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValueError(f"{label}: {path}") from exc
    if _file_identity(before) != _file_identity(after):
        raise ValueError(f"{label}: file changed while reading: {path}")
    return payload


def _parse_artifact_index(path: Path, payload: bytes) -> ArtifactIndex:
    try:
        return ArtifactIndex.model_validate(
            json.loads(payload, object_pairs_hook=_unique_json_object)
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid SFT artifact index: {path}") from exc


def _resolve_indexed_target(index_path: Path, value: str) -> Path:
    target = _resolve_artifact_entry(index_path.parent.resolve(), value)
    if target == index_path.resolve():
        raise ValueError("artifact index cannot list itself")
    return target


def _require_unique_artifact_targets(targets: tuple[Path, ...]) -> None:
    if len(targets) != len(set(targets)):
        raise ValueError("artifact index resolves duplicate targets")


def _require_matching_artifact(entry: ArtifactIndexEntry, path: Path) -> None:
    actual_size, actual_digest = _stream_artifact(path)
    if actual_size != entry.size_bytes:
        raise ValueError(
            f"indexed artifact size mismatch for {entry.file}: "
            f"expected {entry.size_bytes}, found {actual_size}"
        )
    if actual_digest != entry.sha256:
        raise ValueError(
            f"indexed artifact SHA-256 mismatch for {entry.file}: "
            f"expected {entry.sha256}, found {actual_digest}"
        )


def _stream_artifact(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"indexed artifact is not a regular file: {path}")
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValueError(f"indexed artifact is not a readable file: {path}") from exc
    if _file_identity(before) != _file_identity(after):
        raise ValueError(f"indexed artifact changed during validation: {path}")
    return before.st_size, digest.hexdigest()


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _resolve_artifact_entry(root: Path, value: str) -> Path:
    portable = _portable_artifact_path(value)
    resolved = _resolve_existing_artifact(root, portable, value)
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact index path escapes its directory: {value}")
    return resolved


def _portable_artifact_path(value: str) -> PurePosixPath:
    if not value.strip():
        raise ValueError(f"artifact index path must stay relative to its directory: {value}")
    portable = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    unsafe = any(
        (
            portable.is_absolute(),
            windows.is_absolute(),
            bool(windows.drive),
            ".." in portable.parts,
        )
    )
    if unsafe:
        raise ValueError(f"artifact index path must stay relative to its directory: {value}")
    return portable


def _resolve_existing_artifact(root: Path, portable: PurePosixPath, value: str) -> Path:
    try:
        return root.joinpath(*portable.parts).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"indexed artifact does not exist: {value}") from exc
