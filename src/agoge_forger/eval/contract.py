"""Architecture-neutral validation for paired held-out evaluation contracts.

This module freezes comparability metadata only. It intentionally performs no
model loading, inference, scoring, or result generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
    kind: Literal["peft_adapter", "merged_model"]
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
    )


def _require_artifact_index(path: Path, expected: str) -> None:
    index = _load_artifact_index(path, expected)
    targets = tuple(_resolve_indexed_target(path, entry.file) for entry in index.artifacts)
    _require_unique_artifact_targets(targets)
    for entry, artifact_path in zip(index.artifacts, targets, strict=True):
        _require_matching_artifact(entry, artifact_path)


def _load_artifact_index(path: Path, expected: str) -> ArtifactIndex:
    payload = _read_artifact_index(path)
    actual = sha256_bytes(payload)
    if actual != expected:
        raise ValueError(
            f"SFT artifact-index SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return _parse_artifact_index(path, payload)


def _read_artifact_index(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"invalid SFT artifact index: {path}") from exc


def _parse_artifact_index(path: Path, payload: bytes) -> ArtifactIndex:
    try:
        return ArtifactIndex.model_validate(json.loads(payload))
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
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
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"indexed artifact changed during validation: {path}")
    return before.st_size, digest.hexdigest()


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
