"""Pre-registered experiment freeze for base-before-SFT measured runs.

Experiment contracts are versioned. ``parse_experiment_contract`` dispatches on
``schema_version`` to the v1 or v2 schema, and every loader, validator, and the
freeze writer (``build_experiment_contract``) goes through it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import Field, ValidationError, field_validator, model_validator

from .._atomic_file import publish_bytes_noreplace, write_fsynced_bytes
from .._strict_json import decode_json_object
from ..split_contract import SplitManifest, canonical_json_bytes, sha256_bytes, sha256_file
from ..split_validation import validate_split_manifest_snapshot
from ._artifact_schema import FrozenEvaluationModel, portable_contract_reference
from ._experiment_contract_v1 import (
    EXPERIMENT_CONTRACT_VERSION,
    SHA256_PATTERN,
    DatasetProvenance,
    EvaluationPin,
    ExperimentContract,
    ModelPin,
    SerializerPin,
    SplitPin,
    TrainingPin,
)
from ._experiment_contract_v2 import EXPERIMENT_CONTRACT_V2_VERSION, ExperimentContractV2
from .contract import (
    EvaluationArm,
    held_out_task_ids,
    logical_task_set_sha256,
)

__all__ = [
    "EXPERIMENT_CONTRACT_V2_VERSION",
    "EXPERIMENT_CONTRACT_VERSION",
    "AnyExperimentContract",
    "DatasetProvenance",
    "EvaluationPin",
    "ExperimentContract",
    "ExperimentContractError",
    "ExperimentContractV2",
    "G0EvaluationContract",
    "ModelPin",
    "SerializerPin",
    "SplitPin",
    "SplitPolicyPin",
    "TrainingPin",
    "build_experiment_contract",
    "build_g0_evaluation_contract",
    "compose_g0_evaluation_contract",
    "load_experiment_contract",
    "parse_experiment_contract",
    "split_pin_from_manifest",
    "validate_experiment_contract",
]

_SPLIT_NAMES: tuple[Literal["train", "validation", "held_out"], ...] = (
    "train",
    "validation",
    "held_out",
)

AnyExperimentContract = ExperimentContract | ExperimentContractV2
_CONTRACT_MODELS: dict[str, type[ExperimentContract | ExperimentContractV2]] = {
    EXPERIMENT_CONTRACT_VERSION: ExperimentContract,
    EXPERIMENT_CONTRACT_V2_VERSION: ExperimentContractV2,
}


class ExperimentContractError(ValueError):
    """An experiment contract failed schema validation; the message names each field."""


class SplitPolicyPin(NamedTuple):
    """Seed, salt, and weights used when pinning a frozen split."""

    split_seed: int
    split_salt: str
    train_weight: int
    validation_weight: int
    held_out_weight: int


class G0EvaluationContract(FrozenEvaluationModel):
    """Held-out base evaluation slice frozen before any SFT artifact exists."""

    schema_version: Literal["agoge.g0-evaluation-contract.v1"] = "agoge.g0-evaluation-contract.v1"
    experiment_id: str = Field(min_length=1)
    split_manifest_path: str = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    held_out_split_sha256: str = Field(pattern=SHA256_PATTERN)
    logical_task_ids: tuple[str, ...] = Field(min_length=1)
    logical_task_set_sha256: str = Field(pattern=SHA256_PATTERN)
    base: EvaluationArm

    @field_validator("split_manifest_path")
    @classmethod
    def require_relative_manifest_path(cls, value: str) -> str:
        return portable_contract_reference(value)

    @model_validator(mode="after")
    def require_base_only(self) -> G0EvaluationContract:
        if self.base.role != "causal_base" or self.base.artifact is not None:
            raise ValueError("G0 contract requires a causal_base arm without an artifact")
        if logical_task_set_sha256(self.logical_task_ids) != self.logical_task_set_sha256:
            raise ValueError("logical task-set digest does not match logical_task_ids")
        if self.base.logical_task_set_sha256 != self.logical_task_set_sha256:
            raise ValueError("base arm task-set digest mismatch")
        return self


def parse_experiment_contract(value: Mapping[str, object]) -> AnyExperimentContract:
    """Validate a decoded contract against the schema its ``schema_version`` names.

    ``schema_version`` is required: without it there is nothing to dispatch on,
    and guessing v1 would report a v2 contract's new fields as unknown keys
    instead of naming the missing version. Every frozen v1 contract written by
    ``build_experiment_contract`` already carries it. Failures raise
    ``ExperimentContractError`` listing every offending field by dotted path.
    """
    if "schema_version" not in value:
        raise ExperimentContractError(
            "invalid experiment contract: schema_version: missing required field"
        )
    version = value["schema_version"]
    if not isinstance(version, str) or version not in _CONTRACT_MODELS:
        supported = ", ".join(sorted(_CONTRACT_MODELS))
        raise ExperimentContractError(
            f"schema_version: unsupported experiment contract version {version!r} "
            f"(supported: {supported})"
        )
    try:
        return _CONTRACT_MODELS[version].model_validate(value)
    except ValidationError as exc:
        raise ExperimentContractError(_describe_validation_error(version, exc)) from exc


def load_experiment_contract(contract_path: str | Path) -> AnyExperimentContract:
    path = Path(contract_path).expanduser().resolve(strict=True)
    try:
        value = decode_json_object(
            path.read_bytes(),
            str(path),
            object_label="experiment contract",
        )
    except ValueError as exc:
        raise ValueError(f"invalid experiment contract JSON: {path}") from exc
    return parse_experiment_contract(value)


def validate_experiment_contract(contract_path: str | Path) -> AnyExperimentContract:
    path = Path(contract_path).expanduser().resolve(strict=True)
    contract = load_experiment_contract(path)
    manifest = _validate_split_pin(path, contract.split)
    _validate_held_out_counts(contract.split, manifest)
    return contract


def build_experiment_contract(
    *,
    contract_path: str | Path,
    contract: AnyExperimentContract,
) -> AnyExperimentContract:
    """Freeze ``contract`` to a new file after re-validating it by schema version."""
    destination = Path(contract_path).expanduser()
    validated = parse_experiment_contract(contract.model_dump(mode="json"))
    payload = canonical_json_bytes(validated.model_dump(mode="json")) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes_noreplace(
        destination,
        payload,
        refusal="refusing to overwrite experiment contract",
        writer=_write_contract_payload,
    )
    return validated


def compose_g0_evaluation_contract(
    *,
    manifest_path: str | Path,
    contract_path: str | Path,
    experiment_id: str,
    base: EvaluationArm,
) -> G0EvaluationContract:
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    destination = Path(contract_path).expanduser()
    validated_base = EvaluationArm.model_validate(base.model_dump(mode="json"))
    if validated_base.role != "causal_base" or validated_base.artifact is not None:
        raise ValueError("G0 contract requires causal_base without artifact")
    manifest_snapshot = manifest_file.read_bytes()
    manifest = validate_split_manifest_snapshot(manifest_file, manifest_snapshot)
    task_ids = held_out_task_ids(manifest)
    task_digest = logical_task_set_sha256(task_ids)
    return G0EvaluationContract(
        experiment_id=experiment_id,
        split_manifest_path=_portable_relative_path(manifest_file, destination.parent.resolve()),
        split_manifest_sha256=sha256_bytes(manifest_snapshot),
        held_out_split_sha256=manifest.splits["held_out"].sha256,
        logical_task_ids=task_ids,
        logical_task_set_sha256=task_digest,
        base=validated_base.model_copy(update={"logical_task_set_sha256": task_digest}),
    )


def build_g0_evaluation_contract(
    *,
    manifest_path: str | Path,
    contract_path: str | Path,
    experiment_id: str,
    base: EvaluationArm,
) -> G0EvaluationContract:
    destination = Path(contract_path).expanduser()
    contract = compose_g0_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=destination,
        experiment_id=experiment_id,
        base=base,
    )
    payload = canonical_json_bytes(contract.model_dump(mode="json")) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes_noreplace(
        destination,
        payload,
        refusal="refusing to overwrite G0 evaluation contract",
        writer=_write_contract_payload,
    )
    return contract


def split_pin_from_manifest(
    manifest_path: str | Path,
    *,
    contract_anchor: Path,
    policy: SplitPolicyPin,
) -> SplitPin:
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = validate_split_manifest_snapshot(manifest_file, manifest_file.read_bytes())
    return SplitPin(
        split_manifest_path=_portable_relative_path(manifest_file, contract_anchor),
        split_manifest_sha256=sha256_file(manifest_file),
        train_split_sha256=manifest.splits["train"].sha256,
        validation_split_sha256=manifest.splits["validation"].sha256,
        held_out_split_sha256=manifest.splits["held_out"].sha256,
        train_record_count=manifest.splits["train"].record_count,
        validation_record_count=manifest.splits["validation"].record_count,
        held_out_record_count=manifest.splits["held_out"].record_count,
        split_seed=policy.split_seed,
        split_salt=policy.split_salt,
        train_weight=policy.train_weight,
        validation_weight=policy.validation_weight,
        held_out_weight=policy.held_out_weight,
    )


def _validate_split_pin(contract_path: Path, split: SplitPin) -> SplitManifest:
    manifest_path = (contract_path.parent / split.split_manifest_path).resolve(strict=True)
    manifest_snapshot = manifest_path.read_bytes()
    if sha256_bytes(manifest_snapshot) != split.split_manifest_sha256:
        raise ValueError("experiment contract split-manifest SHA-256 mismatch")
    manifest = validate_split_manifest_snapshot(manifest_path, manifest_snapshot)
    for name in _SPLIT_NAMES:
        digest = getattr(split, f"{name}_split_sha256")
        if manifest.splits[name].sha256 != digest:
            raise ValueError(f"experiment contract {name} split SHA-256 mismatch")
    return manifest


def _validate_held_out_counts(split: SplitPin, manifest: SplitManifest) -> None:
    for name in _SPLIT_NAMES:
        expected = getattr(split, f"{name}_record_count")
        if manifest.splits[name].record_count != expected:
            raise ValueError(f"experiment contract {name} record count mismatch")


def _portable_relative_path(path: Path, anchor: Path) -> str:
    return os.path.relpath(path, anchor).replace("\\", "/")


def _write_contract_payload(path: Path, payload: bytes) -> None:
    write_fsynced_bytes(path, payload)


def _describe_validation_error(version: str, exc: ValidationError) -> str:
    lines = [f"invalid {version} experiment contract:"]
    for error in exc.errors(include_url=False):
        field = _dotted_location(error["loc"])
        if error["type"] == "missing":
            lines.append(f"  {field}: missing required field")
        else:
            lines.append(f"  {field}: {error['msg']}")
    return "\n".join(lines)


def _dotted_location(location: tuple[int | str, ...]) -> str:
    dotted = ""
    for part in location:
        if isinstance(part, int):
            dotted += f"[{part}]"
        else:
            dotted += f".{part}" if dotted else part
    return dotted or "<contract>"
