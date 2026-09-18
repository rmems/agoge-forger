"""Paired evaluation-contract identity checks."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from .._strict_json import decode_json_object
from ..eval._artifact_schema import ArtifactIndex
from ..eval.contract import (
    PairedEvaluationContract,
    held_out_task_ids,
    logical_task_set_sha256,
)
from ..split_schema import SplitManifest
from .report import BundleFailure
from .schema import ReproducibilityBundle
from .verify_errors import json_failure, schema_failure, unknown_schema
from .verify_io import classify_io_error, confine_reference, read_relative, resolve_existing


def evaluation_failures(
    root: Path,
    document: ReproducibilityBundle,
    split_manifest: SplitManifest | None,
    split_digest: str | None,
) -> list[BundleFailure]:
    relative = document.evaluation_contract_path
    loaded = _load_evaluation_payload(root, relative)
    if isinstance(loaded, BundleFailure):
        return [loaded]
    arm_failure = _missing_arm_failure(relative, loaded)
    if arm_failure is not None:
        return [arm_failure]
    try:
        contract = PairedEvaluationContract.model_validate(loaded)
    except ValidationError as exc:
        return [schema_failure(exc, relative)]
    return _identity_failures(root, document, contract, split_manifest, split_digest)


def _load_evaluation_payload(root: Path, relative: str) -> dict[str, Any] | BundleFailure:
    try:
        raw, _digest = read_relative(root, PurePosixPath(relative))
        value = decode_json_object(raw, str(root / relative), object_label="evaluation contract")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            return json_failure(exc, relative)
        return classify_io_error(exc, relative)
    version = value.get("schema_version")
    if version != "agoge.evaluation-contract.v2":
        return unknown_schema(relative, f"unknown evaluation-contract schema version: {version!r}")
    return value


def _missing_arm_failure(path: str, payload: dict[str, Any]) -> BundleFailure | None:
    missing = [
        name
        for name in ("base", "sft")
        if name not in payload or payload[name] is None or payload[name] == {}
    ]
    if not missing:
        return None
    return BundleFailure(
        code="missing_evaluation_arm",
        path=path,
        message=f"evaluation contract is missing required arm(s): {missing}",
    )


def _identity_failures(
    root: Path,
    document: ReproducibilityBundle,
    contract: PairedEvaluationContract,
    split_manifest: SplitManifest | None,
    split_digest: str | None,
) -> list[BundleFailure]:
    contract_parent = (root / document.evaluation_contract_path).parent
    split_path = confine_reference(
        contract_parent,
        contract.split_manifest_path,
        root,
        document.evaluation_contract_path,
    )
    if isinstance(split_path, BundleFailure):
        return [split_path]
    expected_split = resolve_existing(
        root, document.split_manifest_path, document.evaluation_contract_path
    )
    if isinstance(expected_split, BundleFailure):
        return [expected_split]
    failures: list[BundleFailure] = []
    if split_path != expected_split:
        failures.append(
            BundleFailure(
                code="split_identity",
                path=document.evaluation_contract_path,
                message="evaluation contract does not reference the bundled split manifest",
            )
        )
    if split_digest is not None and contract.split_manifest_sha256 != split_digest:
        failures.append(
            BundleFailure(
                code="split_identity",
                path=document.evaluation_contract_path,
                message="evaluation contract split-manifest SHA-256 mismatch",
            )
        )
    if split_manifest is not None:
        failures.extend(
            _held_out_failures(document.evaluation_contract_path, contract, split_manifest)
        )
    failures.extend(_artifact_failures(root, document, contract, contract_parent))
    return failures


def _held_out_failures(
    path: str,
    contract: PairedEvaluationContract,
    split_manifest: SplitManifest,
) -> list[BundleFailure]:
    failures: list[BundleFailure] = []
    if split_manifest.splits["held_out"].sha256 != contract.held_out_split_sha256:
        failures.append(
            BundleFailure(
                code="split_identity",
                path=path,
                message="evaluation contract held-out split SHA-256 mismatch",
            )
        )
    expected_ids = held_out_task_ids(split_manifest)
    if contract.logical_task_ids != expected_ids:
        failures.append(
            BundleFailure(
                code="split_identity",
                path=path,
                message="evaluation contract task IDs differ from the frozen held-out manifest",
            )
        )
    if logical_task_set_sha256(expected_ids) != contract.logical_task_set_sha256:
        failures.append(
            BundleFailure(
                code="split_identity",
                path=path,
                message="evaluation contract task-set digest differs from held-out membership",
            )
        )
    return failures


def _artifact_failures(
    root: Path,
    document: ReproducibilityBundle,
    contract: PairedEvaluationContract,
    contract_parent: Path,
) -> list[BundleFailure]:
    artifact = contract.sft.artifact
    if artifact is None:
        return [
            BundleFailure(
                code="missing_evaluation_arm",
                path=document.evaluation_contract_path,
                message="causal_sft arm requires a verified artifact-index reference",
            )
        ]
    index_path = confine_reference(
        contract_parent,
        artifact.artifact_index_path,
        root,
        document.evaluation_contract_path,
    )
    if isinstance(index_path, BundleFailure):
        return [index_path]
    expected = _expected_index_path(root, document, artifact.kind)
    if isinstance(expected, BundleFailure):
        return [expected]
    failures: list[BundleFailure] = []
    if index_path != expected:
        failures.append(
            BundleFailure(
                code="artifact_index",
                path=document.evaluation_contract_path,
                message="evaluation contract artifact kind does not match the selected index",
            )
        )
        return failures
    try:
        _, digest = read_relative(root, PurePosixPath(index_path.relative_to(root).as_posix()))
    except (OSError, ValueError) as exc:
        return [classify_io_error(exc, artifact.artifact_index_path)]
    if digest != artifact.artifact_index_sha256:
        failures.append(
            BundleFailure(
                code="modified_file",
                path=artifact.artifact_index_path,
                message=(
                    "evaluation contract artifact-index SHA-256 mismatch: "
                    f"expected {artifact.artifact_index_sha256}, found {digest}"
                ),
            )
        )
        return failures
    failures.extend(_model_identity_failures(root, document, contract, index_path))
    return failures


def _expected_index_path(
    root: Path,
    document: ReproducibilityBundle,
    kind: str,
) -> Path | BundleFailure:
    if kind == "peft_adapter":
        return resolve_existing(
            root, document.adapter_artifact_index_path, document.evaluation_contract_path
        )
    if kind != "merged_model" or document.merged_artifact_index_path is None:
        return BundleFailure(
            code="artifact_index",
            path=document.evaluation_contract_path,
            message="evaluation contract does not reference a bundled artifact index",
        )
    return resolve_existing(
        root, document.merged_artifact_index_path, document.evaluation_contract_path
    )


def _model_identity_failures(
    root: Path,
    document: ReproducibilityBundle,
    contract: PairedEvaluationContract,
    index_path: Path,
) -> list[BundleFailure]:
    try:
        payload, _digest = read_relative(
            root, PurePosixPath(index_path.relative_to(root).as_posix())
        )
        index = ArtifactIndex.model_validate(
            decode_json_object(payload, str(index_path), object_label="artifact index")
        )
    except (OSError, ValueError) as exc:
        return [classify_io_error(exc, document.evaluation_contract_path)]
    provenance = index.producer_provenance
    if provenance is None:
        return []
    failures: list[BundleFailure] = []
    if contract.sft.model_repository != provenance.base_model_name_or_path:
        failures.append(
            BundleFailure(
                code="artifact_index",
                path=document.evaluation_contract_path,
                message="evaluation SFT model repository does not match the selected artifact",
            )
        )
    if contract.sft.model_revision != provenance.revision:
        failures.append(
            BundleFailure(
                code="artifact_index",
                path=document.evaluation_contract_path,
                message="evaluation SFT model revision does not match the selected artifact",
            )
        )
    return failures
