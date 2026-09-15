"""Offline verification of a sealed reproducibility bundle.

The verifier inspects manifests, inventories, and hashes only. It does not
open safetensors payloads, unpickle weights, execute model code, or contact
the network.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from .._strict_json import decode_json_object
from ..artifacts.safetensors_io import UNSAFE_WEIGHT_PATTERNS
from ..config import ExperimentConfig
from ..eval._artifact_schema import (
    ArtifactIndex,
    ArtifactProducerProvenance,
    _portable_artifact_path,
)
from ..eval._descriptor_bundle import (
    EntryIdentity,
    open_bundle,
    read_relative_file,
    require_descriptor_support,
    scan_bundle,
)
from ..eval.contract import (
    PairedEvaluationContract,
    held_out_task_ids,
    logical_task_set_sha256,
)
from ..path_safety import resolve_existing_path
from ..split_schema import SplitManifest, sha256_bytes
from ..split_validation import validate_split_manifest_snapshot
from .report import BundleFailure, BundleVerificationReport, sort_failures
from .schema import (
    BUNDLE_INVENTORY_NAME,
    BUNDLE_SCHEMA_VERSION,
    ReproducibilityBundle,
)

_UNKNOWN_BUNDLE_SCHEMA = "unknown reproducibility-bundle schema version"
_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"
_MERGED_CONFIG = "config.json"
_MERGED_WEIGHTS = "model.safetensors"
_MERGED_WEIGHTS_INDEX = "model.safetensors.index.json"


def verify_reproducibility_bundle(bundle_dir: str | Path) -> BundleVerificationReport:
    """Return a deterministic pass/fail verdict for one local bundle."""

    root = resolve_existing_path(str(bundle_dir), must_be_dir=True)
    try:
        require_descriptor_support()
        identities = scan_bundle(root)
    except ValueError as exc:
        return _fatal_report(root, _classify_scan_error(exc))
    return _verify_scanned_bundle(root, identities)


def _verify_scanned_bundle(
    root: Path,
    identities: dict[PurePosixPath, EntryIdentity],
) -> BundleVerificationReport:
    loaded = _load_inventory(root)
    if isinstance(loaded, BundleFailure):
        return _fatal_report(root, loaded)
    document, _payload = loaded
    membership = _membership_failures(identities, document)
    hashes = _hash_failures(root, document)
    failures = [
        *membership,
        *hashes,
        *_unsafe_weight_failures(document),
    ]
    blocking = {item.code for item in membership + hashes} & {
        "missing_file",
        "modified_file",
        "extra_file",
    }
    if not blocking:
        failures.extend(_component_failures(root, document))
    return _report(root, document.schema_version, failures)


def _load_inventory(
    root: Path,
) -> tuple[ReproducibilityBundle, bytes] | BundleFailure:
    relative = PurePosixPath(BUNDLE_INVENTORY_NAME)
    try:
        payload, _digest = _read_relative(root, relative)
        value = decode_json_object(
            payload,
            str(root / BUNDLE_INVENTORY_NAME),
            object_label="reproducibility bundle",
        )
    except ValueError as exc:
        return _json_failure(exc, BUNDLE_INVENTORY_NAME)
    version = value.get("schema_version")
    if version != BUNDLE_SCHEMA_VERSION:
        return BundleFailure(
            code="unknown_schema_version",
            path=BUNDLE_INVENTORY_NAME,
            message=f"{_UNKNOWN_BUNDLE_SCHEMA}: {version!r}",
        )
    try:
        return ReproducibilityBundle.model_validate(value), payload
    except ValidationError as exc:
        return _schema_failure(exc, BUNDLE_INVENTORY_NAME)


def _membership_failures(
    identities: dict[PurePosixPath, EntryIdentity],
    document: ReproducibilityBundle,
) -> list[BundleFailure]:
    expected = {PurePosixPath(entry.path) for entry in document.files}
    actual = {
        path
        for path, identity in identities.items()
        if identity.kind == "file" and path.as_posix() != BUNDLE_INVENTORY_NAME
    }
    failures = [
        BundleFailure(
            code="missing_file",
            path=str(path),
            message="listed file is missing from the bundle",
        )
        for path in sorted(expected - actual, key=str)
    ]
    failures.extend(
        BundleFailure(
            code="extra_file",
            path=str(path),
            message="file is not listed in the bundle inventory",
        )
        for path in sorted(actual - expected, key=str)
    )
    return failures


def _hash_failures(root: Path, document: ReproducibilityBundle) -> list[BundleFailure]:
    failures: list[BundleFailure] = []
    descriptor = open_bundle(root)
    try:
        for entry in document.files:
            failures.extend(_hash_entry(descriptor, entry.path, entry.size_bytes, entry.sha256))
    finally:
        os.close(descriptor)
    return failures


def _hash_entry(
    descriptor: int,
    relative: str,
    expected_size: int,
    expected_digest: str,
) -> list[BundleFailure]:
    try:
        payload, digest = read_relative_file(descriptor, PurePosixPath(relative))
    except ValueError:
        return []
    failures: list[BundleFailure] = []
    if len(payload) != expected_size:
        failures.append(
            BundleFailure(
                code="modified_file",
                path=relative,
                message=(f"size mismatch: expected {expected_size} bytes, found {len(payload)}"),
            )
        )
    if digest != expected_digest:
        failures.append(
            BundleFailure(
                code="modified_file",
                path=relative,
                message=f"SHA-256 mismatch: expected {expected_digest}, found {digest}",
            )
        )
    return failures


def _unsafe_weight_failures(document: ReproducibilityBundle) -> list[BundleFailure]:
    failures: list[BundleFailure] = []
    for entry in document.files:
        name = PurePosixPath(entry.path).name
        if _is_unsafe_weight_name(name):
            failures.append(
                BundleFailure(
                    code="unsafe_path",
                    path=entry.path,
                    message="bundle contains an unsafe serialized weight file",
                )
            )
    return failures


def _is_unsafe_weight_name(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in UNSAFE_WEIGHT_PATTERNS)


def _component_failures(root: Path, document: ReproducibilityBundle) -> list[BundleFailure]:
    split = _load_split(root, document)
    failures: list[BundleFailure] = []
    if isinstance(split, BundleFailure):
        failures.append(split)
        split_manifest = None
        split_digest = None
    else:
        split_manifest, split_digest = split
    failures.extend(_run_and_config_failures(root, document))
    failures.extend(_artifact_index_failures(root, document, split_digest, split_manifest))
    failures.extend(_evaluation_failures(root, document, split_manifest, split_digest))
    return failures


def _load_split(
    root: Path,
    document: ReproducibilityBundle,
) -> tuple[SplitManifest, str] | BundleFailure:
    relative = document.split_manifest_path
    try:
        payload, _digest = _read_relative(root, PurePosixPath(relative))
        value = decode_json_object(payload, str(root / relative), object_label="split manifest")
    except ValueError as exc:
        return _json_failure(exc, relative)
    version = value.get("schema_version")
    if version != "agoge.split-manifest.v1":
        return BundleFailure(
            code="unknown_schema_version",
            path=relative,
            message=f"unknown split-manifest schema version: {version!r}",
        )
    try:
        manifest = validate_split_manifest_snapshot(root / relative, payload)
    except ValueError as exc:
        return BundleFailure(code="split_identity", path=relative, message=str(exc))
    return manifest, sha256_bytes(payload)


def _run_and_config_failures(
    root: Path,
    document: ReproducibilityBundle,
) -> list[BundleFailure]:
    manifest = _load_json_object(root, document.run_manifest_path, "run manifest")
    locked = _load_json_object(root, document.locked_config_path, "locked config")
    failures: list[BundleFailure] = []
    if isinstance(manifest, BundleFailure):
        failures.append(manifest)
        manifest_payload = None
    else:
        manifest_payload = manifest
        failures.extend(_run_manifest_field_failures(document.run_manifest_path, manifest_payload))
    if isinstance(locked, BundleFailure):
        failures.append(locked)
        return failures
    try:
        locked_config = ExperimentConfig.model_validate(locked)
    except ValidationError as exc:
        failures.append(_schema_failure(exc, document.locked_config_path))
        return failures
    if manifest_payload is not None:
        failures.extend(
            _locked_config_match_failures(
                document,
                locked_config,
                manifest_payload,
            )
        )
    return failures


def _run_manifest_field_failures(path: str, payload: dict[str, Any]) -> list[BundleFailure]:
    failures: list[BundleFailure] = []
    for field in ("timestamp", "git", "config"):
        if field not in payload:
            failures.append(
                BundleFailure(
                    code="run_manifest",
                    path=path,
                    message=f"run manifest is missing required field {field!r}",
                )
            )
    if "config" in payload and not isinstance(payload["config"], dict):
        failures.append(
            BundleFailure(
                code="run_manifest",
                path=path,
                message="run manifest config must be a JSON object",
            )
        )
    return failures


def _locked_config_match_failures(
    document: ReproducibilityBundle,
    locked_config: ExperimentConfig,
    manifest_payload: dict[str, Any],
) -> list[BundleFailure]:
    raw_config = manifest_payload.get("config")
    if not isinstance(raw_config, dict):
        return []
    try:
        manifest_config = ExperimentConfig.model_validate(raw_config)
    except ValidationError as exc:
        return [_schema_failure(exc, document.run_manifest_path)]
    if locked_config.model_dump(mode="json") != manifest_config.model_dump(mode="json"):
        return [
            BundleFailure(
                code="locked_config",
                path=document.locked_config_path,
                message="locked config does not match the run manifest config",
            )
        ]
    return []


def _artifact_index_failures(
    root: Path,
    document: ReproducibilityBundle,
    split_digest: str | None,
    split_manifest: SplitManifest | None,
) -> list[BundleFailure]:
    failures = _one_artifact_index_failures(
        root,
        document.adapter_artifact_index_path,
        kind="peft_adapter",
        split_digest=split_digest,
        split_manifest=split_manifest,
        locked_path=document.locked_config_path,
    )
    if document.merged_artifact_index_path is not None:
        failures.extend(
            _one_artifact_index_failures(
                root,
                document.merged_artifact_index_path,
                kind="merged_model",
                split_digest=split_digest,
                split_manifest=split_manifest,
                locked_path=document.locked_config_path,
            )
        )
    return failures


def _one_artifact_index_failures(
    root: Path,
    relative: str,
    *,
    kind: str,
    split_digest: str | None,
    split_manifest: SplitManifest | None,
    locked_path: str,
) -> list[BundleFailure]:
    loaded = _load_artifact_index(root, relative)
    if isinstance(loaded, BundleFailure):
        return [loaded]
    index, identities = loaded
    failures = _artifact_membership_failures(relative, index, identities)
    failures.extend(_artifact_layout_failures(relative, kind, index))
    failures.extend(
        _artifact_provenance_failures(
            relative,
            index,
            split_digest,
            split_manifest,
            locked_path,
            root,
        )
    )
    return failures


def _load_artifact_index(
    root: Path,
    relative: str,
) -> tuple[ArtifactIndex, dict[PurePosixPath, EntryIdentity]] | BundleFailure:
    try:
        payload, _digest = _read_relative(root, PurePosixPath(relative))
        value = decode_json_object(payload, str(root / relative), object_label="artifact index")
        index = ArtifactIndex.model_validate(value)
    except ValueError as exc:
        return _index_parse_failure(relative, exc)
    artifact_root = (root / relative).parent
    try:
        identities = scan_bundle(artifact_root)
    except ValueError as exc:
        return _classify_scan_error(exc, relative)
    return index, identities


def _artifact_membership_failures(
    index_path: str,
    index: ArtifactIndex,
    identities: dict[PurePosixPath, EntryIdentity],
) -> list[BundleFailure]:
    try:
        expected = {_portable_artifact_path(entry.file) for entry in index.artifacts}
    except ValueError as exc:
        return [BundleFailure(code="path_escaping", path=index_path, message=str(exc))]
    actual = {
        path
        for path, identity in identities.items()
        if identity.kind == "file" and path.as_posix() != "artifact_index.json"
    }
    failures = [
        BundleFailure(
            code="missing_file",
            path=f"{PurePosixPath(index_path).parent / path}",
            message="artifact index references a missing file",
        )
        for path in sorted(expected - actual, key=str)
    ]
    failures.extend(
        BundleFailure(
            code="extra_file",
            path=f"{PurePosixPath(index_path).parent / path}",
            message="artifact directory contains a file omitted from artifact index",
        )
        for path in sorted(actual - expected, key=str)
    )
    return failures


def _artifact_layout_failures(
    index_path: str,
    kind: str,
    index: ArtifactIndex,
) -> list[BundleFailure]:
    names = {PurePosixPath(entry.file).as_posix() for entry in index.artifacts}
    if kind == "peft_adapter":
        missing = sorted({_ADAPTER_CONFIG, _ADAPTER_WEIGHTS} - names)
        if missing:
            return [
                BundleFailure(
                    code="artifact_index",
                    path=index_path,
                    message=f"peft_adapter artifact is missing required files: {missing}",
                )
            ]
        return []
    if _MERGED_CONFIG not in names:
        return [
            BundleFailure(
                code="artifact_index",
                path=index_path,
                message="merged_model artifact is missing config.json",
            )
        ]
    has_single = _MERGED_WEIGHTS in names
    has_sharded = _MERGED_WEIGHTS_INDEX in names
    if has_single == has_sharded:
        return [
            BundleFailure(
                code="artifact_index",
                path=index_path,
                message=(
                    "merged_model artifact must contain exactly one of "
                    "model.safetensors or model.safetensors.index.json"
                ),
            )
        ]
    return []


def _artifact_provenance_failures(
    index_path: str,
    index: ArtifactIndex,
    split_digest: str | None,
    split_manifest: SplitManifest | None,
    locked_path: str,
    root: Path,
) -> list[BundleFailure]:
    provenance = index.producer_provenance
    if provenance is None:
        return [
            BundleFailure(
                code="artifact_index",
                path=index_path,
                message="artifact index requires producer_provenance",
            )
        ]
    failures: list[BundleFailure] = []
    if split_digest is not None and provenance.training_split_manifest_sha256 != split_digest:
        failures.append(
            BundleFailure(
                code="split_identity",
                path=index_path,
                message="artifact producer provenance does not match the bundled split manifest",
            )
        )
    if split_manifest is not None:
        train_digest = split_manifest.splits["train"].sha256
        if provenance.training_split_sha256 != train_digest:
            failures.append(
                BundleFailure(
                    code="split_identity",
                    path=index_path,
                    message="artifact producer provenance does not match the frozen train split",
                )
            )
    locked = _load_json_object(root, locked_path, "locked config")
    if isinstance(locked, dict):
        failures.extend(_provenance_config_failures(index_path, provenance, locked))
    return failures


def _provenance_config_failures(
    index_path: str,
    provenance: ArtifactProducerProvenance,
    locked: dict[str, Any],
) -> list[BundleFailure]:
    try:
        config = ExperimentConfig.model_validate(locked)
    except ValidationError:
        return []
    failures: list[BundleFailure] = []
    if config.model_id != provenance.base_model_name_or_path:
        failures.append(
            BundleFailure(
                code="locked_config",
                path=index_path,
                message="artifact base model does not match the locked config",
            )
        )
    if config.revision is not None and config.revision != provenance.revision:
        failures.append(
            BundleFailure(
                code="locked_config",
                path=index_path,
                message="artifact revision does not match the locked config",
            )
        )
    return failures


def _evaluation_failures(
    root: Path,
    document: ReproducibilityBundle,
    split_manifest: SplitManifest | None,
    split_digest: str | None,
) -> list[BundleFailure]:
    relative = document.evaluation_contract_path
    loaded = _load_evaluation_payload(root, relative)
    if isinstance(loaded, BundleFailure):
        return [loaded]
    payload = loaded
    arm_failure = _missing_arm_failure(relative, payload)
    if arm_failure is not None:
        return [arm_failure]
    try:
        contract = PairedEvaluationContract.model_validate(payload)
    except ValidationError as exc:
        return [_schema_failure(exc, relative)]
    return _evaluation_identity_failures(
        root,
        document,
        contract,
        split_manifest,
        split_digest,
    )


def _load_evaluation_payload(root: Path, relative: str) -> dict[str, Any] | BundleFailure:
    try:
        raw, _digest = _read_relative(root, PurePosixPath(relative))
        value = decode_json_object(
            raw,
            str(root / relative),
            object_label="evaluation contract",
        )
    except ValueError as exc:
        return _json_failure(exc, relative)
    version = value.get("schema_version")
    if version != "agoge.evaluation-contract.v2":
        return BundleFailure(
            code="unknown_schema_version",
            path=relative,
            message=f"unknown evaluation-contract schema version: {version!r}",
        )
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


def _evaluation_identity_failures(
    root: Path,
    document: ReproducibilityBundle,
    contract: PairedEvaluationContract,
    split_manifest: SplitManifest | None,
    split_digest: str | None,
) -> list[BundleFailure]:
    failures: list[BundleFailure] = []
    contract_parent = (root / document.evaluation_contract_path).parent
    try:
        split_path = _confine_reference(
            contract_parent,
            contract.split_manifest_path,
            root,
            document.evaluation_contract_path,
        )
    except (OSError, ValueError) as exc:
        return [
            BundleFailure(
                code="path_escaping",
                path=document.evaluation_contract_path,
                message=str(exc),
            )
        ]
    expected_split = (root / document.split_manifest_path).resolve(strict=True)
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
    failures.extend(_evaluation_artifact_failures(root, document, contract, contract_parent))
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


def _evaluation_artifact_failures(
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
    try:
        index_path = _confine_reference(
            contract_parent,
            artifact.artifact_index_path,
            root,
            document.evaluation_contract_path,
        )
    except (OSError, ValueError) as exc:
        return [
            BundleFailure(
                code="path_escaping",
                path=document.evaluation_contract_path,
                message=str(exc),
            )
        ]
    expected = (root / document.adapter_artifact_index_path).resolve(strict=True)
    merged = document.merged_artifact_index_path
    allowed = {expected}
    if merged is not None:
        allowed.add((root / merged).resolve(strict=True))
    if index_path not in allowed:
        return [
            BundleFailure(
                code="artifact_index",
                path=document.evaluation_contract_path,
                message="evaluation contract does not reference a bundled artifact index",
            )
        ]
    try:
        _, digest = _read_relative(root, PurePosixPath(index_path.relative_to(root).as_posix()))
    except ValueError as exc:
        return [
            BundleFailure(
                code="missing_file",
                path=artifact.artifact_index_path,
                message=str(exc),
            )
        ]
    if digest != artifact.artifact_index_sha256:
        return [
            BundleFailure(
                code="modified_file",
                path=artifact.artifact_index_path,
                message=(
                    "evaluation contract artifact-index SHA-256 mismatch: "
                    f"expected {artifact.artifact_index_sha256}, found {digest}"
                ),
            )
        ]
    return []


def _confine_reference(
    anchor: Path,
    relative: str,
    bundle_root: Path,
    contract_path: str,
) -> Path:
    portable = _portable_artifact_path(relative)
    resolved = (anchor / Path(*portable.parts)).resolve(strict=True)
    root = bundle_root.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{contract_path} reference escapes the bundle: {relative}")
    return resolved


def _load_json_object(root: Path, relative: str, label: str) -> dict[str, Any] | BundleFailure:
    try:
        payload, _digest = _read_relative(root, PurePosixPath(relative))
        return decode_json_object(payload, str(root / relative), object_label=label)
    except ValueError as exc:
        return _json_failure(exc, relative)


def _read_relative(root: Path, relative: PurePosixPath) -> tuple[bytes, str]:
    descriptor = open_bundle(root)
    try:
        return read_relative_file(descriptor, relative)
    finally:
        os.close(descriptor)


def _json_failure(exc: ValueError, path: str) -> BundleFailure:
    return BundleFailure(code="invalid_json", path=path, message=str(exc))


def _index_parse_failure(path: str, exc: ValueError) -> BundleFailure:
    message = str(exc)
    lowered = message.lower()
    if "duplicate" in lowered:
        return BundleFailure(code="duplicate_path", path=path, message=message)
    if "invalid json" in lowered:
        return BundleFailure(code="invalid_json", path=path, message=message)
    if isinstance(exc, ValidationError):
        return _schema_failure(exc, path)
    return BundleFailure(code="artifact_index", path=path, message=message)


def _schema_failure(exc: ValidationError, path: str) -> BundleFailure:
    message = str(exc)
    lowered = message.lower()
    if "duplicate file paths" in lowered:
        return BundleFailure(code="duplicate_path", path=path, message=message)
    if "must stay relative" in lowered or "canonical posix" in lowered:
        return BundleFailure(code="path_escaping", path=path, message=message)
    if "schema version" in lowered:
        return BundleFailure(code="unknown_schema_version", path=path, message=message)
    return BundleFailure(code="invalid_json", path=path, message=message)


def _classify_scan_error(exc: ValueError, path: str | None = None) -> BundleFailure:
    message = str(exc)
    if "symlink" in message.lower():
        return BundleFailure(code="unsafe_symlink", path=path, message=message)
    if "escape" in message.lower() or ".." in message:
        return BundleFailure(code="path_escaping", path=path, message=message)
    return BundleFailure(code="unsafe_path", path=path, message=message)


def _fatal_report(root: Path, failure: BundleFailure) -> BundleVerificationReport:
    return _report(root, None, [failure])


def _report(
    root: Path,
    bundle_schema_version: str | None,
    failures: list[BundleFailure],
) -> BundleVerificationReport:
    ordered = sort_failures(failures)
    return BundleVerificationReport(
        verdict="pass" if not ordered else "fail",
        bundle_path=str(root),
        bundle_schema_version=bundle_schema_version,
        failures=ordered,
    )
