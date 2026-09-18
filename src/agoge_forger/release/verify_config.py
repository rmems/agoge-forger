"""Locked ExperimentConfig and run-manifest identity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config import ExperimentConfig
from .report import BundleFailure
from .schema import ReproducibilityBundle
from .verify_errors import schema_failure
from .verify_io import load_object


def run_and_config_failures(root: Path, document: ReproducibilityBundle) -> list[BundleFailure]:
    manifest = load_object(root, document.run_manifest_path, "run manifest")
    locked = load_object(root, document.locked_config_path, "locked config")
    failures: list[BundleFailure] = []
    manifest_payload: dict[str, Any] | None
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
        failures.append(schema_failure(exc, document.locked_config_path))
        return failures
    failures.extend(_policy_failures(document.locked_config_path, locked_config))
    if manifest_payload is not None:
        failures.extend(_locked_config_match_failures(document, locked_config, manifest_payload))
    return failures


def _run_manifest_field_failures(path: str, payload: dict[str, Any]) -> list[BundleFailure]:
    failures = [
        BundleFailure(
            code="run_manifest",
            path=path,
            message=f"run manifest is missing required field {field!r}",
        )
        for field in ("timestamp", "git", "config")
        if field not in payload
    ]
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
        return [schema_failure(exc, document.run_manifest_path)]
    failures = _policy_failures(document.run_manifest_path, manifest_config)
    if locked_config.model_dump(mode="json") != manifest_config.model_dump(mode="json"):
        failures.append(
            BundleFailure(
                code="locked_config",
                path=document.locked_config_path,
                message="locked config does not match the run manifest config",
            )
        )
    return failures


def _policy_failures(path: str, config: ExperimentConfig) -> list[BundleFailure]:
    failures: list[BundleFailure] = []
    if config.trust_remote_code:
        failures.append(
            BundleFailure(
                code="locked_config",
                path=path,
                message="bundle config enables trust_remote_code",
            )
        )
    if config.revision is None:
        failures.append(
            BundleFailure(
                code="locked_config",
                path=path,
                message="bundle config must pin an immutable model revision",
            )
        )
    return failures
