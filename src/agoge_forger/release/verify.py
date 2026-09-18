"""Offline verification of a sealed reproducibility bundle.

The verifier inspects manifests, inventories, and hashes only. It does not
open safetensors payloads, unpickle weights, execute model code, or contact
the network.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from ..eval._descriptor_bundle import EntryIdentity, require_descriptor_support, scan_bundle
from ..path_safety import resolve_existing_path
from .report import BundleFailure, BundleVerificationReport
from .schema import ReproducibilityBundle
from .verify_artifacts import artifact_index_failures
from .verify_config import run_and_config_failures
from .verify_errors import classify_scan_error, fatal_report, make_report
from .verify_evaluation import evaluation_failures
from .verify_inventory import (
    hash_failures,
    identity_drift_failures,
    load_inventory,
    membership_failures,
    unsafe_weight_failures,
)
from .verify_split import load_split, split_artifact_failures


def verify_reproducibility_bundle(bundle_dir: str | Path) -> BundleVerificationReport:
    """Return a deterministic pass/fail verdict for one local bundle."""

    root = resolve_existing_path(str(bundle_dir), must_be_dir=True)
    try:
        require_descriptor_support()
        identities = scan_bundle(root)
    except ValueError as exc:
        return fatal_report(root, classify_scan_error(exc))
    return _verify_scanned_bundle(root, identities)


def _verify_scanned_bundle(
    root: Path, identities: dict[PurePosixPath, EntryIdentity]
) -> BundleVerificationReport:
    loaded = load_inventory(root)
    if not isinstance(loaded, tuple):
        return fatal_report(root, loaded)
    document, _payload = loaded
    membership = membership_failures(identities, document)
    hashes = hash_failures(root, document)
    failures = [*membership, *hashes, *unsafe_weight_failures(document)]
    blocking = {item.code for item in membership + hashes} & {
        "missing_file",
        "modified_file",
        "extra_file",
    }
    if not blocking:
        failures.extend(_component_failures(root, document))
        try:
            after = scan_bundle(root)
        except ValueError as exc:
            failures.append(classify_scan_error(exc))
        else:
            failures.extend(identity_drift_failures(root, identities, after))
    return make_report(root, document.schema_version, failures)


def _component_failures(root: Path, document: ReproducibilityBundle) -> list[BundleFailure]:
    split = load_split(root, document)
    failures = []
    if not isinstance(split, tuple):
        failures.append(split)
        split_manifest = None
        split_digest = None
    else:
        split_manifest, split_digest = split
        failures.extend(split_artifact_failures(document, split_manifest))
    failures.extend(run_and_config_failures(root, document))
    failures.extend(artifact_index_failures(root, document, split_digest, split_manifest))
    failures.extend(evaluation_failures(root, document, split_manifest, split_digest))
    return failures
