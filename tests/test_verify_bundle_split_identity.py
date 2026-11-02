"""verify-bundle re-derives split member identity, leakage, and ownership offline."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from agoge_forger.eval.contract import held_out_task_ids, logical_task_set_sha256
from agoge_forger.release.verify import verify_reproducibility_bundle
from agoge_forger.split_contract import (
    SplitManifest,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from tests.reproducibility_bundle_cases import (
    EVAL_CONTRACT_PATH,
    SPLIT_MANIFEST_PATH,
    reseal_bundle,
    write_valid_bundle,
)
from tests.test_verify_bundle import _chmod_tree, _rewrite_adapter_index

Lines = dict[str, list[bytes]]


def _rewrite_split(bundle: Path, mutate: Callable[[dict, Lines], None]) -> None:
    """Apply ``mutate`` to the manifest and split lines, then re-chain every digest."""

    manifest_path = bundle / SPLIT_MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines: Lines = {
        name: (bundle / artifact["path"]).read_bytes().splitlines(keepends=True)
        for name, artifact in manifest["splits"].items()
    }
    mutate(manifest, lines)
    for name, artifact in manifest["splits"].items():
        split_path = bundle / artifact["path"]
        split_path.write_bytes(b"".join(lines[name]))
        artifact["sha256"] = sha256_file(split_path)
        artifact["record_count"] = len(artifact["members"])
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    task_ids = list(held_out_task_ids(SplitManifest.model_validate(manifest)))
    task_digest = logical_task_set_sha256(task_ids)
    contract_path = bundle / EVAL_CONTRACT_PATH
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        split_manifest_sha256=sha256_file(manifest_path),
        held_out_split_sha256=manifest["splits"]["held_out"]["sha256"],
        logical_task_ids=task_ids,
        logical_task_set_sha256=task_digest,
    )
    contract["base"]["logical_task_set_sha256"] = task_digest
    contract["sft"]["logical_task_set_sha256"] = task_digest
    contract_path.write_bytes(canonical_json_bytes(contract) + b"\n")
    _rewrite_adapter_index(bundle)
    reseal_bundle(bundle)


def _split_identity_failures(report) -> list[tuple[str | None, str]]:
    return [
        (failure.path, failure.message)
        for failure in report.failures
        if failure.code == "split_identity"
    ]


def test_rechained_valid_bundle_still_passes(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    _rewrite_split(bundle, lambda _manifest, _lines: None)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "pass", report.failures


def test_relabeled_split_members_fail(tmp_path):
    def swap_labels(manifest: dict, _lines: Lines) -> None:
        train = manifest["splits"]["train"]["members"][0]
        held_out = manifest["splits"]["held_out"]["members"][0]
        train["canonical_id"], held_out["canonical_id"] = (
            held_out["canonical_id"],
            train["canonical_id"],
        )

    bundle = write_valid_bundle(tmp_path)
    _rewrite_split(bundle, swap_labels)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    failures = _split_identity_failures(report)
    assert (
        "splits/held_out.jsonl",
        "held_out split records do not match the declared manifest members",
    ) in failures
    assert any(path == "splits/train.jsonl" and "member" in message for path, message in failures)


def test_held_out_record_moved_into_train_fails(tmp_path):
    def move_held_out_record(manifest: dict, lines: Lines) -> None:
        held_out = manifest["splits"]["held_out"]
        train = manifest["splits"]["train"]
        train["members"].append(held_out["members"].pop(0))
        lines["train"].append(lines["held_out"].pop(0))

    bundle = write_valid_bundle(tmp_path)
    _rewrite_split(bundle, move_held_out_record)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    messages = [
        message for path, message in _split_identity_failures(report) if path == SPLIT_MANIFEST_PATH
    ]
    assert any("leakage audit failed" in message for message in messages)
    assert any("split ownership differs" in message for message in messages)


def test_member_identity_drift_with_matching_raw_digest_fails(tmp_path):
    def drift_lineage(manifest: dict, _lines: Lines) -> None:
        member = manifest["splits"]["validation"]["members"][0]
        member["lineage_id"] = "lineage-forged"

    bundle = write_valid_bundle(tmp_path)
    _rewrite_split(bundle, drift_lineage)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert any(
        path == "splits/validation.jsonl" and "member" in message
        for path, message in _split_identity_failures(report)
    )


def test_non_strict_split_line_is_a_structured_failure(tmp_path):
    def duplicate_key_line(manifest: dict, lines: Lines) -> None:
        row = json.loads(lines["train"][0])
        canonical = canonical_json_bytes(row)
        # Same decoded object under lenient JSON, but strict decoding rejects duplicate keys.
        line = b'{"text":"shadowed",' + canonical[1:] + b"\n"
        lines["train"][0] = line
        member = manifest["splits"]["train"]["members"][0]
        member["raw_line_sha256"] = sha256_bytes(line)
        member["materialized_line_sha256"] = sha256_bytes(line)

    bundle = write_valid_bundle(tmp_path)
    _rewrite_split(bundle, duplicate_key_line)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert any(
        path == "splits/train.jsonl" and "member" in message
        for path, message in _split_identity_failures(report)
    )


def test_bundle_frozen_from_non_canonical_source_passes(tmp_path):
    """raw_line_sha256 binds the original source line, which the bundle does not carry."""

    def legacy_raw_digests(manifest: dict, lines: Lines) -> None:
        for name, artifact in manifest["splits"].items():
            for line, member in zip(lines[name], artifact["members"], strict=True):
                source_line = json.dumps(json.loads(line)) + "\n"
                member["raw_line_sha256"] = sha256_bytes(source_line.encode("utf-8"))
                assert member["raw_line_sha256"] != member["materialized_line_sha256"]

    bundle = write_valid_bundle(tmp_path)
    _rewrite_split(bundle, legacy_raw_digests)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "pass", report.failures


def test_identity_checks_stay_read_only(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    before = sorted(path.relative_to(bundle) for path in bundle.rglob("*"))
    _chmod_tree(bundle, dir_mode=0o555, file_mode=0o444)
    try:
        report = verify_reproducibility_bundle(bundle)
    finally:
        _chmod_tree(bundle, dir_mode=0o755, file_mode=0o644)
    assert report.verdict == "pass", report.failures
    assert sorted(path.relative_to(bundle) for path in bundle.rglob("*")) == before
