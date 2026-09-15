"""Offline reproducibility-bundle verification."""

from __future__ import annotations

import json
from pathlib import Path

from agoge_forger.eval.contract import PairedEvaluationContract, load_evaluation_contract
from agoge_forger.release.report import format_verification_table
from agoge_forger.release.verify import verify_reproducibility_bundle
from agoge_forger.split_contract import canonical_json_bytes, validate_split_manifest
from tests.reproducibility_bundle_cases import (
    EVAL_CONTRACT_PATH,
    reseal_bundle,
    rewrite_inventory,
    write_valid_bundle,
)


def _codes(report) -> list[str]:
    return [failure.code for failure in report.failures]


def test_valid_miniature_bundle_passes(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "pass"
    assert report.failures == ()
    assert report.bundle_schema_version == "agoge.reproducibility-bundle.v1"


def test_same_bundle_is_deterministic(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    first = verify_reproducibility_bundle(bundle).as_dict()
    second = verify_reproducibility_bundle(bundle).as_dict()
    assert first == second
    table = format_verification_table(verify_reproducibility_bundle(bundle))
    assert table.splitlines()[0].startswith("verdict")


def test_mutated_digest_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    target = bundle / "adapter" / "adapter_model.safetensors"
    target.write_bytes(target.read_bytes() + b"\x00")
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "modified_file" in _codes(report)


def test_missing_evaluation_arm_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    contract_path = bundle / EVAL_CONTRACT_PATH
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    del payload["sft"]
    contract_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    reseal_bundle(bundle)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "missing_evaluation_arm" in _codes(report)


def test_extra_file_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    (bundle / "stray.txt").write_text("unexpected\n", encoding="utf-8")
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "extra_file" in _codes(report)
    assert any(failure.path == "stray.txt" for failure in report.failures)


def test_unsafe_symlink_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    (bundle / "escape").symlink_to(tmp_path / "outside")
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "unsafe_symlink" in _codes(report)


def test_duplicate_path_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)

    def duplicate(payload: dict[str, object]) -> None:
        files = payload["files"]
        assert isinstance(files, list)
        files.append(files[0])

    rewrite_inventory(bundle, duplicate)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "duplicate_path" in _codes(report)


def test_truncated_json_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    inventory = bundle / "reproducibility-bundle.json"
    inventory.write_bytes(inventory.read_bytes()[:40])
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "invalid_json" in _codes(report)


def test_unknown_schema_version_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)

    def bump_version(payload: dict[str, object]) -> None:
        payload["schema_version"] = "agoge.reproducibility-bundle.v0"

    rewrite_inventory(bundle, bump_version)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert _codes(report) == ["unknown_schema_version"]
    assert "v0" in report.failures[0].message


def test_path_escaping_inventory_entry_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)

    def escape(payload: dict[str, object]) -> None:
        files = payload["files"]
        assert isinstance(files, list)
        files.append(
            {
                "path": "../secret.txt",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        )

    rewrite_inventory(bundle, escape)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "path_escaping" in _codes(report)


def test_verify_does_not_deserialize_tensors(tmp_path, monkeypatch):
    bundle = write_valid_bundle(tmp_path)

    def explode(*_args, **_kwargs):
        raise AssertionError("tensor deserialization")

    monkeypatch.setattr("agoge_forger.eval._artifact_validation.safe_open", explode)
    monkeypatch.setattr("agoge_forger.eval._tensor_schema.safe_open", explode)
    monkeypatch.setattr("agoge_forger.artifacts.safetensors_io.safe_open", explode)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "pass"


def test_verify_does_not_use_the_network(tmp_path, monkeypatch):
    bundle = write_valid_bundle(tmp_path)

    def explode(*_args, **_kwargs):
        raise AssertionError("network access")

    monkeypatch.setattr("socket.socket", explode)
    monkeypatch.setattr("socket.create_connection", explode)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "pass"


def test_bundled_split_still_validates(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    manifest = validate_split_manifest(bundle / "split_manifest.json")
    assert manifest.schema_version == "agoge.split-manifest.v1"
    contract = load_evaluation_contract(bundle / EVAL_CONTRACT_PATH)
    assert isinstance(contract, PairedEvaluationContract)
    assert contract.sft.role == "causal_sft"
    assert contract.base.role == "causal_base"


def test_verify_module_does_not_import_model_runtimes():
    source = Path(__file__).resolve().parents[1] / "src" / "agoge_forger" / "release" / "verify.py"
    text = source.read_text(encoding="utf-8")
    assert "transformers" not in text
    assert "torch" not in text
    assert "safe_open" not in text
    assert "eval.generate" not in text


def test_unknown_evaluation_contract_version_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    contract_path = bundle / EVAL_CONTRACT_PATH
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "agoge.evaluation-contract.v1"
    contract_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    reseal_bundle(bundle)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "unknown_schema_version" in _codes(report)
