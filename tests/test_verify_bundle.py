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
    release = Path(__file__).resolve().parents[1] / "src" / "agoge_forger" / "release"
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(release.glob("*.py")))
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


def test_deleted_listed_file_is_missing_not_crash(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    (bundle / "adapter" / "adapter_model.safetensors").unlink()
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "missing_file" in _codes(report)


def test_artifact_index_digest_mismatch_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    target = bundle / "adapter" / "adapter_model.safetensors"
    target.write_bytes(target.read_bytes() + b"\x00")
    reseal_bundle(bundle)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "modified_file" in _codes(report)


def test_trust_remote_code_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    _rewrite_locked_and_run_config(bundle, trust_remote_code=True)
    reseal_bundle(bundle)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "locked_config" in _codes(report)
    assert any("trust_remote_code" in failure.message for failure in report.failures)


def test_missing_revision_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    _rewrite_locked_and_run_config(bundle, revision=None)
    reseal_bundle(bundle)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "locked_config" in _codes(report)
    assert any("revision" in failure.message for failure in report.failures)


def test_evaluation_model_identity_mismatch_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    contract_path = bundle / EVAL_CONTRACT_PATH
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["base"]["model_repository"] = "other/base-model"
    payload["sft"]["model_repository"] = "other/base-model"
    contract_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    reseal_bundle(bundle)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert any("model repository" in failure.message for failure in report.failures)


def test_artifact_kind_must_match_selected_index(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    contract_path = bundle / EVAL_CONTRACT_PATH
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["sft"]["artifact"]["kind"] = "merged_model"
    contract_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    reseal_bundle(bundle)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert "artifact_index" in _codes(report)


def test_malformed_adapter_config_fails(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    config_path = bundle / "adapter" / "adapter_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    del payload["peft_type"]
    config_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rewrite_adapter_index(bundle)
    reseal_bundle(bundle)
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert any("peft_type" in failure.message for failure in report.failures)


def test_sharded_merged_artifact_requires_shards(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    merged = bundle / "merged"
    merged.mkdir()
    (merged / "config.json").write_bytes(canonical_json_bytes({"model_type": "llama"}) + b"\n")
    (merged / "model.safetensors.index.json").write_bytes(
        canonical_json_bytes({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}) + b"\n"
    )
    from agoge_forger.release.schema import BundlePointers, write_reproducibility_bundle
    from agoge_forger.split_contract import sha256_file
    from tests.evaluation_contract_cases import model_provenance, write_artifact_index
    from tests.reproducibility_bundle_cases import (
        ADAPTER_INDEX_PATH,
        EVAL_CONTRACT_PATH,
        LOCKED_CONFIG_PATH,
        RUN_MANIFEST_PATH,
        SPLIT_MANIFEST_PATH,
    )

    split_digest = sha256_file(bundle / SPLIT_MANIFEST_PATH)
    train_digest = json.loads((bundle / SPLIT_MANIFEST_PATH).read_text(encoding="utf-8"))["splits"][
        "train"
    ]["sha256"]
    write_artifact_index(
        merged,
        model_provenance(split_manifest_sha256=split_digest, train_split_sha256=train_digest),
    )
    write_reproducibility_bundle(
        bundle,
        BundlePointers(
            run_manifest_path=RUN_MANIFEST_PATH,
            locked_config_path=LOCKED_CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            adapter_artifact_index_path=ADAPTER_INDEX_PATH,
            evaluation_contract_path=EVAL_CONTRACT_PATH,
            merged_artifact_index_path="merged/artifact_index.json",
        ),
    )
    report = verify_reproducibility_bundle(bundle)
    assert report.verdict == "fail"
    assert any("missing shards" in failure.message for failure in report.failures)


def test_table_report_escapes_control_characters(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    (bundle / "bad\nname.txt").write_text("unexpected\n", encoding="utf-8")
    report = verify_reproducibility_bundle(bundle)
    table = format_verification_table(report)
    assert "\\u000a" in table
    assert "bad\nname" not in table


def test_read_only_bundle_still_passes(tmp_path):
    bundle = write_valid_bundle(tmp_path)
    _chmod_tree(bundle, dir_mode=0o555, file_mode=0o444)
    try:
        report = verify_reproducibility_bundle(bundle)
        assert report.verdict == "pass"
    finally:
        _chmod_tree(bundle, dir_mode=0o755, file_mode=0o644)


def _rewrite_adapter_index(bundle) -> None:
    from agoge_forger.split_contract import sha256_file
    from tests.evaluation_contract_cases import model_provenance, write_artifact_index
    from tests.reproducibility_bundle_cases import SPLIT_MANIFEST_PATH

    split_digest = sha256_file(bundle / SPLIT_MANIFEST_PATH)
    train_digest = json.loads((bundle / SPLIT_MANIFEST_PATH).read_text(encoding="utf-8"))["splits"][
        "train"
    ]["sha256"]
    write_artifact_index(
        bundle / "adapter",
        model_provenance(split_manifest_sha256=split_digest, train_split_sha256=train_digest),
    )
    index = bundle / "adapter" / "artifact_index.json"
    contract_path = bundle / EVAL_CONTRACT_PATH
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["sft"]["artifact"]["artifact_index_sha256"] = sha256_file(index)
    contract_path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _rewrite_locked_and_run_config(bundle, **updates):
    from tests.reproducibility_bundle_cases import LOCKED_CONFIG_PATH, RUN_MANIFEST_PATH

    locked_path = bundle / LOCKED_CONFIG_PATH
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    locked.update(updates)
    locked_path.write_bytes(canonical_json_bytes(locked) + b"\n")
    run_path = bundle / RUN_MANIFEST_PATH
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["config"].update(updates)
    run_path.write_bytes(canonical_json_bytes(run) + b"\n")


def _chmod_tree(root, *, dir_mode: int, file_mode: int) -> None:
    import os

    for dirpath, _dirnames, filenames in os.walk(root):
        os.chmod(dirpath, dir_mode)
        for name in filenames:
            os.chmod(Path(dirpath) / name, file_mode)
