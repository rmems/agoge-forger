import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agoge_forger.eval import contract as contract_module
from agoge_forger.eval.contract import (
    DecodingContract,
    PairedEvaluationContract,
    build_evaluation_contract,
    held_out_task_ids,
    logical_task_set_sha256,
    validate_evaluation_contract,
)
from agoge_forger.split_contract import (
    canonical_json_bytes,
    sha256_file,
)
from tests.evaluation_contract_cases import (
    build_contract,
    evaluation_case,
    paths_absent,
)

pytestmark = pytest.mark.usefixtures("cached_test_base_config")


def test_schema_only_contract_consumes_frozen_held_out_manifest(tmp_path):
    manifest_path, manifest, base, sft = evaluation_case(tmp_path)
    task_ids = held_out_task_ids(manifest)
    contract_path = tmp_path / "eval" / "contract.json"

    written = build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=contract_path,
        base=base,
        sft=sft,
    )
    validated = validate_evaluation_contract(contract_path)

    assert validated == written
    assert validated.logical_task_ids == task_ids
    assert validated.held_out_split_sha256 == manifest.splits["held_out"].sha256
    assert validated.base.model_repository == validated.sft.model_repository
    assert validated.schema_version == "agoge.evaluation-contract.v2"
    assert validated.sft.artifact is not None
    assert not Path(validated.sft.artifact.artifact_index_path).is_absolute()
    assert paths_absent(contract_path.parent / "base", contract_path.parent / "sft")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("tokenizer_sha256", "5" * 64),
        ("serializer_sha256", "5" * 64),
        ("context_window", 8192),
        ("truncation_policy", "reject"),
        (
            "decoding",
            DecodingContract(
                do_sample=False,
                seed=17,
                max_new_tokens=256,
                temperature=0,
                top_p=1,
            ),
        ),
        ("scoring_version", "exact-match-v2"),
    ],
)
def test_paired_contract_fails_closed_on_comparability_drift(tmp_path, field, replacement):
    _, manifest, base, sft = evaluation_case(tmp_path)
    task_ids = held_out_task_ids(manifest)
    task_digest = logical_task_set_sha256(task_ids)
    drifted_sft = sft.model_copy(update={field: replacement})

    with pytest.raises(ValidationError, match="non-comparable"):
        PairedEvaluationContract(
            split_manifest_path="split_manifest.json",
            split_manifest_sha256="6" * 64,
            held_out_split_sha256=manifest.splits["held_out"].sha256,
            logical_task_ids=task_ids,
            logical_task_set_sha256=task_digest,
            base=base,
            sft=drifted_sft,
        )


def test_evaluation_contract_detects_manifest_mutation(tmp_path):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_contract(tmp_path, manifest_path, base, sft)

    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="split-manifest SHA-256 mismatch"):
        validate_evaluation_contract(contract_path)


def test_evaluation_contract_recomputes_manifest_split_ownership(tmp_path):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_contract(tmp_path, manifest_path, base, sft)
    manifest_payload = json.loads(manifest_path.read_bytes())
    manifest_payload["split_policy"]["salt"] = "rewritten-policy"
    manifest_path.write_bytes(canonical_json_bytes(manifest_payload) + b"\n")
    contract_payload = json.loads(contract_path.read_bytes())
    contract_payload["split_manifest_sha256"] = sha256_file(manifest_path)
    contract_path.write_bytes(canonical_json_bytes(contract_payload) + b"\n")

    with pytest.raises(ValueError, match="split ownership differs"):
        validate_evaluation_contract(contract_path)


def test_evaluation_contract_hashes_and_validates_one_manifest_snapshot(tmp_path, monkeypatch):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    written = build_contract(tmp_path, manifest_path, base, sft)
    original_validate = contract_module.validate_split_manifest_snapshot

    def replace_after_snapshot(path, content):
        replacement = json.loads(manifest_path.read_bytes())
        replacement["source"]["revision"] = "f" * 40
        manifest_path.write_bytes(canonical_json_bytes(replacement) + b"\n")
        return original_validate(path, content)

    monkeypatch.setattr(
        contract_module,
        "validate_split_manifest_snapshot",
        replace_after_snapshot,
    )

    assert validate_evaluation_contract(contract_path) == written


def test_evaluation_contract_refuses_overwrite(tmp_path):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    build_contract(tmp_path, manifest_path, base, sft)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_contract(tmp_path, manifest_path, base, sft)


def test_evaluation_contract_rejects_non_finite_temperature_without_creating_file(tmp_path):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    invalid_decoding = base.decoding.model_copy(update={"temperature": float("inf")})
    invalid_base = base.model_copy(update={"decoding": invalid_decoding})
    invalid_sft = sft.model_copy(update={"decoding": invalid_decoding})
    contract_path = tmp_path / "eval" / "contract.json"

    with pytest.raises(ValidationError, match="finite number"):
        build_contract(tmp_path, manifest_path, invalid_base, invalid_sft)
    assert not contract_path.exists()


def test_evaluation_contract_serializes_before_creating_file(tmp_path, monkeypatch):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    original_canonical_json_bytes = contract_module.canonical_json_bytes

    def reject_contract(value):
        if (
            isinstance(value, dict)
            and value.get("schema_version") == "agoge.evaluation-contract.v2"
        ):
            raise ValueError("synthetic serialization failure")
        return original_canonical_json_bytes(value)

    monkeypatch.setattr(contract_module, "canonical_json_bytes", reject_contract)

    with pytest.raises(ValueError, match="synthetic serialization failure"):
        build_contract(tmp_path, manifest_path, base, sft)
    assert not contract_path.exists()


def test_evaluation_contract_cleans_partial_staging_after_write_failure(tmp_path, monkeypatch):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"

    def fail_after_partial_write(path, payload):
        path.write_bytes(payload[:8])
        raise OSError("synthetic contract write failure")

    monkeypatch.setattr(
        contract_module,
        "_write_contract_payload",
        fail_after_partial_write,
        raising=False,
    )

    with pytest.raises(OSError, match="synthetic contract write failure"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=contract_path,
            base=base,
            sft=sft,
        )

    assert not contract_path.exists()
    assert not any(contract_path.parent.iterdir())


def test_evaluation_contract_requires_immutable_model_and_tokenizer_revisions(tmp_path):
    _, _, base, _ = evaluation_case(tmp_path)

    with pytest.raises(ValidationError, match="model_revision"):
        base.model_copy(update={"model_revision": "main"}).model_validate(
            base.model_copy(update={"model_revision": "main"}).model_dump()
        )
    with pytest.raises(ValidationError, match="tokenizer_revision"):
        base.model_copy(update={"tokenizer_revision": "latest"}).model_validate(
            base.model_copy(update={"tokenizer_revision": "latest"}).model_dump()
        )


def test_contract_references_are_serialized_with_posix_separators(monkeypatch):
    monkeypatch.setattr(contract_module.os.path, "relpath", lambda path, anchor: r"..\bundle\x")

    assert contract_module._portable_relative_path(Path("x"), Path("y")) == "../bundle/x"


def test_evaluation_contract_rejects_absolute_manifest_reference(tmp_path):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_contract(tmp_path, manifest_path, base, sft)
    payload = json.loads(contract_path.read_bytes())
    payload["split_manifest_path"] = str(manifest_path)
    contract_path.write_bytes(canonical_json_bytes(payload) + b"\n")

    with pytest.raises(ValidationError, match="portable relative paths"):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize("reference", ["/absolute/index.json", r"C:\\temp\\index.json"])
def test_evaluation_contract_rejects_absolute_artifact_reference(tmp_path, reference):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_contract(tmp_path, manifest_path, base, sft)
    payload = json.loads(contract_path.read_bytes())
    payload["sft"]["artifact"]["artifact_index_path"] = reference
    contract_path.write_bytes(canonical_json_bytes(payload) + b"\n")

    with pytest.raises(ValidationError, match="portable relative paths"):
        validate_evaluation_contract(contract_path)


def test_evaluation_contract_rejects_duplicate_json_keys(tmp_path):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_contract(tmp_path, manifest_path, base, sft)
    payload = contract_path.read_bytes().replace(
        b"{",
        b'{"schema_version":"agoge.evaluation-contract.v2",',
        1,
    )
    contract_path.write_bytes(payload)

    with pytest.raises(ValueError, match="invalid evaluation contract JSON"):
        validate_evaluation_contract(contract_path)
