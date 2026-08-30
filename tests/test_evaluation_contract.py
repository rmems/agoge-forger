from pathlib import Path

import pytest
from pydantic import ValidationError

from agoge_forger.eval.contract import (
    ArtifactIndexReference,
    DecodingContract,
    EvaluationArm,
    PairedEvaluationContract,
    build_evaluation_contract,
    held_out_task_ids,
    logical_task_set_sha256,
    validate_evaluation_contract,
)
from agoge_forger.split_contract import (
    SplitMaterializationSpec,
    SplitPolicy,
    canonical_json_bytes,
    materialize_split,
    sha256_file,
)


def _frozen_manifest(tmp_path: Path):
    source = tmp_path / "curated.jsonl"
    rows = [
        {
            "canonical_id": f"task-{index:03d}",
            "lineage_id": f"lineage-{index // 2:03d}",
            "group_id": f"family-{index // 3:03d}",
            "text": f"Task {index}: return deterministic answer {index * 13}.",
        }
        for index in range(90)
    ]
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    output = tmp_path / "snapshot"
    spec = SplitMaterializationSpec(
        source_repository="rmems/synthetic-factory",
        source_revision="abcdef0123456789abcdef0123456789abcdef01",
        dataset_version="curated-eval-v1",
        split_policy=SplitPolicy(
            seed=99,
            salt="paired-evaluation-v1",
            weights={"train": 6, "validation": 2, "held_out": 2},
        ),
    )
    manifest = materialize_split(source, output, spec)
    return output / "split_manifest.json", manifest


def _write_artifact_index(output_dir: Path) -> Path:
    artifact_index = output_dir / "artifact_index.json"
    artifacts = [
        {
            "file": str(path.relative_to(output_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != artifact_index
    ]
    artifact_index.write_bytes(
        canonical_json_bytes({"output_dir": str(output_dir), "artifacts": artifacts}) + b"\n"
    )
    return artifact_index


def _arms(task_digest: str, tmp_path: Path):
    output_dir = tmp_path / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = output_dir / "adapter_model.safetensors"
    adapter_weights.write_bytes(b"adapter-v1")
    adapter_config = output_dir / "adapter_config.json"
    adapter_config.write_bytes(b"{}\n")
    artifact_index = _write_artifact_index(output_dir)
    common = {
        "tokenizer_repository": "example/tokenizer",
        "tokenizer_revision": "1111111111111111111111111111111111111111",
        "serializer_id": "messages-v1",
        "serializer_version": "1",
        "serializer_sha256": "2" * 64,
        "logical_task_set_sha256": task_digest,
        "context_window": 4096,
        "truncation_policy": "mark_unsupported",
        "decoding": DecodingContract(
            do_sample=False,
            seed=17,
            max_new_tokens=128,
            temperature=0,
            top_p=1,
        ),
        "scoring_version": "exact-match-v1",
    }
    base = EvaluationArm(
        role="causal_base",
        model_repository="example/base-model",
        model_revision="3" * 40,
        **common,
    )
    sft = EvaluationArm(
        role="causal_sft",
        model_repository="example/base-model",
        model_revision="3" * 40,
        artifact=ArtifactIndexReference(
            kind="peft_adapter",
            artifact_index_path=str(artifact_index),
            artifact_index_sha256=sha256_file(artifact_index),
        ),
        **common,
    )
    return base, sft


def test_schema_only_contract_consumes_frozen_held_out_manifest(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    task_ids = held_out_task_ids(manifest)
    task_digest = logical_task_set_sha256(task_ids)
    base, sft = _arms(task_digest, tmp_path)
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
    assert validated.sft.artifact is not None
    assert not Path(validated.sft.artifact.artifact_index_path).is_absolute()
    assert not (contract_path.parent / "base").exists()
    assert not (contract_path.parent / "sft").exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
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
    _, manifest = _frozen_manifest(tmp_path)
    task_ids = held_out_task_ids(manifest)
    task_digest = logical_task_set_sha256(task_ids)
    base, sft = _arms(task_digest, tmp_path)
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
    manifest_path, manifest = _frozen_manifest(tmp_path)
    task_ids = held_out_task_ids(manifest)
    task_digest = logical_task_set_sha256(task_ids)
    base, sft = _arms(task_digest, tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=contract_path,
        base=base,
        sft=sft,
    )

    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="split-manifest SHA-256 mismatch"):
        validate_evaluation_contract(contract_path)


def test_evaluation_contract_refuses_overwrite(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    task_digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(task_digest, tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=contract_path,
        base=base,
        sft=sft,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=contract_path,
            base=base,
            sft=sft,
        )


def test_evaluation_contract_requires_immutable_model_and_tokenizer_revisions(tmp_path):
    _, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, _ = _arms(digest, tmp_path)

    with pytest.raises(ValidationError, match="model_revision"):
        base.model_copy(update={"model_revision": "main"}).model_validate(
            base.model_copy(update={"model_revision": "main"}).model_dump()
        )
    with pytest.raises(ValidationError, match="tokenizer_revision"):
        base.model_copy(update={"tokenizer_revision": "latest"}).model_validate(
            base.model_copy(update={"tokenizer_revision": "latest"}).model_dump()
        )


def test_evaluation_contract_detects_artifact_index_mutation(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    contract = build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=contract_path,
        base=base,
        sft=sft,
    )
    assert contract.sft.artifact is not None
    artifact_path = (contract_path.parent / contract.sft.artifact.artifact_index_path).resolve()
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifact-index SHA-256 mismatch"):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize(
    ("mutated", "expected_error"),
    [
        (b"adapter-v2", "indexed artifact SHA-256 mismatch"),
        (b"adapter-version-two", "indexed artifact size mismatch"),
    ],
)
def test_evaluation_contract_detects_indexed_artifact_mutation(tmp_path, mutated, expected_error):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=contract_path,
        base=base,
        sft=sft,
    )

    weights = tmp_path / "adapter" / "adapter_model.safetensors"
    weights.write_bytes(mutated)
    with pytest.raises(ValueError, match=expected_error):
        validate_evaluation_contract(contract_path)


def test_evaluation_contract_rejects_artifact_index_path_escape(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    assert sft.artifact is not None
    artifact_index = Path(sft.artifact.artifact_index_path)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    artifact_index.write_bytes(
        canonical_json_bytes(
            {
                "output_dir": str(artifact_index.parent),
                "artifacts": [
                    {
                        "file": "../outside.safetensors",
                        "size_bytes": outside.stat().st_size,
                        "sha256": sha256_file(outside),
                    }
                ],
            }
        )
        + b"\n"
    )
    escaped_sft = sft.model_copy(
        update={
            "artifact": ArtifactIndexReference(
                kind="peft_adapter",
                artifact_index_path=str(artifact_index),
                artifact_index_sha256=sha256_file(artifact_index),
            )
        }
    )
    contract_path = tmp_path / "eval" / "contract.json"

    with pytest.raises(ValueError, match="must stay relative"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=contract_path,
            base=base,
            sft=escaped_sft,
        )
    assert not contract_path.exists()


def test_evaluation_contract_revalidates_copied_arms_before_writing(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    invalid_base = base.model_copy(update={"context_window": 0})
    invalid_sft = sft.model_copy(update={"context_window": 0})
    contract_path = tmp_path / "eval" / "contract.json"

    with pytest.raises(ValidationError, match="context_window"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=contract_path,
            base=invalid_base,
            sft=invalid_sft,
        )
    assert not contract_path.exists()
