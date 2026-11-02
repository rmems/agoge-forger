"""Shared builders for evaluation-contract tests."""

import json
from pathlib import Path

from agoge_forger.eval.contract import (
    ArtifactIndexReference,
    DecodingContract,
    EvaluationArm,
    build_evaluation_contract,
    held_out_task_ids,
    logical_task_set_sha256,
)
from agoge_forger.split_contract import (
    SplitMaterializationSpec,
    SplitPolicy,
    canonical_json_bytes,
    materialize_split,
    sha256_file,
)
from tests.peft_adapter_fixtures import write_complete_adapter_model

MODEL_REPOSITORY = "example/base-model"
MODEL_REVISION = "abcdef0123456789abcdef0123456789abcdef01"


def write_safetensors(path: Path, value: int = 0, *, keys: tuple[str, ...] = ("weight",)) -> None:
    tensors = {
        key: {"dtype": "U8", "shape": [1], "data_offsets": [index, index + 1]}
        for index, key in enumerate(keys)
    }
    header = json.dumps(tensors, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    data = bytes((value + index) % 256 for index in range(len(keys)))
    path.write_bytes(len(header).to_bytes(8, "little") + header + data)


def frozen_manifest(tmp_path: Path):
    source = tmp_path / "curated.jsonl"
    rows = [
        {
            "canonical_id": f"task-{index:03d}",
            "lineage_id": f"lineage-{index // 2:03d}",
            "group_id": f"family-{index // 3:03d}",
            "text": f"Task {index}: return deterministic answer {index * 13}.",
            "completion_start_char": len(f"Task {index}: "),
        }
        for index in range(90)
    ]
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    output = tmp_path / "snapshot"
    spec = SplitMaterializationSpec(
        source_repository="rmems/synthetic-factory",
        source_revision="abcdef0123456789abcdef0123456789abcdef01",
        source_path="curated.jsonl",
        dataset_version="curated-eval-v1",
        split_policy=SplitPolicy(
            seed=99,
            salt="paired-evaluation-v1",
            weights={"train": 6, "validation": 2, "held_out": 2},
        ),
    )
    manifest = materialize_split(source, output, spec)
    return output / "split_manifest.json", manifest


def write_artifact_index(output_dir: Path, provenance=None) -> Path:
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
    payload: dict[str, object] = {"output_dir": str(output_dir), "artifacts": artifacts}
    if provenance is not None:
        payload["producer_provenance"] = provenance
    artifact_index.write_bytes(canonical_json_bytes(payload) + b"\n")
    return artifact_index


def with_artifact(sft: EvaluationArm, output_dir: Path, *, kind, provenance=None):
    if provenance is None:
        manifest_path = output_dir.parent / "snapshot" / "split_manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        provenance = model_provenance(
            split_manifest_sha256=sha256_file(manifest_path),
            train_split_sha256=manifest["splits"]["train"]["sha256"],
        )
    index_provenance = None if provenance is False else provenance
    artifact_index = write_artifact_index(output_dir, index_provenance)
    return sft.model_copy(
        update={
            "artifact": ArtifactIndexReference(
                kind=kind,
                artifact_index_path=str(artifact_index),
                artifact_index_sha256=sha256_file(artifact_index),
            )
        }
    )


def model_provenance(
    repository: object = MODEL_REPOSITORY,
    revision: object = MODEL_REVISION,
    *,
    split_manifest_sha256: object = "7" * 64,
    train_split_sha256: object = "8" * 64,
) -> dict[str, object]:
    return {
        "base_model_name_or_path": repository,
        "revision": revision,
        "training_split_manifest_sha256": split_manifest_sha256,
        "training_split_name": "train",
        "training_split_sha256": train_split_sha256,
    }


def write_merged_config(output_dir: Path, payload: dict[str, str] | None = None) -> None:
    complete = {"model_type": "llama"} if payload is None else payload
    (output_dir / "config.json").write_bytes(canonical_json_bytes(complete) + b"\n")


def evaluation_arms(task_digest: str, tmp_path: Path):
    output_dir = tmp_path / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_complete_adapter_model(output_dir)
    artifact_index = write_artifact_index(output_dir)
    common = {
        "tokenizer_repository": "example/tokenizer",
        "tokenizer_revision": "1111111111111111111111111111111111111111",
        "tokenizer_sha256": "3" * 64,
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
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        **common,
    )
    sft = EvaluationArm(
        role="causal_sft",
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        artifact=ArtifactIndexReference(
            kind="peft_adapter",
            artifact_index_path=str(artifact_index),
            artifact_index_sha256=sha256_file(artifact_index),
        ),
        **common,
    )
    return base, sft


def evaluation_case(tmp_path: Path):
    manifest_path, manifest = frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = evaluation_arms(digest, tmp_path)
    sft = with_artifact(sft, tmp_path / "adapter", kind="peft_adapter")
    return manifest_path, manifest, base, sft


def build_contract(tmp_path: Path, manifest_path: Path, base: EvaluationArm, sft: EvaluationArm):
    return build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=tmp_path / "eval" / "contract.json",
        base=base,
        sft=sft,
    )


def write_shard_index(output_dir: Path, payload: dict[str, object]) -> None:
    (output_dir / "model.safetensors.index.json").write_bytes(canonical_json_bytes(payload) + b"\n")


def paths_absent(*paths: Path) -> bool:
    return all(not path.exists() for path in paths)
