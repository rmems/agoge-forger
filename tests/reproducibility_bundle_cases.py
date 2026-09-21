"""Builders for miniature reproducibility-bundle fixtures."""

import json
from pathlib import Path

from agoge_forger.config import ExperimentConfig
from agoge_forger.eval.contract import (
    ArtifactIndexReference,
    DecodingContract,
    EvaluationArm,
    held_out_task_ids,
    logical_task_set_sha256,
)
from agoge_forger.release.schema import BundlePointers, write_reproducibility_bundle
from agoge_forger.split_contract import (
    SplitManifest,
    SplitMaterializationSpec,
    SplitPolicy,
    canonical_json_bytes,
    materialize_split,
    sha256_file,
)
from tests.evaluation_contract_cases import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    model_provenance,
    write_artifact_index,
    write_safetensors,
)

RUN_MANIFEST_PATH = "run/manifest.json"
LOCKED_CONFIG_PATH = "config/locked.json"
SPLIT_MANIFEST_PATH = "split_manifest.json"
ADAPTER_INDEX_PATH = "adapter/artifact_index.json"
EVAL_CONTRACT_PATH = "contract.json"


def write_valid_bundle(tmp_path: Path) -> Path:
    """Populate a sealed miniature bundle under ``tmp_path / "bundle"``."""

    source = tmp_path / "curated.jsonl"
    _write_source(source)
    bundle = tmp_path / "bundle"
    manifest = materialize_split(source, bundle, _split_spec())
    split_digest = sha256_file(bundle / SPLIT_MANIFEST_PATH)
    _write_adapter(bundle, split_digest, manifest.splits["train"].sha256)
    locked = _locked_config()
    _write_locked_config(bundle, locked)
    _write_run_manifest(bundle, locked)
    _write_evaluation_contract(bundle, manifest, split_digest)
    reseal_bundle(bundle)
    return bundle


def reseal_bundle(bundle: Path) -> Path:
    return write_reproducibility_bundle(
        bundle,
        BundlePointers(
            run_manifest_path=RUN_MANIFEST_PATH,
            locked_config_path=LOCKED_CONFIG_PATH,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            adapter_artifact_index_path=ADAPTER_INDEX_PATH,
            evaluation_contract_path=EVAL_CONTRACT_PATH,
        ),
    )


def rewrite_inventory(bundle: Path, mutate) -> None:
    inventory = bundle / "reproducibility-bundle.json"
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    mutate(payload)
    inventory.write_bytes(canonical_json_bytes(payload) + b"\n")


def _write_source(path: Path) -> None:
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
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _split_spec() -> SplitMaterializationSpec:
    return SplitMaterializationSpec(
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


def _locked_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        dataset_path="splits/train.jsonl",
        run_name="bundle-canary",
    )


def _write_adapter(bundle: Path, split_digest: str, train_digest: str) -> None:
    adapter = bundle / "adapter"
    adapter.mkdir()
    write_safetensors(adapter / "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_bytes(
        canonical_json_bytes(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
            }
        )
        + b"\n"
    )
    write_artifact_index(
        adapter,
        model_provenance(
            split_manifest_sha256=split_digest,
            train_split_sha256=train_digest,
        ),
    )


def _write_locked_config(bundle: Path, locked: ExperimentConfig) -> None:
    path = bundle / LOCKED_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(locked.model_dump(mode="json")) + b"\n")


def _write_run_manifest(bundle: Path, locked: ExperimentConfig) -> None:
    path = bundle / RUN_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": "2026-09-15T00:00:00+00:00",
        "git": {"commit": "a" * 40, "branch": "main", "dirty": False},
        "config": locked.model_dump(mode="json"),
        "metrics": {"artifact_index": ADAPTER_INDEX_PATH},
        "environment": {
            "python_version": "3.12.0",
            "torch_version": "2.4.0",
            "cuda_version": "None",
        },
    }
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _write_evaluation_contract(bundle: Path, manifest: SplitManifest, split_digest: str) -> None:
    task_ids = held_out_task_ids(manifest)
    common = _arm_common(logical_task_set_sha256(task_ids))
    index = bundle / ADAPTER_INDEX_PATH
    base = EvaluationArm(role="causal_base", **common)
    sft = EvaluationArm(
        role="causal_sft",
        artifact=ArtifactIndexReference(
            kind="peft_adapter",
            artifact_index_path=ADAPTER_INDEX_PATH,
            artifact_index_sha256=sha256_file(index),
        ),
        **common,
    )
    contract = {
        "schema_version": "agoge.evaluation-contract.v2",
        "split_manifest_path": SPLIT_MANIFEST_PATH,
        "split_manifest_sha256": split_digest,
        "held_out_split_sha256": manifest.splits["held_out"].sha256,
        "logical_task_ids": list(task_ids),
        "logical_task_set_sha256": logical_task_set_sha256(task_ids),
        "base": base.model_dump(mode="json"),
        "sft": sft.model_dump(mode="json"),
    }
    (bundle / EVAL_CONTRACT_PATH).write_bytes(canonical_json_bytes(contract) + b"\n")


def _arm_common(task_digest: str) -> dict[str, object]:
    return {
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
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
