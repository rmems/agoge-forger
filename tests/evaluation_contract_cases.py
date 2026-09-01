"""Shared builders for evaluation-contract tests."""

import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

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
BuildArgs = tuple[Path, Path, EvaluationArm, EvaluationArm]
WRITABLE_SAFETENSORS_DTYPES = (
    (torch.float64, "F64"),
    (torch.float32, "F32"),
    (torch.float16, "F16"),
    (torch.bfloat16, "BF16"),
    (torch.int64, "I64"),
    (torch.int32, "I32"),
    (torch.int16, "I16"),
    (torch.int8, "I8"),
    (torch.uint64, "U64"),
    (torch.uint32, "U32"),
    (torch.uint16, "U16"),
    (torch.uint8, "U8"),
    (torch.bool, "BOOL"),
    (torch.float8_e4m3fn, "F8_E4M3"),
    (torch.float8_e5m2, "F8_E5M2"),
    (torch.float8_e8m0fnu, "F8_E8M0"),
    (torch.float4_e2m1fn_x2, "F4"),
    (torch.complex64, "C64"),
)


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


def write_adapter_config(output_dir: Path, payload: dict[str, object]) -> None:
    complete = {"peft_type": "LORA", **payload}
    (output_dir / "adapter_config.json").write_bytes(canonical_json_bytes(complete) + b"\n")


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


def write_complete_merged_model(
    output_dir: Path,
    *,
    sharded: bool,
    tie_word_embeddings: bool = False,
    dtype: torch.dtype | None = None,
) -> None:
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tie_word_embeddings,
        dtype=dtype,
    )
    model = LlamaForCausalLM(config)
    if dtype is not None:
        model.to(dtype=dtype)
    model.save_pretrained(output_dir, max_shard_size="1KB" if sharded else "1GB")


def refresh_contract_artifact_digest(contract_path: Path, artifact_index: Path) -> None:
    payload = json.loads(contract_path.read_bytes())
    payload["sft"]["artifact"]["artifact_index_sha256"] = sha256_file(artifact_index)
    contract_path.write_bytes(canonical_json_bytes(payload) + b"\n")


def evaluation_arms(task_digest: str, tmp_path: Path):
    output_dir = tmp_path / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_complete_adapter_model(output_dir)
    artifact_index = write_artifact_index(output_dir)
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


def artifact_case(tmp_path: Path, directory: str):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    output_dir = tmp_path / directory
    output_dir.mkdir(exist_ok=True)
    return manifest_path, base, sft, output_dir


def build_contract(tmp_path: Path, manifest_path: Path, base: EvaluationArm, sft: EvaluationArm):
    return build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=tmp_path / "eval" / "contract.json",
        base=base,
        sft=sft,
    )


def assert_build_rejected(
    build_args: BuildArgs,
    expected_error: str,
    error_type: type[BaseException] = ValueError,
) -> None:
    with pytest.raises(error_type, match=expected_error):
        build_contract(*build_args)


def write_shard_index(output_dir: Path, payload: dict[str, object]) -> None:
    (output_dir / "model.safetensors.index.json").write_bytes(canonical_json_bytes(payload) + b"\n")


def paths_absent(*paths: Path) -> bool:
    return all(not path.exists() for path in paths)


def write_invalid_merged_layout(output_dir: Path, variant: str) -> None:
    write_merged_config(output_dir)
    single = "model-00001-of-00001.safetensors"
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    tensor_key = "layer.0"
    if variant.startswith("provenance-"):
        write_safetensors(output_dir / "model.safetensors")
        return
    tensor_maps = {
        "tensor-missing": (("actual.a",), ("actual.b",), {"claimed.a": first, "actual.b": second}),
        "tensor-extra": (
            ("mapped.a", "extra.a"),
            ("mapped.b",),
            {"mapped.a": first, "mapped.b": second},
        ),
        "tensor-duplicate": (
            ("shared", "a"),
            ("shared", "b"),
            {"shared": first, "a": first, "b": second},
        ),
        "tensor-misplaced": (("a",), ("b",), {"a": second, "b": first}),
    }
    if variant in tensor_maps:
        first_keys, second_keys, weight_map = tensor_maps[variant]
        write_safetensors(output_dir / first, keys=first_keys)
        write_safetensors(output_dir / second, keys=second_keys)
        write_shard_index(output_dir, {"metadata": {"total_size": 2}, "weight_map": weight_map})
        return

    shard_references = {
        "shard-bin": "model-00001-of-00001.bin",
        "shard-noncanonical": "./model-00001-of-00001.safetensors",
        "shard-unindexed": "model-00001-of-00001.safetensors",
    }
    if variant in shard_references:
        write_shard_index(
            output_dir,
            {"metadata": {"total_size": 1}, "weight_map": {tensor_key: shard_references[variant]}},
        )
    elif variant == "ambiguous":
        write_safetensors(output_dir / "model.safetensors")
        write_shard_index(
            output_dir,
            {"metadata": {"total_size": 1}, "weight_map": {tensor_key: single}},
        )
    elif variant == "duplicate-map-key":
        write_safetensors(output_dir / single)
        (output_dir / "model.safetensors.index.json").write_text(
            f'{{"metadata":{{}},"weight_map":{{"{tensor_key}":"{single}",'
            f'"{tensor_key}":"{single}"}}}}\n',
            encoding="utf-8",
        )
    elif variant == "unreferenced-shard":
        write_safetensors(output_dir / single)
        write_safetensors(output_dir / second, value=1)
        write_shard_index(
            output_dir, {"metadata": {"total_size": 2}, "weight_map": {tensor_key: single}}
        )
    else:
        invalid_indices = {
            "empty-map-missing": {},
            "empty-map-list": {"weight_map": []},
            "non-string-shard": {"metadata": {}, "weight_map": {tensor_key: 7}},
            "missing-metadata": {"weight_map": {tensor_key: single}},
        }
        write_shard_index(output_dir, invalid_indices[variant])
