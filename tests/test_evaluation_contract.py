import json
from pathlib import Path
from typing import Literal

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

MODEL_REPOSITORY = "example/base-model"
MODEL_REVISION = "abcdef0123456789abcdef0123456789abcdef01"


def _write_safetensors(path: Path, value: int = 0, *, keys: tuple[str, ...] = ("weight",)) -> None:
    tensors = {
        key: {"dtype": "U8", "shape": [1], "data_offsets": [index, index + 1]}
        for index, key in enumerate(keys)
    }
    header = json.dumps(tensors, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    data = bytes((value + index) % 256 for index in range(len(keys)))
    path.write_bytes(len(header).to_bytes(8, "little") + header + data)


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


def _with_artifact(
    sft: EvaluationArm,
    output_dir: Path,
    *,
    kind: Literal["peft_adapter", "merged_model"],
) -> EvaluationArm:
    artifact_index = _write_artifact_index(output_dir)
    return sft.model_copy(
        update={
            "artifact": ArtifactIndexReference(
                kind=kind,
                artifact_index_path=str(artifact_index),
                artifact_index_sha256=sha256_file(artifact_index),
            )
        }
    )


def _write_adapter_config(output_dir: Path, payload: dict[str, object]) -> None:
    complete = {"peft_type": "LORA", **payload}
    (output_dir / "adapter_config.json").write_bytes(canonical_json_bytes(complete) + b"\n")


def _write_merged_config(output_dir: Path, payload: dict[str, str] | None = None) -> None:
    complete = {"model_type": "llama"} if payload is None else payload
    (output_dir / "config.json").write_bytes(canonical_json_bytes(complete) + b"\n")


def _refresh_contract_artifact_digest(contract_path: Path, artifact_index: Path) -> None:
    payload = json.loads(contract_path.read_bytes())
    payload["sft"]["artifact"]["artifact_index_sha256"] = sha256_file(artifact_index)
    contract_path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _arms(task_digest: str, tmp_path: Path):
    output_dir = tmp_path / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = output_dir / "adapter_model.safetensors"
    _write_safetensors(adapter_weights)
    _write_adapter_config(
        output_dir,
        {
            "base_model_name_or_path": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
    )
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
    ("mutation", "expected_error"),
    [
        ("same-size", "indexed artifact SHA-256 mismatch"),
        ("different-size", "indexed artifact size mismatch"),
    ],
)
def test_evaluation_contract_detects_indexed_artifact_mutation(tmp_path, mutation, expected_error):
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
    original = weights.read_bytes()
    mutated = bytes([original[0] ^ 1]) + original[1:]
    if mutation == "different-size":
        mutated += b"x"
    weights.write_bytes(mutated)
    with pytest.raises(ValueError, match=expected_error):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize(
    ("kind", "expected_error"),
    [
        ("peft_adapter", "peft_adapter artifact is missing required indexed files"),
        ("merged_model", "merged_model artifact is missing required indexed files"),
    ],
)
def test_evaluation_contract_enforces_declared_artifact_kind(tmp_path, kind, expected_error):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "wrong-kind"
    output_dir.mkdir()
    if kind == "peft_adapter":
        _write_merged_config(output_dir)
        _write_safetensors(output_dir / "model.safetensors")
    else:
        _write_adapter_config(
            output_dir,
            {
                "base_model_name_or_path": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
            },
        )
        _write_safetensors(output_dir / "adapter_model.safetensors")
    wrong_kind_sft = _with_artifact(sft, output_dir, kind=kind)

    with pytest.raises(ValueError, match=expected_error):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=wrong_kind_sft,
        )


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "adapter_model.bin",
        "checkpoint-1/adapter_model.bin",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model.bin.index.json",
        "adapter_model.BIN",
    ],
)
def test_evaluation_contract_rejects_unsafe_adapter_weight_substitute(tmp_path, unsafe_name):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "unsafe-adapter"
    output_dir.mkdir()
    _write_adapter_config(
        output_dir,
        {
            "base_model_name_or_path": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
    )
    unsafe_path = output_dir / unsafe_name
    unsafe_path.parent.mkdir(parents=True, exist_ok=True)
    unsafe_path.write_bytes(b"pickle-weights")
    unsafe_sft = _with_artifact(sft, output_dir, kind="peft_adapter")

    with pytest.raises(ValueError, match="unsafe serialized weights"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=unsafe_sft,
        )


def test_evaluation_contract_accepts_indexed_adapter_training_state(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "adapter"
    (output_dir / "README.md").write_text("adapter card\n", encoding="utf-8")
    checkpoint = output_dir / "checkpoint-1"
    checkpoint.mkdir()
    _write_safetensors(checkpoint / "adapter_model.safetensors", value=1)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-state")
    (checkpoint / "training_args.bin").write_bytes(b"trainer-arguments")
    complete_sft = _with_artifact(sft, output_dir, kind="peft_adapter")
    contract_path = tmp_path / "eval" / "contract.json"

    written = build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=contract_path,
        base=base,
        sft=complete_sft,
    )

    assert validate_evaluation_contract(contract_path) == written


def test_evaluation_contract_rejects_conflicting_root_adapter_safetensors(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "adapter"
    _write_safetensors(output_dir / "model.safetensors")
    conflicting_sft = _with_artifact(sft, output_dir, kind="peft_adapter")

    with pytest.raises(ValueError, match="only adapter_model.safetensors weights"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=conflicting_sft,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "omitted.txt",
        ".hidden",
        "nested/omitted.safetensors",
        "nested/artifact_index.json",
    ],
)
def test_evaluation_contract_rejects_files_omitted_from_artifact_index(tmp_path, relative_path):
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
    omitted = tmp_path / "adapter" / relative_path
    omitted.parent.mkdir(parents=True, exist_ok=True)
    omitted.write_bytes(b"not-indexed")

    with pytest.raises(ValueError, match=f"omitted from artifact index.*{omitted.name}"):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize("target", ["adapter_model.safetensors", ".", "missing"])
def test_evaluation_contract_rejects_symlink_in_artifact_bundle(tmp_path, target):
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
    (tmp_path / "adapter" / "artifact-link").symlink_to(target)

    with pytest.raises(ValueError, match="artifact bundle cannot contain symlinks"):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize(
    ("config", "expected_error"),
    [
        ({"revision": MODEL_REVISION}, "missing required base_model_name_or_path"),
        ({"base_model_name_or_path": MODEL_REPOSITORY}, "missing required revision"),
        (
            {
                "base_model_name_or_path": "example/other-model",
                "revision": MODEL_REVISION,
            },
            "base_model_name_or_path does not match",
        ),
        (
            {
                "base_model_name_or_path": f"{MODEL_REPOSITORY}/",
                "revision": MODEL_REVISION,
            },
            "base_model_name_or_path does not match",
        ),
        (
            {
                "base_model_name_or_path": MODEL_REPOSITORY.upper(),
                "revision": MODEL_REVISION,
            },
            "base_model_name_or_path does not match",
        ),
        (
            {
                "base_model_name_or_path": MODEL_REPOSITORY,
                "revision": "4" * 40,
            },
            "revision does not match",
        ),
        (
            {
                "base_model_name_or_path": MODEL_REPOSITORY,
                "revision": MODEL_REVISION.upper(),
            },
            "revision does not match",
        ),
        (
            {
                "base_model_name_or_path": MODEL_REPOSITORY,
                "revision": "main",
            },
            "revision does not match",
        ),
        (
            {
                "base_model_name_or_path": MODEL_REPOSITORY,
                "revision": None,
            },
            "missing required revision",
        ),
    ],
)
def test_evaluation_contract_binds_peft_adapter_to_sft_base(tmp_path, config, expected_error):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "adapter"
    _write_adapter_config(output_dir, config)
    mismatched_sft = _with_artifact(sft, output_dir, kind="peft_adapter")

    with pytest.raises(ValueError, match=expected_error):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=mismatched_sft,
        )


def test_evaluation_contract_revalidates_peft_provenance(tmp_path):
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
    output_dir = tmp_path / "adapter"
    _write_adapter_config(
        output_dir,
        {
            "base_model_name_or_path": MODEL_REPOSITORY,
            "revision": "4" * 40,
        },
    )
    artifact_index = _write_artifact_index(output_dir)
    _refresh_contract_artifact_digest(contract_path, artifact_index)

    with pytest.raises(ValueError, match="revision does not match"):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize("layout", ["single", "sharded"])
def test_evaluation_contract_accepts_valid_merged_model_layouts(tmp_path, layout):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    _write_merged_config(output_dir)
    if layout == "single":
        _write_safetensors(output_dir / "model.safetensors")
    else:
        first = "model-00001-of-00002.safetensors"
        second = "model-00002-of-00002.safetensors"
        _write_safetensors(
            output_dir / first,
            keys=("layer.0.weight", "layer.0.bias"),
        )
        _write_safetensors(output_dir / second, value=1, keys=("layer.1.weight",))
        (output_dir / "model.safetensors.index.json").write_bytes(
            canonical_json_bytes(
                {
                    "metadata": {"total_size": 2},
                    "weight_map": {
                        "layer.0.weight": first,
                        "layer.0.bias": first,
                        "layer.1.weight": second,
                    },
                }
            )
            + b"\n"
        )
    merged_sft = _with_artifact(sft, output_dir, kind="merged_model")
    contract_path = tmp_path / "eval" / "contract.json"

    written = build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=contract_path,
        base=base,
        sft=merged_sft,
    )

    assert validate_evaluation_contract(contract_path) == written
    assert written.sft.artifact is not None
    assert written.sft.artifact.kind == "merged_model"


@pytest.mark.parametrize(
    ("variant", "expected_error"),
    [
        ("missing", "names tensors absent from shards"),
        ("extra", "tensors absent from weight_map"),
        ("duplicate", "tensor keys occur in multiple"),
        ("misplaced", "assigns tensors to wrong shards"),
    ],
)
def test_evaluation_contract_requires_exact_merged_tensor_shard_map(
    tmp_path, variant, expected_error
):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    _write_merged_config(output_dir)
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    if variant == "missing":
        first_keys, second_keys = ("actual.a",), ("actual.b",)
        weight_map = {"claimed.a": first, "actual.b": second}
    elif variant == "extra":
        first_keys, second_keys = ("mapped.a", "extra.a"), ("mapped.b",)
        weight_map = {"mapped.a": first, "mapped.b": second}
    elif variant == "duplicate":
        first_keys, second_keys = ("shared", "a"), ("shared", "b")
        weight_map = {"shared": first, "a": first, "b": second}
    else:
        first_keys, second_keys = ("a",), ("b",)
        weight_map = {"a": second, "b": first}
    _write_safetensors(output_dir / first, keys=first_keys)
    _write_safetensors(output_dir / second, keys=second_keys)
    (output_dir / "model.safetensors.index.json").write_bytes(
        canonical_json_bytes({"metadata": {"total_size": 2}, "weight_map": weight_map}) + b"\n"
    )
    merged_sft = _with_artifact(sft, output_dir, kind="merged_model")

    with pytest.raises(ValueError, match=expected_error):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=merged_sft,
        )


@pytest.mark.parametrize(
    ("shard", "expected_error"),
    [
        ("model-00001-of-00001.bin", "must reference safetensors shards only"),
        ("./model-00001-of-00001.safetensors", "shard paths must be canonical"),
        ("model-00001-of-00001.safetensors", "references unindexed shards"),
    ],
)
def test_evaluation_contract_rejects_invalid_merged_model_shard_reference(
    tmp_path, shard, expected_error
):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    _write_merged_config(output_dir)
    (output_dir / "model.safetensors.index.json").write_bytes(
        canonical_json_bytes({"metadata": {"total_size": 1}, "weight_map": {"layer.0": shard}})
        + b"\n"
    )
    merged_sft = _with_artifact(sft, output_dir, kind="merged_model")

    with pytest.raises(ValueError, match=expected_error):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=merged_sft,
        )


def test_evaluation_contract_rejects_ambiguous_merged_model_weights(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    _write_merged_config(output_dir)
    _write_safetensors(output_dir / "model.safetensors")
    (output_dir / "model.safetensors.index.json").write_bytes(
        canonical_json_bytes(
            {
                "metadata": {"total_size": 1},
                "weight_map": {"layer.0": "model-00001-of-00001.safetensors"},
            }
        )
        + b"\n"
    )
    ambiguous_sft = _with_artifact(sft, output_dir, kind="merged_model")

    with pytest.raises(ValueError, match="exactly one of model.safetensors"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=ambiguous_sft,
        )


def test_evaluation_contract_rejects_duplicate_merged_weight_map_keys(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    _write_merged_config(output_dir)
    shard = "model-00001-of-00001.safetensors"
    _write_safetensors(output_dir / shard)
    (output_dir / "model.safetensors.index.json").write_text(
        f'{{"metadata":{{}},"weight_map":{{"layer.0":"{shard}","layer.0":"{shard}"}}}}\n',
        encoding="utf-8",
    )
    merged_sft = _with_artifact(sft, output_dir, kind="merged_model")

    with pytest.raises(ValueError, match="invalid merged-model shard index"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=merged_sft,
        )


@pytest.mark.parametrize(
    ("payload", "error_type", "expected_error"),
    [
        ({}, ValueError, "requires a non-empty weight_map"),
        ({"weight_map": []}, ValueError, "requires a non-empty weight_map"),
        (
            {"metadata": {}, "weight_map": {"layer.0": 7}},
            TypeError,
            "shard paths must be strings",
        ),
        (
            {"weight_map": {"layer.0": "model-00001-of-00001.safetensors"}},
            TypeError,
            "requires a metadata object",
        ),
    ],
)
def test_evaluation_contract_rejects_invalid_merged_weight_map(
    tmp_path, payload, error_type, expected_error
):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    _write_merged_config(output_dir)
    (output_dir / "model.safetensors.index.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    merged_sft = _with_artifact(sft, output_dir, kind="merged_model")

    with pytest.raises(error_type, match=expected_error):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=merged_sft,
        )


def test_evaluation_contract_rejects_unreferenced_merged_model_shard(tmp_path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    _write_merged_config(output_dir)
    referenced = "model-00001-of-00001.safetensors"
    _write_safetensors(output_dir / referenced)
    _write_safetensors(output_dir / "model-00002-of-00002.safetensors", value=1)
    (output_dir / "model.safetensors.index.json").write_bytes(
        canonical_json_bytes({"metadata": {"total_size": 2}, "weight_map": {"layer.0": referenced}})
        + b"\n"
    )
    merged_sft = _with_artifact(sft, output_dir, kind="merged_model")

    with pytest.raises(ValueError, match="shards absent from weight_map"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=merged_sft,
        )


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
