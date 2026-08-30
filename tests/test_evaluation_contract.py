import json
import os
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

MODEL_REPOSITORY = "example/base-model"
MODEL_REVISION = "abcdef0123456789abcdef0123456789abcdef01"
_BuildArgs = tuple[Path, Path, EvaluationArm, EvaluationArm]


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


def _write_artifact_index(output_dir: Path, provenance=None) -> Path:
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


def _with_artifact(sft: EvaluationArm, output_dir: Path, *, kind, provenance=None):
    if provenance is None and kind == "merged_model":
        provenance = _provenance()
    index_provenance = None if provenance is False else provenance
    artifact_index = _write_artifact_index(output_dir, index_provenance)
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


def _provenance(
    repository: object = MODEL_REPOSITORY, revision: object = MODEL_REVISION
) -> dict[str, object]:
    return {"base_model_name_or_path": repository, "revision": revision}


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
    _write_adapter_config(output_dir, _provenance())
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


def _case(tmp_path: Path):
    manifest_path, manifest = _frozen_manifest(tmp_path)
    digest = logical_task_set_sha256(held_out_task_ids(manifest))
    base, sft = _arms(digest, tmp_path)
    return manifest_path, manifest, base, sft


def _artifact_case(tmp_path: Path, directory: str):
    manifest_path, _, base, sft = _case(tmp_path)
    output_dir = tmp_path / directory
    output_dir.mkdir(exist_ok=True)
    return manifest_path, base, sft, output_dir


def _build(tmp_path: Path, manifest_path: Path, base: EvaluationArm, sft: EvaluationArm):
    return build_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=tmp_path / "eval" / "contract.json",
        base=base,
        sft=sft,
    )


def _assert_build_rejected(
    build_args: _BuildArgs,
    expected_error: str,
    error_type: type[BaseException] = ValueError,
) -> None:
    with pytest.raises(error_type, match=expected_error):
        _build(*build_args)


def _write_shard_index(output_dir: Path, payload: dict[str, object]) -> None:
    (output_dir / "model.safetensors.index.json").write_bytes(canonical_json_bytes(payload) + b"\n")


def _paths_absent(*paths: Path) -> bool:
    return all(not path.exists() for path in paths)


def _write_invalid_merged_layout(output_dir: Path, variant: str) -> None:
    _write_merged_config(output_dir)
    single = "model-00001-of-00001.safetensors"
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    if variant.startswith("provenance-"):
        _write_safetensors(output_dir / "model.safetensors")
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
        _write_safetensors(output_dir / first, keys=first_keys)
        _write_safetensors(output_dir / second, keys=second_keys)
        _write_shard_index(output_dir, {"metadata": {"total_size": 2}, "weight_map": weight_map})
        return

    shard_references = {
        "shard-bin": "model-00001-of-00001.bin",
        "shard-noncanonical": "./model-00001-of-00001.safetensors",
        "shard-unindexed": "model-00001-of-00001.safetensors",
    }
    if variant in shard_references:
        _write_shard_index(
            output_dir,
            {"metadata": {"total_size": 1}, "weight_map": {"layer.0": shard_references[variant]}},
        )
    elif variant == "ambiguous":
        _write_safetensors(output_dir / "model.safetensors")
        _write_shard_index(
            output_dir,
            {"metadata": {"total_size": 1}, "weight_map": {"layer.0": single}},
        )
    elif variant == "duplicate-map-key":
        _write_safetensors(output_dir / single)
        (output_dir / "model.safetensors.index.json").write_text(
            f'{{"metadata":{{}},"weight_map":{{"layer.0":"{single}","layer.0":"{single}"}}}}\n',
            encoding="utf-8",
        )
    elif variant == "unreferenced-shard":
        _write_safetensors(output_dir / single)
        _write_safetensors(output_dir / second, value=1)
        _write_shard_index(
            output_dir, {"metadata": {"total_size": 2}, "weight_map": {"layer.0": single}}
        )
    else:
        invalid_indices = {
            "empty-map-missing": {},
            "empty-map-list": {"weight_map": []},
            "non-string-shard": {"metadata": {}, "weight_map": {"layer.0": 7}},
            "missing-metadata": {"weight_map": {"layer.0": single}},
        }
        _write_shard_index(output_dir, invalid_indices[variant])


def test_schema_only_contract_consumes_frozen_held_out_manifest(tmp_path):
    manifest_path, manifest, base, sft = _case(tmp_path)
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
    assert validated.sft.artifact is not None
    assert not Path(validated.sft.artifact.artifact_index_path).is_absolute()
    assert _paths_absent(contract_path.parent / "base", contract_path.parent / "sft")


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
    _, manifest, base, sft = _case(tmp_path)
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
    manifest_path, _, base, sft = _case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    _build(tmp_path, manifest_path, base, sft)

    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="split-manifest SHA-256 mismatch"):
        validate_evaluation_contract(contract_path)


def test_evaluation_contract_refuses_overwrite(tmp_path):
    manifest_path, _, base, sft = _case(tmp_path)
    _build(tmp_path, manifest_path, base, sft)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(tmp_path, manifest_path, base, sft)


def test_evaluation_contract_requires_immutable_model_and_tokenizer_revisions(tmp_path):
    _, _, base, _ = _case(tmp_path)

    with pytest.raises(ValidationError, match="model_revision"):
        base.model_copy(update={"model_revision": "main"}).model_validate(
            base.model_copy(update={"model_revision": "main"}).model_dump()
        )
    with pytest.raises(ValidationError, match="tokenizer_revision"):
        base.model_copy(update={"tokenizer_revision": "latest"}).model_validate(
            base.model_copy(update={"tokenizer_revision": "latest"}).model_dump()
        )


def test_evaluation_contract_detects_artifact_index_mutation(tmp_path):
    manifest_path, _, base, sft = _case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    contract = _build(tmp_path, manifest_path, base, sft)
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
    manifest_path, _, base, sft = _case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    _build(tmp_path, manifest_path, base, sft)

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
    manifest_path, base, sft, output_dir = _artifact_case(tmp_path, "wrong-kind")
    if kind == "peft_adapter":
        _write_merged_config(output_dir)
        _write_safetensors(output_dir / "model.safetensors")
    else:
        _write_adapter_config(output_dir, _provenance())
        _write_safetensors(output_dir / "adapter_model.safetensors")
    wrong_kind_sft = _with_artifact(sft, output_dir, kind=kind)
    _assert_build_rejected((tmp_path, manifest_path, base, wrong_kind_sft), expected_error)


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
    manifest_path, base, sft, output_dir = _artifact_case(tmp_path, "unsafe-adapter")
    _write_adapter_config(output_dir, _provenance())
    unsafe_path = output_dir / unsafe_name
    unsafe_path.parent.mkdir(parents=True, exist_ok=True)
    unsafe_path.write_bytes(b"pickle-weights")
    unsafe_sft = _with_artifact(sft, output_dir, kind="peft_adapter")
    _assert_build_rejected((tmp_path, manifest_path, base, unsafe_sft), "unsafe serialized weights")


def test_evaluation_contract_accepts_indexed_adapter_training_state(tmp_path):
    manifest_path, base, sft, output_dir = _artifact_case(tmp_path, "adapter")
    (output_dir / "README.md").write_text("adapter card\n", encoding="utf-8")
    checkpoint = output_dir / "checkpoint-1"
    checkpoint.mkdir()
    _write_safetensors(checkpoint / "adapter_model.safetensors", value=1)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-state")
    (checkpoint / "training_args.bin").write_bytes(b"trainer-arguments")
    complete_sft = _with_artifact(sft, output_dir, kind="peft_adapter")
    contract_path = tmp_path / "eval" / "contract.json"

    written = _build(tmp_path, manifest_path, base, complete_sft)

    assert validate_evaluation_contract(contract_path) == written


def test_evaluation_contract_rejects_conflicting_root_adapter_safetensors(tmp_path):
    manifest_path, base, sft, output_dir = _artifact_case(tmp_path, "adapter")
    _write_safetensors(output_dir / "model.safetensors")
    conflicting_sft = _with_artifact(sft, output_dir, kind="peft_adapter")
    _assert_build_rejected(
        (tmp_path, manifest_path, base, conflicting_sft),
        "only adapter_model.safetensors weights",
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
    manifest_path, _, base, sft = _case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    _build(tmp_path, manifest_path, base, sft)
    omitted = tmp_path / "adapter" / relative_path
    omitted.parent.mkdir(parents=True, exist_ok=True)
    omitted.write_bytes(b"not-indexed")

    with pytest.raises(ValueError, match=f"omitted from artifact index.*{omitted.name}"):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize("target", ["adapter_model.safetensors", ".", "missing"])
def test_evaluation_contract_rejects_symlink_in_artifact_bundle(tmp_path, target):
    manifest_path, _, base, sft = _case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    _build(tmp_path, manifest_path, base, sft)
    (tmp_path / "adapter" / "artifact-link").symlink_to(target)

    with pytest.raises(ValueError, match="artifact bundle cannot contain symlinks"):
        validate_evaluation_contract(contract_path)


def test_evaluation_contract_fails_if_index_disappears_during_bundle_scan(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, base, sft = _case(tmp_path)
    assert sft.artifact is not None
    artifact_index = Path(sft.artifact.artifact_index_path)
    original_scandir = os.scandir

    def remove_index_before_scan(directory):
        artifact_index.unlink()
        return original_scandir(directory)

    monkeypatch.setattr(os, "scandir", remove_index_before_scan)
    with pytest.raises(FileNotFoundError):
        _build(tmp_path, manifest_path, base, sft)


@pytest.mark.parametrize(
    ("config", "expected_error"),
    [
        ({"revision": MODEL_REVISION}, "missing required base_model_name_or_path"),
        ({"base_model_name_or_path": MODEL_REPOSITORY}, "missing required revision"),
        (_provenance("example/other-model"), "base_model_name_or_path does not match"),
        (_provenance(f"{MODEL_REPOSITORY}/"), "base_model_name_or_path does not match"),
        (_provenance(MODEL_REPOSITORY.upper()), "base_model_name_or_path does not match"),
        (_provenance(revision="4" * 40), "revision does not match"),
        (_provenance(revision=MODEL_REVISION.upper()), "revision does not match"),
        (_provenance(revision="main"), "revision does not match"),
        (_provenance(revision=None), "missing required revision"),
    ],
)
def test_evaluation_contract_binds_peft_adapter_to_sft_base(tmp_path, config, expected_error):
    manifest_path, base, sft, output_dir = _artifact_case(tmp_path, "adapter")
    _write_adapter_config(output_dir, config)
    mismatched_sft = _with_artifact(sft, output_dir, kind="peft_adapter")
    _assert_build_rejected((tmp_path, manifest_path, base, mismatched_sft), expected_error)


def test_evaluation_contract_revalidates_peft_provenance(tmp_path):
    manifest_path, base, sft, output_dir = _artifact_case(tmp_path, "adapter")
    contract_path = tmp_path / "eval" / "contract.json"
    _build(tmp_path, manifest_path, base, sft)
    _write_adapter_config(output_dir, _provenance(revision="4" * 40))
    artifact_index = _write_artifact_index(output_dir)
    _refresh_contract_artifact_digest(contract_path, artifact_index)

    with pytest.raises(ValueError, match="revision does not match"):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize("layout", ["single", "sharded"])
def test_evaluation_contract_accepts_valid_merged_model_layouts(tmp_path, layout):
    manifest_path, base, sft, output_dir = _artifact_case(tmp_path, "merged")
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
        _write_shard_index(
            output_dir,
            {
                "metadata": {"total_size": 2},
                "weight_map": {
                    "layer.0.weight": first,
                    "layer.0.bias": first,
                    "layer.1.weight": second,
                },
            },
        )
    merged_sft = _with_artifact(sft, output_dir, kind="merged_model")
    contract_path = tmp_path / "eval" / "contract.json"

    written = _build(tmp_path, manifest_path, base, merged_sft)

    assert validate_evaluation_contract(contract_path) == written
    assert written.sft.artifact is not None
    assert written.sft.artifact.kind == "merged_model"


@pytest.mark.parametrize(
    ("variant", "error_type", "expected_error"),
    [
        ("tensor-missing", ValueError, "names tensors absent from shards"),
        ("tensor-extra", ValueError, "tensors absent from weight_map"),
        ("tensor-duplicate", ValueError, "tensor keys occur in multiple"),
        ("tensor-misplaced", ValueError, "assigns tensors to wrong shards"),
        ("shard-bin", ValueError, "must reference safetensors shards only"),
        ("shard-noncanonical", ValueError, "shard paths must be canonical"),
        ("shard-unindexed", ValueError, "references unindexed shards"),
        ("ambiguous", ValueError, "exactly one of model.safetensors"),
        ("duplicate-map-key", ValueError, "invalid merged-model shard index"),
        ("empty-map-missing", ValueError, "requires a non-empty weight_map"),
        ("empty-map-list", ValueError, "requires a non-empty weight_map"),
        ("non-string-shard", TypeError, "shard paths must be strings"),
        ("missing-metadata", TypeError, "requires a metadata object"),
        ("unreferenced-shard", ValueError, "shards absent from weight_map"),
        ("provenance-missing", ValueError, "requires producer_provenance"),
        ("provenance-repository", ValueError, "base_model_name_or_path does not match"),
        ("provenance-revision", ValueError, "revision does not match"),
    ],
)
def test_evaluation_contract_rejects_invalid_merged_model_layout(
    tmp_path, variant, error_type, expected_error
):
    manifest_path, base, sft, output_dir = _artifact_case(tmp_path, "merged")
    _write_invalid_merged_layout(output_dir, variant)
    provenance = {
        "provenance-missing": False,
        "provenance-repository": _provenance("example/other-model"),
        "provenance-revision": _provenance(revision="4" * 40),
    }.get(variant)
    merged_sft = _with_artifact(sft, output_dir, kind="merged_model", provenance=provenance)
    _assert_build_rejected((tmp_path, manifest_path, base, merged_sft), expected_error, error_type)


def test_evaluation_contract_rejects_contract_inside_artifact_bundle(tmp_path):
    manifest_path, _, base, sft = _case(tmp_path)
    contract_path = tmp_path / "adapter" / "contract.json"

    with pytest.raises(ValueError, match="cannot be written inside its artifact bundle"):
        build_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=contract_path,
            base=base,
            sft=sft,
        )
    assert not contract_path.exists()


def test_evaluation_contract_rejects_artifact_index_path_escape(tmp_path):
    manifest_path, _, base, sft = _case(tmp_path)
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

    _assert_build_rejected((tmp_path, manifest_path, base, escaped_sft), "must stay relative")
    assert not contract_path.exists()


def test_evaluation_contract_revalidates_copied_arms_before_writing(tmp_path):
    manifest_path, _, base, sft = _case(tmp_path)
    invalid_base = base.model_copy(update={"context_window": 0})
    invalid_sft = sft.model_copy(update={"context_window": 0})
    contract_path = tmp_path / "eval" / "contract.json"

    _assert_build_rejected(
        (tmp_path, manifest_path, invalid_base, invalid_sft),
        "context_window",
        ValidationError,
    )
    assert not contract_path.exists()
