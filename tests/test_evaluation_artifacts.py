import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from agoge_forger.eval import (
    _artifact_validation,
    _descriptor_bundle,
    _merged_model_schema,
    _tensor_schema,
)
from agoge_forger.eval.contract import validate_evaluation_contract
from agoge_forger.split_contract import canonical_json_bytes
from tests.evaluation_contract_cases import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    WRITABLE_SAFETENSORS_DTYPES,
    artifact_case,
    assert_build_rejected,
    build_contract,
    evaluation_case,
    model_provenance,
    refresh_contract_artifact_digest,
    with_artifact,
    write_adapter_config,
    write_artifact_index,
    write_complete_merged_model,
    write_merged_config,
    write_safetensors,
)

pytestmark = pytest.mark.usefixtures("cached_test_base_config")


@pytest.mark.parametrize(("dtype", "serialized_dtype"), WRITABLE_SAFETENSORS_DTYPES)
def test_torch_tensor_schema_matches_safetensors_header(tmp_path, dtype, serialized_dtype):
    tensor = torch.empty((1,), dtype=dtype)
    weights = tmp_path / f"{serialized_dtype}.safetensors"
    save_file({"weight": tensor}, weights)

    with safe_open(weights, framework="pt") as handle:
        stored = handle.get_slice("weight")
        actual = _tensor_schema.TensorSchemaEntry(tuple(stored.get_shape()), stored.get_dtype())

    assert _tensor_schema.safetensors_dtype(dtype) == serialized_dtype
    assert _tensor_schema.torch_tensor_schema_entry(tensor) == actual


def test_evaluation_contract_detects_artifact_index_mutation(tmp_path):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    contract = build_contract(tmp_path, manifest_path, base, sft)
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
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_contract(tmp_path, manifest_path, base, sft)

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
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "wrong-kind")
    if kind == "peft_adapter":
        write_merged_config(output_dir)
        write_safetensors(output_dir / "model.safetensors")
    else:
        write_adapter_config(output_dir, model_provenance())
        write_safetensors(output_dir / "adapter_model.safetensors")
    wrong_kind_sft = with_artifact(sft, output_dir, kind=kind)
    assert_build_rejected((tmp_path, manifest_path, base, wrong_kind_sft), expected_error)


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
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "unsafe-adapter")
    write_adapter_config(output_dir, model_provenance())
    unsafe_path = output_dir / unsafe_name
    unsafe_path.parent.mkdir(parents=True, exist_ok=True)
    unsafe_path.write_bytes(b"pickle-weights")
    unsafe_sft = with_artifact(sft, output_dir, kind="peft_adapter")
    assert_build_rejected((tmp_path, manifest_path, base, unsafe_sft), "unsafe serialized weights")


def test_evaluation_contract_accepts_indexed_adapter_training_state(tmp_path):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "adapter")
    (output_dir / "README.md").write_text("adapter card\n", encoding="utf-8")
    checkpoint = output_dir / "checkpoint-1"
    checkpoint.mkdir()
    write_safetensors(checkpoint / "adapter_model.safetensors", value=1)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-state")
    (checkpoint / "training_args.bin").write_bytes(b"trainer-arguments")
    complete_sft = with_artifact(sft, output_dir, kind="peft_adapter")
    contract_path = tmp_path / "eval" / "contract.json"

    written = build_contract(tmp_path, manifest_path, base, complete_sft)

    assert validate_evaluation_contract(contract_path) == written


def test_evaluation_contract_rejects_conflicting_root_adapter_safetensors(tmp_path):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "adapter")
    write_safetensors(output_dir / "model.safetensors")
    conflicting_sft = with_artifact(sft, output_dir, kind="peft_adapter")
    assert_build_rejected(
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
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_contract(tmp_path, manifest_path, base, sft)
    omitted = tmp_path / "adapter" / relative_path
    omitted.parent.mkdir(parents=True, exist_ok=True)
    omitted.write_bytes(b"not-indexed")

    with pytest.raises(ValueError, match=f"omitted from artifact index.*{omitted.name}"):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize("target", ["adapter_model.safetensors", ".", "missing"])
def test_evaluation_contract_rejects_symlink_in_artifact_bundle(tmp_path, target):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    contract_path = tmp_path / "eval" / "contract.json"
    build_contract(tmp_path, manifest_path, base, sft)
    (tmp_path / "adapter" / "artifact-link").symlink_to(target)

    with pytest.raises(ValueError, match="artifact bundle cannot contain symlinks"):
        validate_evaluation_contract(contract_path)


def test_evaluation_contract_fails_if_index_disappears_during_bundle_scan(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    assert sft.artifact is not None
    artifact_index = Path(sft.artifact.artifact_index_path)
    original_scandir = os.scandir

    def remove_index_before_scan(directory):
        artifact_index.unlink()
        return original_scandir(directory)

    monkeypatch.setattr(os, "scandir", remove_index_before_scan)
    monkeypatch.setattr(
        os,
        "supports_fd",
        (os.supports_fd - {original_scandir}) | {remove_index_before_scan},
    )
    with pytest.raises(FileNotFoundError):
        build_contract(tmp_path, manifest_path, base, sft)


@pytest.mark.parametrize(
    ("config", "expected_error"),
    [
        ({"revision": MODEL_REVISION}, "missing required base_model_name_or_path"),
        ({"base_model_name_or_path": MODEL_REPOSITORY}, "missing required revision"),
        (model_provenance("example/other-model"), "base_model_name_or_path does not match"),
        (model_provenance(f"{MODEL_REPOSITORY}/"), "base_model_name_or_path does not match"),
        (model_provenance(MODEL_REPOSITORY.upper()), "base_model_name_or_path does not match"),
        (model_provenance(revision="4" * 40), "revision does not match"),
        (model_provenance(revision=MODEL_REVISION.upper()), "revision does not match"),
        (model_provenance(revision="main"), "revision does not match"),
        (model_provenance(revision=None), "missing required revision"),
    ],
)
def test_evaluation_contract_binds_peft_adapter_to_sft_base(tmp_path, config, expected_error):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "adapter")
    write_adapter_config(output_dir, config)
    mismatched_sft = with_artifact(sft, output_dir, kind="peft_adapter")
    assert_build_rejected((tmp_path, manifest_path, base, mismatched_sft), expected_error)


def test_evaluation_contract_revalidates_peft_provenance(tmp_path):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "adapter")
    contract_path = tmp_path / "eval" / "contract.json"
    build_contract(tmp_path, manifest_path, base, sft)
    write_adapter_config(output_dir, model_provenance(revision="4" * 40))
    artifact_index = write_artifact_index(output_dir)
    refresh_contract_artifact_digest(contract_path, artifact_index)

    with pytest.raises(ValueError, match="revision does not match"):
        validate_evaluation_contract(contract_path)


@pytest.mark.parametrize("layout", ["single", "sharded"])
def test_evaluation_contract_accepts_valid_merged_model_layouts(tmp_path, layout):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged")
    write_complete_merged_model(output_dir, sharded=layout == "sharded")
    merged_sft = with_artifact(sft, output_dir, kind="merged_model")
    contract_path = tmp_path / "eval" / "contract.json"

    written = build_contract(tmp_path, manifest_path, base, merged_sft)

    assert validate_evaluation_contract(contract_path) == written
    assert written.sft.artifact is not None
    assert written.sft.artifact.kind == "merged_model"


def test_evaluation_contract_accepts_configured_merged_model_dtype(tmp_path):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged-bfloat16")
    write_complete_merged_model(output_dir, sharded=False, dtype=torch.bfloat16)
    merged_sft = with_artifact(sft, output_dir, kind="merged_model")

    written = build_contract(tmp_path, manifest_path, base, merged_sft)

    assert validate_evaluation_contract(tmp_path / "eval" / "contract.json") == written


def test_merged_tensor_schema_uses_one_verified_bundle_snapshot(tmp_path, monkeypatch):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged-sharded")
    write_complete_merged_model(output_dir, sharded=True)
    merged_sft = with_artifact(sft, output_dir, kind="merged_model")
    original_collect = _tensor_schema.collect_tensor_schema
    observed_roots: set[Path] = set()

    def collect_bundle_snapshot(path, portable, schema):
        snapshots = list(path.parent.glob("*.safetensors"))
        assert path in snapshots
        observed_roots.add(path.parent)
        original_collect(path, portable, schema)

    monkeypatch.setattr(
        _tensor_schema,
        "collect_tensor_schema",
        collect_bundle_snapshot,
    )

    build_contract(tmp_path, manifest_path, base, merged_sft)

    assert len(observed_roots) == 1
    assert next(iter(observed_roots)).parent == output_dir.parent


def test_evaluation_contract_rejects_incomplete_merged_model_state(tmp_path):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged-incomplete")
    write_merged_config(
        output_dir,
        {
            "model_type": "llama",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    write_safetensors(output_dir / "model.safetensors", keys=("unrelated.weight",))
    incomplete_sft = with_artifact(sft, output_dir, kind="merged_model")

    assert_build_rejected(
        (tmp_path, manifest_path, base, incomplete_sft),
        "merged-model tensor schema",
    )


def test_evaluation_contract_rejects_unsupported_merged_architecture_offline(tmp_path):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged-unsupported")
    write_merged_config(output_dir, {"model_type": "remote_custom_model"})
    write_safetensors(output_dir / "model.safetensors")
    unsupported_sft = with_artifact(sft, output_dir, kind="merged_model")

    assert_build_rejected(
        (tmp_path, manifest_path, base, unsupported_sft),
        "local, remote-code-disabled causal LM schema",
    )


def test_ignored_model_save_keys_are_literal_not_patterns():
    class ModelWithIgnoredKey:
        _keys_to_ignore_on_save = ("layer.0.weight",)
        all_tied_weights_keys = None

    expected = {
        "layer.0.weight": _tensor_schema.TensorSchemaEntry((1,), "F32"),
        "layerX0Yweight": _tensor_schema.TensorSchemaEntry((1,), "F32"),
    }

    _merged_model_schema.drop_omitted_model_keys(expected, ModelWithIgnoredKey())

    assert expected == {"layerX0Yweight": _tensor_schema.TensorSchemaEntry((1,), "F32")}


def test_artifact_regular_replacement_during_schema_validation_fails_closed(tmp_path, monkeypatch):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged-race")
    write_complete_merged_model(output_dir, sharded=False)
    weights_path = output_dir / "model.safetensors"
    replacement = tmp_path / "identical-model.safetensors"
    shutil.copy2(weights_path, replacement)
    merged_sft = with_artifact(sft, output_dir, kind="merged_model")
    original_layout = _artifact_validation._require_artifact_layout
    replaced = False

    def replace_before_layout(context, indexed, provenance):
        nonlocal replaced
        snapshot_paths = [path for _, path in indexed.values()]
        assert all(not path.is_relative_to(output_dir) for path in snapshot_paths)
        snapshot_roots = {
            path.parents[len(portable.parts) - 1] for portable, (_, path) in indexed.items()
        }
        assert len(snapshot_roots) == 1
        os.replace(replacement, weights_path)
        replaced = True
        return original_layout(context, indexed, provenance)

    monkeypatch.setattr(
        _artifact_validation,
        "_require_artifact_layout",
        replace_before_layout,
    )

    assert_build_rejected(
        (tmp_path, manifest_path, base, merged_sft),
        "artifact bundle changed",
    )
    assert replaced


def test_artifact_validation_reports_unsupported_descriptor_traversal(tmp_path, monkeypatch):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged-platform")
    write_complete_merged_model(output_dir, sharded=False)
    merged_sft = with_artifact(sft, output_dir, kind="merged_model")
    monkeypatch.setattr(
        _descriptor_bundle.os,
        "supports_fd",
        _descriptor_bundle.os.supports_fd - {_descriptor_bundle.os.scandir},
    )

    assert_build_rejected(
        (tmp_path, manifest_path, base, merged_sft),
        "cannot be validated safely on this platform",
    )


def test_artifact_weight_symlink_swap_during_schema_validation_fails_closed(tmp_path, monkeypatch):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged-symlink-race")
    write_complete_merged_model(output_dir, sharded=False)
    weights_path = output_dir / "model.safetensors"
    identical = tmp_path / "identical-model.safetensors"
    shutil.copy2(weights_path, identical)
    merged_sft = with_artifact(sft, output_dir, kind="merged_model")
    original_layout = _artifact_validation._require_artifact_layout
    replaced = False

    def replace_before_layout(context, indexed, provenance):
        nonlocal replaced
        weights_path.unlink()
        weights_path.symlink_to(identical)
        replaced = True
        return original_layout(context, indexed, provenance)

    monkeypatch.setattr(
        _artifact_validation,
        "_require_artifact_layout",
        replace_before_layout,
    )

    assert_build_rejected(
        (tmp_path, manifest_path, base, merged_sft),
        "artifact bundle cannot contain symlinks|artifact bundle changed",
    )
    assert replaced


def test_evaluation_contract_accepts_omitted_tied_embedding_alias(tmp_path):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged-tied")
    write_complete_merged_model(output_dir, sharded=False, tie_word_embeddings=True)
    merged_sft = with_artifact(sft, output_dir, kind="merged_model")

    written = build_contract(tmp_path, manifest_path, base, merged_sft)

    assert validate_evaluation_contract(tmp_path / "eval" / "contract.json") == written


@pytest.mark.parametrize(
    "mutation", ["missing", "unexpected", "wrong-shape", "wrong-dtype", "config-drift"]
)
def test_evaluation_contract_rejects_merged_model_tensor_schema_drift(tmp_path, mutation):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, f"merged-{mutation}")
    write_complete_merged_model(output_dir, sharded=False)
    weights_path = output_dir / "model.safetensors"
    if mutation == "config-drift":
        config = json.loads((output_dir / "config.json").read_bytes())
        config["num_hidden_layers"] = 2
        (output_dir / "config.json").write_bytes(canonical_json_bytes(config) + b"\n")
    else:
        tensors = load_file(weights_path)
        first_key = next(iter(tensors))
        if mutation == "missing":
            tensors.pop(first_key)
        elif mutation == "unexpected":
            tensors["unexpected.weight"] = torch.zeros((1, 1))
        elif mutation == "wrong-shape":
            tensors[first_key] = torch.zeros((1,))
        else:
            tensors[first_key] = torch.zeros(tensors[first_key].shape, dtype=torch.uint8)
        save_file(tensors, weights_path)
    drifted_sft = with_artifact(sft, output_dir, kind="merged_model")

    assert_build_rejected(
        (tmp_path, manifest_path, base, drifted_sft),
        "merged-model tensor schema",
    )
