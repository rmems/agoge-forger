from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file, save_file

from agoge_forger.eval.contract import ArtifactIndexReference, build_evaluation_contract
from agoge_forger.split_contract import canonical_json_bytes, sha256_file
from tests.evaluation_contract_cases import (
    artifact_case,
    assert_build_rejected,
    evaluation_case,
    model_provenance,
    with_artifact,
    write_invalid_merged_layout,
)
from tests.peft_adapter_fixtures import write_complete_adapter_model

pytestmark = pytest.mark.usefixtures("cached_test_base_config")


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "wrong-shape", "wrong-dtype"])
def test_evaluation_contract_rejects_adapter_tensor_schema_drift(tmp_path, mutation):
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, f"adapter-{mutation}")
    write_complete_adapter_model(output_dir)
    weights_path = output_dir / "adapter_model.safetensors"
    tensors = load_file(weights_path)
    first_key = next(iter(tensors))
    if mutation == "missing":
        tensors.pop(first_key)
    elif mutation == "unexpected":
        tensors["unrelated.weight"] = torch.zeros((1, 1))
    elif mutation == "wrong-shape":
        tensors[first_key] = torch.zeros((1,))
    else:
        tensors[first_key] = torch.zeros(tensors[first_key].shape, dtype=torch.uint8)
    save_file(tensors, weights_path)
    drifted_sft = with_artifact(sft, output_dir, kind="peft_adapter")

    assert_build_rejected(
        (tmp_path, manifest_path, base, drifted_sft),
        "PEFT adapter tensor schema",
    )


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
    manifest_path, base, sft, output_dir = artifact_case(tmp_path, "merged")
    write_invalid_merged_layout(output_dir, variant)
    provenance = {
        "provenance-missing": False,
        "provenance-repository": model_provenance("example/other-model"),
        "provenance-revision": model_provenance(revision="4" * 40),
    }.get(variant)
    merged_sft = with_artifact(sft, output_dir, kind="merged_model", provenance=provenance)
    assert_build_rejected((tmp_path, manifest_path, base, merged_sft), expected_error, error_type)


def test_evaluation_contract_rejects_contract_inside_artifact_bundle(tmp_path):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
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
    manifest_path, _, base, sft = evaluation_case(tmp_path)
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

    assert_build_rejected((tmp_path, manifest_path, base, escaped_sft), "must stay relative")
    assert not contract_path.exists()


def test_evaluation_contract_revalidates_copied_arms_before_writing(tmp_path):
    manifest_path, _, base, sft = evaluation_case(tmp_path)
    invalid_base = base.model_copy(update={"context_window": 0})
    invalid_sft = sft.model_copy(update={"context_window": 0})
    contract_path = tmp_path / "eval" / "contract.json"

    assert_build_rejected(
        (tmp_path, manifest_path, invalid_base, invalid_sft),
        "context_window",
        ValidationError,
    )
    assert not contract_path.exists()
