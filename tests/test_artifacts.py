import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from agoge_forger.artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index
from agoge_forger.eval import ArtifactProducerProvenance
from agoge_forger.eval.contract import ArtifactValidationContext, require_artifact_index
from agoge_forger.split_contract import sha256_file
from tests.evaluation_contract_cases import model_provenance
from tests.peft_adapter_fixtures import write_complete_adapter_model


def test_artifact_index_hashes_file(tmp_path):
    out_dir = tmp_path / "test_out"
    out_dir.mkdir()

    test_file = out_dir / "test.txt"
    test_file.write_text("hello world")

    index_path = write_artifact_index(str(out_dir))
    assert os.path.exists(index_path)

    with open(index_path) as f:
        data = json.load(f)
        assert data["output_dir"] == str(out_dir)
        assert len(data["artifacts"]) == 1
        assert data["artifacts"][0]["file"] == "test.txt"
        assert data["artifacts"][0]["size_bytes"] == 11
        assert data["artifacts"][0]["sha256"] != "unknown"
        assert "producer_provenance" not in data


def test_write_artifact_index_provenance_is_required_by_eval_validation(
    tmp_path, cached_test_base_config
):
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    write_complete_adapter_model(output_dir)

    provenance = ArtifactProducerProvenance.model_validate(model_provenance())
    context = ArtifactValidationContext(
        kind="peft_adapter",
        model_repository=provenance.base_model_name_or_path,
        model_revision=provenance.revision,
        split_manifest_sha256=provenance.training_split_manifest_sha256,
        train_split_sha256=provenance.training_split_sha256,
    )

    index_path = Path(write_artifact_index(str(output_dir), producer_provenance=provenance))
    require_artifact_index(index_path, sha256_file(index_path), context)

    index_path.unlink()
    index_path = Path(write_artifact_index(str(output_dir)))
    with pytest.raises(
        ValueError, match="peft_adapter artifact index requires producer_provenance"
    ):
        require_artifact_index(index_path, sha256_file(index_path), context)

    mismatched = replace(context, model_revision="9" * 40)
    index_path.unlink()
    index_path = Path(write_artifact_index(str(output_dir), producer_provenance=provenance))
    with pytest.raises(ValueError, match="does not match the contracted"):
        require_artifact_index(index_path, sha256_file(index_path), mismatched)


def test_no_bin_outputs_when_safe_serialization_required(tmp_path):
    out_dir = tmp_path / "safe_dir"
    out_dir.mkdir()

    bin_file = out_dir / "pytorch_model.bin"
    bin_file.touch()

    with pytest.raises(RuntimeError, match="Unsafe weight binaries found"):
        assert_no_unsafe_weight_bins(str(out_dir))


def test_assert_no_unsafe_weight_bins_ignores_trainer_state_pt(tmp_path):
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"trainer-state")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    assert_no_unsafe_weight_bins(str(tmp_path))


def test_assert_no_unsafe_weight_bins_detects_bin_in_checkpoint_subdir(tmp_path):
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.bin").write_bytes(b"unsafe")
    with pytest.raises(RuntimeError, match="Unsafe weight binaries found"):
        assert_no_unsafe_weight_bins(str(tmp_path))
