import os

import pytest

from agoge_forger.artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index


def test_artifact_index_hashes_file(tmp_path):
    out_dir = tmp_path / "test_out"
    out_dir.mkdir()

    test_file = out_dir / "test.txt"
    test_file.write_text("hello world")

    index_path = write_artifact_index(str(out_dir))
    assert os.path.exists(index_path)

    import json

    with open(index_path) as f:
        data = json.load(f)
        assert data["output_dir"] == str(out_dir)
        assert len(data["artifacts"]) == 1
        assert data["artifacts"][0]["file"] == "test.txt"
        assert data["artifacts"][0]["size_bytes"] == 11
        assert data["artifacts"][0]["sha256"] != "unknown"


def test_artifact_index_excludes_only_its_root_index(tmp_path):
    out_dir = tmp_path / "test_out"
    nested_dir = out_dir / "nested"
    nested_dir.mkdir(parents=True)
    nested_index = nested_dir / "artifact_index.json"
    nested_index.write_text('{"nested": true}\n')

    index_path = write_artifact_index(str(out_dir))

    import json

    with open(index_path) as f:
        data = json.load(f)
    assert [artifact["file"] for artifact in data["artifacts"]] == [
        os.path.join("nested", "artifact_index.json")
    ]


def test_artifact_index_writes_producer_provenance(tmp_path):
    out_dir = tmp_path / "test_out"
    out_dir.mkdir()
    (out_dir / "model.safetensors").write_bytes(b"weights")
    provenance = {
        "base_model_name_or_path": "example/base",
        "revision": "a" * 40,
        "training_split_manifest_sha256": "b" * 64,
        "training_split_name": "train",
        "training_split_sha256": "c" * 64,
    }

    index_path = write_artifact_index(str(out_dir), producer_provenance=provenance)

    import json

    with open(index_path) as handle:
        assert json.load(handle)["producer_provenance"] == provenance


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
