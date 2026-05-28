import pytest
import os
from agoge_forger.artifacts.safetensors_io import write_artifact_index, assert_no_unsafe_weight_bins

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

def test_no_bin_outputs_when_safe_serialization_required(tmp_path):
    out_dir = tmp_path / "safe_dir"
    out_dir.mkdir()
    
    bin_file = out_dir / "pytorch_model.bin"
    bin_file.touch()
    
    with pytest.raises(RuntimeError, match="Unsafe weight binaries found"):
        assert_no_unsafe_weight_bins(str(out_dir))
