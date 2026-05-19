import os
import json
from agoge_forger.manifests import write_run_manifest

def test_write_manifest(tmp_path):
    run_dir = str(tmp_path / "test_run")
    write_run_manifest(run_dir, {"a": 1}, {"vram": 10})
    manifest_file = os.path.join(run_dir, "manifest.json")
    assert os.path.exists(manifest_file)
    with open(manifest_file) as f:
        data = json.load(f)
        assert data["config"]["a"] == 1
        assert data["metrics"]["vram"] == 10
        assert "git" in data
        assert "environment" in data
