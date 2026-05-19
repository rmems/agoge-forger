import json
import os
from datetime import datetime

def write_run_manifest(run_dir: str, config_dict: dict, metrics: dict = None):
    os.makedirs(run_dir, exist_ok=True)
    manifest = {
        "timestamp": datetime.utcnow().isoformat(),
        "config": config_dict,
        "metrics": metrics or {}
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
