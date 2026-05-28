import json
import os
import subprocess
from datetime import datetime, timezone
import torch

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {}

def write_run_manifest(run_dir: str, config_dict: dict, metrics: dict = None, model=None, tokenizer=None, dataset=None):
    os.makedirs(run_dir, exist_ok=True)
    
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "config": config_dict,
        "metrics": metrics or {},
        "environment": {
            "python_version": os.sys.version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        }
    }
    
    if model is not None:
        manifest["model_metadata"] = {
            "dtype": str(model.dtype),
            "parameters": model.num_parameters(),
        }
    if tokenizer is not None:
        manifest["tokenizer"] = {
            "name_or_path": getattr(tokenizer, "name_or_path", "unknown"),
            "vocab_size": len(tokenizer)
        }
    if dataset is not None:
        manifest["dataset"] = {
            "num_rows": len(dataset)
        }
        
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
