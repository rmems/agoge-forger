import os
import shutil
import torch
import re
from ..logging import logger

BYTES_PER_GB = 1024 ** 3

def check_cuda_available(required=True):
    if not torch.cuda.is_available():
        if required:
            raise RuntimeError("CUDA is required but not available.")
        else:
            logger.warning("CUDA is not available.")
            return False
    return True

def get_gpu_report():
    if not torch.cuda.is_available():
        return {}
    
    report = {
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "total_vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "allocated_vram_gb": torch.cuda.memory_allocated(0) / 1e9,
        "bf16_supported": torch.cuda.is_bf16_supported()
    }
    return report

def estimate_training_risk(config, gpu_report):
    if not gpu_report:
        return
        
    vram = gpu_report.get("total_vram_gb", 0)
    
    if vram <= 16.5:
        if not config.quantization.load_in_4bit:
            logger.warning("RISK: Training on <= 16GB VRAM without load_in_4bit is highly likely to OOM.")
        
        if config.training.batch_size > 1:
            logger.warning("RISK: Batch size > 1 on 16GB VRAM may cause OOM. Consider gradient_accumulation_steps instead.")
            
        if config.training.max_seq_length > 2048:
            logger.warning("RISK: max_seq_length > 2048 on 16GB VRAM may cause OOM.")


def _directory_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                logger.warning(f"Could not read file size for {file_path}")
    return total


def collect_disk_pressure_report(config, monitored_paths=None):
    monitored = monitored_paths or [
        os.path.expanduser("~/.unsloth/studio/outputs"),
        os.path.expanduser("~/.cache/huggingface"),
    ]
    output_root = os.path.abspath(config.output_dir)
    disk = shutil.disk_usage(output_root)
    report = {
        "output_dir": output_root,
        "free_gb": disk.free / BYTES_PER_GB,
        "warning_threshold_gb": config.runtime.disk_free_warning_gb,
        "checkpoint_buffer_gb": config.runtime.checkpoint_disk_buffer_gb,
        "paths": [],
    }

    for path in monitored:
        entry = {"path": path, "exists": os.path.exists(path), "size_gb": 0.0}
        if entry["exists"]:
            entry["size_gb"] = _directory_size_bytes(path) / BYTES_PER_GB
        report["paths"].append(entry)

    return report


def warn_on_disk_pressure(config, monitored_paths=None):
    report = collect_disk_pressure_report(config, monitored_paths=monitored_paths)
    free_gb = report["free_gb"]

    for entry in report["paths"]:
        if entry["exists"]:
            logger.info(f"Disk preflight: {entry['path']} currently uses {entry['size_gb']:.2f} GB")
        else:
            logger.info(f"Disk preflight: {entry['path']} does not exist yet")

    if free_gb < report["warning_threshold_gb"]:
        logger.warning(
            f"Disk preflight: only {free_gb:.2f} GB free under {report['output_dir']}; "
            f"configured warning threshold is {report['warning_threshold_gb']:.2f} GB."
        )

    if free_gb < report["checkpoint_buffer_gb"]:
        logger.warning(
            f"Disk preflight: free space {free_gb:.2f} GB is below the checkpoint buffer "
            f"of {report['checkpoint_buffer_gb']:.2f} GB. Checkpoint saves may fail."
        )

    return report

def validate_lora_targets_exist(model, lora_config):
    mode = getattr(lora_config, "target_modules_mode", "auto_common")
    requested_targets = lora_config.target_modules
    
    if mode == "discover_required" and not requested_targets:
        raise ValueError("target_modules_mode is discover_required but no target_modules were provided.")
        
    model_modules = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        model_modules.add(leaf)
        
    valid_targets = []
    
    if mode == "auto_common":
        common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        candidates = requested_targets if requested_targets else common
        common_set = set(common)
        valid_targets = [t for t in candidates if t in common_set and t in model_modules]
        if not valid_targets:
            logger.warning("No common projection targets found. Proceeding with caution.")
    else:
        for t in requested_targets:
            # simple check if leaf target exists, or if regex matches
            found = False
            for name, _ in model.named_modules():
                if t in name or re.search(t, name):
                    found = True
                    break
            if found:
                valid_targets.append(t)
            else:
                logger.warning(f"Target module '{t}' not found in model.")
                if mode == "explicit":
                    raise ValueError(f"Explicit target module '{t}' does not exist in the model.")
                    
    if not valid_targets:
        raise ValueError("No valid LoRA target modules found or configured.")
        
    logger.info(f"Validated target modules: {valid_targets}")
    return valid_targets
