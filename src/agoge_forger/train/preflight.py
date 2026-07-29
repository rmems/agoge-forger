import os
import re
import shutil

import torch

from ..logging import logger

BYTES_PER_GB = 1024**3
COMMON_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    return report


def estimate_training_risk(config, gpu_report):
    if not gpu_report:
        return

    vram = gpu_report.get("total_vram_gb", 0)

    if vram <= 16.5:
        if not config.quantization.load_in_4bit:
            logger.warning(
                "RISK: Training on <= 16GB VRAM without load_in_4bit is highly likely to OOM."
            )

        if config.training.batch_size > 1:
            logger.warning(
                "RISK: Batch size > 1 on 16GB VRAM may cause OOM. Consider gradient_accumulation_steps instead."
            )

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
    """Report disk usage; override ``monitored_paths`` to replace default Unsloth/HF cache roots."""
    if monitored_paths is None:
        monitored_paths = [
            os.path.expanduser("~/.unsloth/studio/outputs"),
            os.path.expanduser("~/.cache/huggingface"),
        ]
    output_root = os.path.abspath(config.output_dir)
    disk_path = output_root
    if not os.path.exists(disk_path):
        parent = os.path.dirname(disk_path) or "."
        disk_path = parent if os.path.exists(parent) else "."
    disk = shutil.disk_usage(disk_path)
    report = {
        "output_dir": output_root,
        "free_gb": disk.free / BYTES_PER_GB,
        "warning_threshold_gb": config.runtime.disk_free_warning_gb,
        "checkpoint_buffer_gb": config.runtime.checkpoint_disk_buffer_gb,
        "paths": [],
    }

    for path in monitored_paths:
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


def _collect_model_leaf_modules(model):
    modules = set()
    for name, _ in model.named_modules():
        modules.add(name.split(".")[-1])
    return modules


def _module_matches_target(model, target):
    try:
        pattern = re.compile(target)
    except re.error:
        logger.warning(
            f"Invalid regex in LoRA target '{target}'; using literal substring match only."
        )
        pattern = None

    for name, _ in model.named_modules():
        if target in name:
            return True
        if pattern and pattern.search(name):
            return True
    return False


def _resolve_auto_common_targets(requested_targets, model_modules):
    candidates = requested_targets if requested_targets else COMMON_LORA_TARGETS
    common_set = set(COMMON_LORA_TARGETS)
    valid_targets = [t for t in candidates if t in common_set and t in model_modules]
    if not valid_targets:
        logger.warning("No common projection targets found. Proceeding with caution.")
    return valid_targets


def _resolve_explicit_targets(requested_targets, model, mode):
    valid_targets = []
    for target in requested_targets:
        if _module_matches_target(model, target):
            valid_targets.append(target)
            continue
        logger.warning(f"Target module '{target}' not found in model.")
        if mode == "explicit":
            raise ValueError(f"Explicit target module '{target}' does not exist in the model.")
    return valid_targets


def validate_lora_targets_exist(model, lora_config):
    mode = getattr(lora_config, "target_modules_mode", "auto_common")
    requested_targets = lora_config.target_modules

    if mode == "discover_required" and not requested_targets:
        raise ValueError(
            "target_modules_mode is discover_required but no target_modules were provided."
        )

    model_modules = _collect_model_leaf_modules(model)
    if mode == "auto_common":
        valid_targets = _resolve_auto_common_targets(requested_targets, model_modules)
    else:
        valid_targets = _resolve_explicit_targets(requested_targets, model, mode)

    if not valid_targets:
        raise ValueError("No valid LoRA target modules found or configured.")

    logger.info(f"Validated target modules: {valid_targets}")
    return valid_targets
