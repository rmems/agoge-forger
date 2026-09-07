import os
import re
import shutil

import torch

from ..datasets import iter_normalized_rows
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

    # Field names stay *_vram_gb for manifest compatibility; values are binary GiB
    # (1024**3), matching disk preflight — not decimal SI GB (/1e9).
    report = {
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "total_vram_gb": torch.cuda.get_device_properties(0).total_memory / BYTES_PER_GB,
        "allocated_vram_gb": torch.cuda.memory_allocated(0) / BYTES_PER_GB,
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


def directory_size_bytes(path: str) -> int:
    """Sum the apparent size of every regular file under ``path``.

    Unreadable files are logged and counted as zero rather than raising, so a
    size report never fails on a permission error or a file that vanished mid
    walk. Callers that present the total to an operator should therefore call it
    an estimate. ``os.walk`` does not descend symlinked directories.
    """
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
            entry["size_gb"] = directory_size_bytes(path) / BYTES_PER_GB
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


def validate_dataset_text_field(columns, dataset_text_field):
    """Fail fast when ``dataset_text_field`` names a column the dataset lacks.

    ``datasets.normalize_row`` normalizes every accepted row format to a
    ``text`` column, so any other value only survives when the source JSONL
    carries that column *alongside* ``text``. Without this check the mismatch
    surfaces deep inside TRL's preprocessing, long after the run has paid for
    loading the base model onto the GPU.

    Takes column *names* rather than a dataset so it can serve as the
    post-load backstop; `validate_dataset_text_field_in_source` is the stronger
    pre-load check.
    """
    if dataset_text_field not in columns:
        raise ValueError(
            f"dataset_text_field '{dataset_text_field}' is not a column in the loaded dataset "
            f"(columns: {sorted(columns)}). Row normalization always produces 'text', so either "
            f'set dataset_text_field: "text" or carry that column on every row of the source '
            f"JSONL."
        )

    logger.info(f"Validated dataset text field: {dataset_text_field}")
    return dataset_text_field


def validate_dataset_text_field_in_source(dataset_path, dataset_text_field):
    """Check ``dataset_text_field`` on every source row, before the model load.

    Checking column *names* alone is not enough. A file whose first row carries
    the field but whose later rows do not fails in two different ways, both of
    them after the model is already resident:

      * key omitted (``{"text": "b"}``) -> Arrow does not pad the column, and
        `Dataset.from_generator` raises `DatasetGenerationError`
      * key present but null (``{"text": "b", "body": null}``) -> the column
        exists, so a name-only check passes and TRL then tokenizes a `None`

    Reading the raw JSONL needs no tokenizer, so scanning every row here costs
    one pass over a text file and saves a wasted model download on both.
    """
    rows = 0
    for index, row in iter_normalized_rows(dataset_path):
        rows += 1
        value = row.get(dataset_text_field)
        if not isinstance(value, str):
            problem = (
                "is missing" if value is None else f"is not a string (got {type(value).__name__})"
            )
            # ValueError, not TypeError: this reports malformed *user data*, and
            # every other dataset/config rejection in this module and in
            # `datasets.normalize_row` raises ValueError.
            raise ValueError(  # noqa: TRY004
                f"Line {index}: dataset_text_field '{dataset_text_field}' {problem}. Row "
                f"normalization always produces 'text', so either set dataset_text_field: "
                f'"text" or carry that field on every row of {dataset_path}.'
            )

    if not rows:
        raise ValueError(f"Dataset {dataset_path} contains no rows.")

    logger.info(f"Validated dataset text field '{dataset_text_field}' across {rows} rows")
    return dataset_text_field


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
