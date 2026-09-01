import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import torch
from huggingface_hub.utils import HFValidationError, validate_repo_id
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index
from ..datasets import load_jsonl_dataset
from ..eval import ArtifactProducerProvenance
from ..logging import logger
from ..manifests import write_run_manifest
from ..models.load import load_base_model
from ..split_loaders import FrozenSplitBinding, bind_frozen_split, load_frozen_dataset
from .checkpoints import resolve_resume_checkpoint
from .preflight import (
    BYTES_PER_GB,
    check_cuda_available,
    estimate_training_risk,
    get_gpu_report,
    validate_dataset_text_field,
    validate_dataset_text_field_in_source,
    validate_lora_targets_exist,
    warn_on_disk_pressure,
)


@dataclass(frozen=True)
class _TrainingFinalization:
    out_dir: str
    gpu_report: Any
    producer_provenance: ArtifactProducerProvenance | None


def _build_training_args(config, out_dir):
    """Map the experiment config onto TRL's `SFTConfig`.

    The YAML/pydantic keys are the stable surface; the TRL names are not.
    Translate at this boundary only, so upstream renames never leak into
    `configs/*.yaml` or `ExperimentConfig`:

      * `training.max_seq_length` -> `SFTConfig.max_length`
      * `dataset_text_field`      -> `SFTConfig.dataset_text_field`

    `runtime.save_safetensors` is deliberately *not* forwarded: Transformers 5
    removed `save_safetensors` from `TrainingArguments` because checkpoint
    saving is now unconditionally safetensors. The config key still governs the
    final adapter save in `_finalize_training_run` and the unsafe-bin
    assertion, so nothing about the safetensors policy is lost here.
    """
    return SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        bf16=config.training.bf16,
        logging_steps=1,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        seed=config.training.seed,
        gradient_checkpointing=config.training.gradient_checkpointing,
        max_length=config.training.max_seq_length,
        dataset_text_field=config.dataset_text_field,
    )


def _build_sft_trainer(model, dataset, tokenizer, training_args):
    """Construct the trainer against the TRL 1.x `SFTTrainer` signature.

    TRL 1.x moved the SFT-specific knobs onto `SFTConfig` and renamed
    `tokenizer` to `processing_class`; passing the pre-1.x kwargs here raises
    `TypeError` at construction time. `tests/test_trainer_trl_api.py` pins this
    call against the installed TRL so the next rename fails CI.
    """
    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )


def _prepare_peft_model(config, model):
    if config.training.gradient_checkpointing:
        model.config.use_cache = False

    if config.quantization.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.training.gradient_checkpointing
        )

    target_modules = validate_lora_targets_exist(model, config.lora)
    peft_config = LoraConfig(
        r=config.lora.lora_r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        revision=config.revision,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def _finalize_training_run(config, trainer, finalization: _TrainingFinalization):
    out_dir = finalization.out_dir
    logger.info(f"Saving adapter to {out_dir}")
    trainer.model.save_pretrained(out_dir, safe_serialization=config.runtime.save_safetensors)
    # `Trainer.tokenizer` was removed in Transformers 5; the tokenizer now
    # lives on `processing_class` (still exposes `save_pretrained`).
    tokenizer = trainer.processing_class
    tokenizer.save_pretrained(out_dir)

    if not config.runtime.allow_unsafe_serialization:
        assert_no_unsafe_weight_bins(out_dir)

    index_path = write_artifact_index(
        out_dir,
        producer_provenance=(
            finalization.producer_provenance.model_dump(mode="json")
            if finalization.producer_provenance is not None
            else None
        ),
    )
    logger.info(f"Artifact index written to {index_path}")

    vram_used = torch.cuda.max_memory_allocated() / BYTES_PER_GB
    logger.info(f"Max VRAM used: {vram_used:.2f} GiB")

    metrics = {
        "max_vram_gb": vram_used,
        "gpu_report": finalization.gpu_report,
        "artifact_index": index_path,
    }
    write_run_manifest(
        os.path.join("runs", config.run_name),
        config.model_dump(),
        metrics,
        trainer.model,
        tokenizer,
        trainer.train_dataset,
    )


def run_training(config):
    check_cuda_available(required=True)
    gpu_report = get_gpu_report()
    logger.info(f"GPU Report: {gpu_report}")

    estimate_training_risk(config, gpu_report)
    warn_on_disk_pressure(config)

    frozen_binding = _bind_frozen_training_input(config)
    if frozen_binding is None:
        # Reject a bad dataset_text_field here, while it is still cheap: scanning
        # the raw JSONL needs no tokenizer, so a misconfigured run dies before
        # `load_base_model` spends time and VRAM below.
        validate_dataset_text_field_in_source(config.dataset_path, config.dataset_text_field)

    logger.info(f"Loading {config.model_id} for run {config.run_name}")
    model, tokenizer = load_base_model(
        config.model_id,
        config.trust_remote_code,
        config.quantization,
        config.training.bf16,
        revision=config.revision,
    )
    model = _prepare_peft_model(config, model)

    producer_provenance = None
    if frozen_binding is None:
        dataset = load_jsonl_dataset(config.dataset_path, tokenizer)
    else:
        dataset = load_frozen_dataset(
            frozen_binding.manifest_path,
            "train",
            tokenizer,
            expected_binding=frozen_binding,
        )
        producer_provenance = _frozen_producer_provenance(config, frozen_binding)
    # Authoritative re-check: the peek above only saw the first row, and a
    # later row in a mixed-format file can normalize to different columns.
    validate_dataset_text_field(dataset.column_names, config.dataset_text_field)
    logger.info(f"Dataset size: {len(dataset)}")

    out_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(out_dir, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(out_dir, config)

    training_args = _build_training_args(config, out_dir)
    trainer = _build_sft_trainer(model, dataset, tokenizer, training_args)

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    _finalize_training_run(
        config,
        trainer,
        _TrainingFinalization(out_dir, gpu_report, producer_provenance),
    )


def _bind_frozen_training_input(config) -> FrozenSplitBinding | None:
    if config.split_manifest_path is None:
        return None
    if config.split_name != "train":
        raise ValueError("frozen training requires split_name: train")
    if config.dataset_text_field != "text":
        raise ValueError("frozen training requires dataset_text_field: text")
    if config.trust_remote_code:
        raise ValueError("frozen training requires trust_remote_code: false")
    _reject_local_frozen_base(config.model_id)
    _require_frozen_revision(config.revision)
    _reject_frozen_resume(config)
    _require_empty_frozen_run_directory(config)
    return bind_frozen_split(config.split_manifest_path, "train")


def _reject_local_frozen_base(model_id: str) -> None:
    if _looks_like_local_model(model_id):
        raise ValueError("evaluation eligible frozen training requires a Hub model repository")
    try:
        validate_repo_id(model_id)
    except HFValidationError as exc:
        raise ValueError(
            "evaluation eligible frozen training requires a Hub model repository"
        ) from exc


def _looks_like_local_model(model_id: str) -> bool:
    supplied = Path(model_id).expanduser()
    windows = PureWindowsPath(model_id)
    if supplied.is_absolute():
        return True
    if windows.is_absolute():
        return True
    if windows.drive:
        return True
    if model_id.startswith(("./", "../", "~")):
        return True
    return os.path.lexists(supplied)


def _require_frozen_revision(revision: str | None) -> None:
    if revision is None or re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
        raise ValueError("frozen training requires an immutable lowercase commit revision")


def _reject_frozen_resume(config) -> None:
    if config.training.resume_checkpoint_path or config.training.resume_from_latest_checkpoint:
        raise ValueError(
            "frozen training resume is disabled until checkpoints carry verified provenance"
        )


def _require_empty_frozen_run_directory(config) -> None:
    run_dir = Path(config.output_dir).expanduser() / config.run_name
    if not os.path.lexists(run_dir):
        return
    if _is_empty_run_directory(run_dir):
        return
    raise FileExistsError(f"frozen training requires an empty run directory: {run_dir}")


def _is_empty_run_directory(run_dir: Path) -> bool:
    if run_dir.is_symlink():
        return False
    if not run_dir.is_dir():
        return False
    return next(run_dir.iterdir(), None) is None


def _frozen_producer_provenance(config, binding: FrozenSplitBinding) -> ArtifactProducerProvenance:
    return ArtifactProducerProvenance(
        base_model_name_or_path=config.model_id,
        revision=config.revision,
        training_split_manifest_sha256=binding.manifest_sha256,
        training_split_name="train",
        training_split_sha256=binding.split_sha256,
    )
