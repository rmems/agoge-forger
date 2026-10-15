import functools
import inspect
import os
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index
from ..datasets import load_jsonl_dataset
from ..eval import ArtifactProducerProvenance
from ..logging import logger
from ..manifests import write_run_manifest
from ..models.load import load_base_model
from ..telemetry import attach_telemetry, open_training_session
from ..telemetry.gpu import gpu_identity
from .checkpoints import resolve_resume_checkpoint
from .completion import prepare_completion_dataset
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


def _build_training_args(config, out_dir):
    """Map the experiment config onto TRL's `SFTConfig`.

    The YAML/pydantic keys are the stable surface; the TRL names are not.
    Translate at this boundary only, so upstream renames never leak into
    `configs/*.yaml` or `ExperimentConfig`:

      * `training.max_seq_length` -> `SFTConfig.max_length`
      * `dataset_text_field`      -> `SFTConfig.dataset_text_field`
      * `training.loss_type`      -> `nll` when the YAML value is `chunked_nll`

    `runtime.save_safetensors` is deliberately *not* forwarded: Transformers 5
    removed `save_safetensors` from `TrainingArguments` because checkpoint
    saving is now unconditionally safetensors. The config key still governs the
    final adapter save in `_finalize_training_run` and the unsafe-bin
    assertion, so nothing about the safetensors policy is lost here.
    """
    if config.training.completion_only_loss and config.dataset_text_field != "text":
        raise ValueError("completion_only_loss requires dataset_text_field=text")
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
        completion_only_loss=config.training.completion_only_loss,
        loss_type=_sft_loss_type(config.training.loss_type),
        activation_offloading=config.training.activation_offloading,
    )


def _sft_loss_type(loss_type: str) -> str:
    """Send TRL the loss name it still accepts.

    TRL 1.15 warns and rewrites ``chunked_nll`` to ``nll`` because that default
    already scores tokens without building the full logits tensor.
    """
    if loss_type == "chunked_nll":
        return "nll"
    return loss_type


def _build_sft_trainer(model, dataset, tokenizer, training_args):
    """Construct the trainer against the TRL 1.x `SFTTrainer` signature.

    TRL 1.x moved the SFT-specific knobs onto `SFTConfig` and renamed
    `tokenizer` to `processing_class`; passing the pre-1.x kwargs here raises
    `TypeError` at construction time. `tests/test_trainer_trl_api.py` pins this
    call against the installed TRL so the next rename fails CI.
    """
    if training_args.completion_only_loss:
        dataset = prepare_completion_dataset(dataset, tokenizer, training_args)
        training_args.dataset_kwargs = {
            **(training_args.dataset_kwargs or {}),
            "skip_prepare_dataset": True,
        }
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    _score_cpu_tokens_in_pytorch(trainer.model)
    return trainer


@dataclass(frozen=True)
class _CpuScoreCall:
    forward: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    labels: torch.Tensor | None
    shift_labels: torch.Tensor | None


@dataclass(frozen=True)
class _CpuTokenScores:
    log_probs: torch.Tensor
    entropy: torch.Tensor
    is_top1: torch.Tensor
    label_mask: torch.Tensor


def _model_parameters_are_on_cpu(model: torch.nn.Module) -> bool:
    try:
        parameter = next(model.parameters())
    except StopIteration:
        return False
    return parameter.device.type == "cpu"


def _forward_uses_fused_lm_head(model: torch.nn.Module) -> bool:
    signature = inspect.signature(model.forward, follow_wrapped=False)
    return "fused_lm_head" in signature.parameters


def _optional_label_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError("CPU SFT scoring expected tensor labels")


def _needs_cpu_logprob(model: torch.nn.Module) -> bool:
    if not _model_parameters_are_on_cpu(model):
        return False
    return _forward_uses_fused_lm_head(model)


def _score_targets_missing(labels: torch.Tensor | None, shift_labels: torch.Tensor | None) -> bool:
    if labels is not None:
        return False
    return shift_labels is None


def _delegate_to_unfused_forward(
    fused_lm_head: bool, labels: torch.Tensor | None, shift_labels: torch.Tensor | None
) -> bool:
    if not fused_lm_head:
        return True
    return _score_targets_missing(labels, shift_labels)


def _restore_forward_kwargs(
    kwargs: dict[str, Any],
    labels: torch.Tensor | None,
    shift_labels: torch.Tensor | None,
    num_items_in_batch: Any,
) -> None:
    if labels is not None:
        kwargs["labels"] = labels
    if shift_labels is not None:
        kwargs["shift_labels"] = shift_labels
    if num_items_in_batch is not None:
        kwargs["num_items_in_batch"] = num_items_in_batch


def _score_cpu_tokens_in_pytorch(model: torch.nn.Module) -> None:
    """Replace TRL's Triton fused head when the model is on CPU.

    TRL 1.15 always scores SFT tokens with a Triton kernel. CPU runners have no
    Triton driver, so the bounded CPU trainer tests cannot take a step. GPU
    training keeps the fused head.
    """
    if not _needs_cpu_logprob(model):
        return
    replacement = _pytorch_cpu_forward(model, model.forward)
    model.forward = types.MethodType(replacement, model)


def _pytorch_cpu_forward(
    model: torch.nn.Module, fused_forward: Callable[..., Any]
) -> Callable[..., Any]:
    @functools.wraps(type(model).forward)
    def _forward(_self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        fused_lm_head = bool(kwargs.pop("fused_lm_head", False))
        labels = _optional_label_tensor(kwargs.pop("labels", None))
        shift_labels = _optional_label_tensor(kwargs.pop("shift_labels", None))
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        if _delegate_to_unfused_forward(fused_lm_head, labels, shift_labels):
            _restore_forward_kwargs(kwargs, labels, shift_labels, num_items_in_batch)
            return fused_forward(*args, fused_lm_head=False, **kwargs)
        return _cpu_token_scores(
            _CpuScoreCall(
                forward=fused_forward,
                args=args,
                kwargs=kwargs,
                labels=labels,
                shift_labels=shift_labels,
            )
        )

    return _forward


def _cpu_token_scores(call: _CpuScoreCall) -> _CpuTokenScores:
    outputs = call.forward(*call.args, fused_lm_head=False, **call.kwargs)
    logits = outputs.logits
    if not isinstance(logits, torch.Tensor):
        raise TypeError("CPU SFT scoring requires logits from the model forward")
    if call.shift_labels is None:
        if call.labels is None:
            raise TypeError("CPU SFT scoring requires labels or shift_labels")
        shift_logits = logits[:, :-1, :]
        target = call.labels[:, 1:]
    else:
        shift_logits = logits
        target = call.shift_labels
    log_probs_full = torch.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs_full.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = target != -100
    zeros = torch.zeros_like(token_log_probs)
    log_probs = torch.where(mask, token_log_probs, zeros)
    entropy_full = -(log_probs_full.exp() * log_probs_full).sum(dim=-1)
    predicted = shift_logits.argmax(dim=-1)
    return _CpuTokenScores(
        log_probs=log_probs,
        entropy=torch.where(mask, entropy_full, zeros).detach(),
        is_top1=(predicted == target) & mask,
        label_mask=mask,
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


@dataclass(frozen=True)
class _TrainingFinalization:
    trainer: Any
    out_dir: str
    gpu_report: Mapping[str, object]
    producer_provenance: ArtifactProducerProvenance | Mapping[str, object] | None = None
    telemetry_metrics: Mapping[str, object] | None = None


def _finalize_training_run(config, finalization: _TrainingFinalization):
    trainer = finalization.trainer
    out_dir = finalization.out_dir
    provenance = _require_training_provenance(finalization.producer_provenance)
    logger.info(f"Saving adapter to {out_dir}")
    trainer.model.save_pretrained(out_dir, safe_serialization=config.runtime.save_safetensors)
    # `Trainer.tokenizer` was removed in Transformers 5; the tokenizer now
    # lives on `processing_class` (still exposes `save_pretrained`).
    tokenizer = trainer.processing_class
    tokenizer.save_pretrained(out_dir)

    if not config.runtime.allow_unsafe_serialization:
        assert_no_unsafe_weight_bins(out_dir)

    index_path = write_artifact_index(out_dir, producer_provenance=provenance)
    logger.info(f"Artifact index written to {index_path}")

    vram_used: float = torch.cuda.max_memory_allocated() / BYTES_PER_GB
    logger.info(f"Max VRAM used: {vram_used:.2f} GiB")

    metrics = {
        "max_vram_gb": vram_used,
        "gpu_report": finalization.gpu_report,
        "artifact_index": index_path,
    }
    if finalization.telemetry_metrics:
        metrics["telemetry"] = dict(finalization.telemetry_metrics)
    write_run_manifest(
        os.path.join("runs", config.run_name),
        config.model_dump(),
        metrics,
        trainer.model,
        tokenizer,
        trainer.train_dataset,
    )


def _require_training_provenance(producer_provenance: Any) -> ArtifactProducerProvenance:
    if producer_provenance is None:
        raise ValueError("training cannot save an adapter without ArtifactProducerProvenance")
    if isinstance(producer_provenance, ArtifactProducerProvenance):
        return producer_provenance
    return ArtifactProducerProvenance.model_validate(producer_provenance)


def run_training(config, producer_provenance=None):
    session = open_training_session(config)
    try:
        _execute_training(config, producer_provenance, session)
    except Exception:
        session.record_failure()
        raise
    finally:
        session.close()


def _execute_training(config, producer_provenance, session):
    check_cuda_available(required=True)
    session.writer.update_gpu(gpu_identity())
    gpu_report = get_gpu_report()
    logger.info(f"GPU Report: {gpu_report}")

    estimate_training_risk(config, gpu_report)
    warn_on_disk_pressure(config)

    # Reject a bad dataset_text_field here, while it is still cheap: scanning
    # the raw JSONL needs no tokenizer, so a misconfigured run dies before
    # `load_base_model` spends time and VRAM below.
    validate_dataset_text_field_in_source(config.dataset_path, config.dataset_text_field)

    logger.info(f"Loading {config.model_id} for run {config.run_name}")
    model, tokenizer = load_base_model(
        config.model_id,
        config.trust_remote_code,
        config.quantization,
        revision=config.revision,
    )
    model = _prepare_peft_model(config, model)
    session.emit("model_load", phase="setup", global_step=0)

    dataset = load_jsonl_dataset(config.dataset_path, tokenizer)
    # Authoritative re-check: the peek above only saw the first row, and a
    # later row in a mixed-format file can normalize to different columns.
    validate_dataset_text_field(dataset.column_names, config.dataset_text_field)
    logger.info(f"Dataset size: {len(dataset)}")

    out_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(out_dir, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(out_dir, config)

    training_args = _build_training_args(config, out_dir)
    trainer = _build_sft_trainer(model, dataset, tokenizer, training_args)
    attach_telemetry(trainer, session)

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    _finalize_training_run(
        config,
        _TrainingFinalization(
            trainer=trainer,
            out_dir=out_dir,
            gpu_report=gpu_report,
            producer_provenance=producer_provenance,
            telemetry_metrics=session.manifest_entry(),
        ),
    )
