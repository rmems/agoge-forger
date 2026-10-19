"""Local CUDA phases for the frozen Granite 4.1 first-SFT contract.

These functions load weights. Tests construct ``ExperimentConfig`` and read
artifact files through the pure helpers; they do not call the weight-loading
entry points.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import transformers

from .._strict_json import decode_json_object
from ..config import ExperimentConfig, LoraConfigModel, QuantizationConfig, TrainingConfig
from ..eval.contract import ArtifactIndexReference, EvaluationArm
from ..eval.generate import generate_completions, load_arm_model, require_tokenizer_matches
from ..eval.harness import apply_truncation, prepare_tasks
from ..eval.score import GenerationRecord
from ..eval.serializers import code_repair_prompt
from ..split_contract import SerializerBinding, iter_frozen_records, sha256_file
from ..train.qlora import train_qlora
from .arms import base_arm, sft_arm
from .execute import ExperimentPhases
from .reload import clean_reload
from .schema import (
    GraniteFirstSftContract,
    ReloadObservation,
    SoftwareEnvironment,
    TrainingResults,
)

_PREPROCESSING_NAME = "completion_preprocessing.json"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"
_TRAINER_STATE = "trainer_state.json"


def local_phases(contract_path: Path) -> ExperimentPhases:
    def reload_phase(contract: GraniteFirstSftContract, adapter_dir: Path) -> ReloadObservation:
        del contract
        return clean_reload(contract_path=contract_path, adapter_dir=adapter_dir)

    return ExperimentPhases(
        evaluate_base=evaluate_base_locally,
        train=train_locally,
        reload=reload_phase,
        evaluate_sft=evaluate_sft_locally,
    )


def build_training_config(contract: GraniteFirstSftContract, adapter_dir: Path) -> ExperimentConfig:
    """Map the frozen budget onto ``ExperimentConfig`` without reading weights."""

    dataset = _train_split_path(contract)
    if sha256_file(dataset) != contract.split.train_sha256:
        raise ValueError("train split file digest does not match the frozen contract")
    budget = contract.budget
    return ExperimentConfig(
        model_id=contract.model_repository,
        revision=contract.model_revision,
        trust_remote_code=False,
        dataset_path=str(dataset),
        dataset_text_field=budget.dataset_text_field,
        output_dir=str(adapter_dir.parent),
        run_name=adapter_dir.name,
        quantization=QuantizationConfig(
            load_in_4bit=budget.load_in_4bit,
            bnb_4bit_quant_type=budget.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=budget.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=budget.bnb_4bit_use_double_quant,
        ),
        training=TrainingConfig(
            completion_only_loss=True,
            max_seq_length=budget.max_seq_length,
            batch_size=budget.batch_size,
            gradient_accumulation_steps=budget.gradient_accumulation_steps,
            gradient_checkpointing=budget.gradient_checkpointing,
            learning_rate=budget.learning_rate,
            num_train_epochs=budget.num_train_epochs,
            bf16=budget.bf16,
            seed=budget.seed,
            save_steps=budget.save_steps,
            save_total_limit=budget.save_total_limit,
        ),
        lora=LoraConfigModel(
            lora_r=budget.lora_r,
            lora_alpha=budget.lora_alpha,
            lora_dropout=budget.lora_dropout,
            target_modules=list(budget.target_modules),
            target_modules_mode=budget.target_modules_mode,
        ),
    )


def observe_completed_run(
    adapter_dir: Path,
    wall_clock_seconds: float,
    *,
    peak_vram_bytes: int | None = None,
    gpu_name: str | None = None,
) -> TrainingResults:
    """Read preserved training artifacts. Peak VRAM comes from CUDA unless injected."""

    evidence = _read_json(adapter_dir / _PREPROCESSING_NAME)
    weights = adapter_dir / _ADAPTER_WEIGHTS
    accepted_tokens = _as_int(evidence["shifted_supervised_tokens"], "shifted_supervised_tokens")
    vram = _cuda_peak_vram() if peak_vram_bytes is None else peak_vram_bytes
    name = _cuda_gpu_name() if gpu_name is None else gpu_name
    return TrainingResults(
        status="completed",
        accepted_records=_as_int(evidence["rows"], "rows"),
        accepted_tokens=accepted_tokens,
        train_loss=_train_loss(adapter_dir),
        validation_loss=_validation_loss(adapter_dir),
        wall_clock_seconds=wall_clock_seconds,
        tokens_per_second=accepted_tokens / wall_clock_seconds,
        peak_vram_bytes=vram,
        adapter_bytes=weights.stat().st_size,
        adapter_sha256=sha256_file(weights),
        checkpoint_bytes=_checkpoint_bytes(adapter_dir),
        environment=current_environment(name),
    )


def current_environment(gpu_name: str | None) -> SoftwareEnvironment:
    return SoftwareEnvironment(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu_name=gpu_name,
        transformer_version=transformers.__version__,
    )


def evaluate_base_locally(contract: GraniteFirstSftContract) -> tuple[GenerationRecord, ...]:
    return _generate(contract, base_arm(contract), artifact_root=None)


def evaluate_sft_locally(
    contract: GraniteFirstSftContract, adapter_dir: Path
) -> tuple[GenerationRecord, ...]:
    index_path = adapter_dir / "artifact_index.json"
    artifact = ArtifactIndexReference(
        kind="peft_adapter",
        artifact_index_path=str(index_path),
        artifact_index_sha256=sha256_file(index_path),
    )
    return _generate(contract, sft_arm(contract, artifact), artifact_root=str(adapter_dir))


def train_locally(contract: GraniteFirstSftContract, adapter_dir: Path) -> TrainingResults:
    started = time.perf_counter()
    try:
        train_qlora(build_training_config(contract, adapter_dir))
        return observe_completed_run(adapter_dir, time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - surface the trainer error as a failed observation
        reason = str(exc) or exc.__class__.__name__
        return TrainingResults(status="failed", failure_reason=reason)


def _train_split_path(contract: GraniteFirstSftContract) -> Path:
    return Path(contract.split.split_manifest_path).parent / "splits" / "train.jsonl"


def _generate(
    contract: GraniteFirstSftContract,
    arm: EvaluationArm,
    *,
    artifact_root: str | None,
) -> tuple[GenerationRecord, ...]:
    records = tuple(iter_frozen_records(contract.split.split_manifest_path, "held_out"))
    serializer = SerializerBinding(
        implementation=code_repair_prompt,
        expected_serializer_sha256=contract.serializer_sha256,
    )
    tasks = prepare_tasks(records, serializer)
    model, tokenizer = load_arm_model(
        arm,
        artifact_root=artifact_root,
        trust_remote_code=False,
    )
    require_tokenizer_matches(tokenizer, arm)
    windowed = apply_truncation(tasks, tokenizer, arm)
    try:
        return generate_completions(model, tokenizer, windowed, arm.decoding)
    finally:
        del model


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    return float(value)


def _read_json(path: Path) -> dict[str, object]:
    return decode_json_object(path.read_bytes(), str(path), object_label=path.name)


def _train_loss(adapter_dir: Path) -> float:
    history = _log_history(adapter_dir)
    for entry in reversed(history):
        if "train_loss" in entry:
            return _as_float(entry["train_loss"], "train_loss")
    for entry in reversed(history):
        if "loss" in entry:
            return _as_float(entry["loss"], "loss")
    raise ValueError(f"{adapter_dir / _TRAINER_STATE} has no train loss")


def _validation_loss(adapter_dir: Path) -> float | None:
    for entry in reversed(_log_history(adapter_dir)):
        if "eval_loss" in entry:
            return _as_float(entry["eval_loss"], "eval_loss")
    return None


def _log_history(adapter_dir: Path) -> list[dict[str, object]]:
    document = _read_json(adapter_dir / _TRAINER_STATE)
    history = document.get("log_history")
    if not isinstance(history, list):
        raise TypeError("trainer_state.json log_history must be a list")
    return [entry for entry in history if isinstance(entry, dict)]


def _checkpoint_bytes(adapter_dir: Path) -> int:
    total = 0
    for checkpoint in adapter_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            total += sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    return total


def _cuda_peak_vram() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("completed training requires CUDA to record peak VRAM")
    return int(torch.cuda.max_memory_allocated())


def _cuda_gpu_name() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("completed training requires a CUDA device name")
    return torch.cuda.get_device_name(0)
