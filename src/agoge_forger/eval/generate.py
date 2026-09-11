"""Transformers generation for one held-out evaluation arm."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from peft import PeftModel
from transformers import set_seed

from .._token_provenance import TokenizerBinding
from ..artifacts.safetensors_io import assert_no_unsafe_weight_bins
from ..models.load import load_base_model
from .contract import DecodingContract, EvaluationArm
from .score import GenerationRecord

PreparedStatus = Literal["ready", "invalid", "unsupported"]


@dataclass(slots=True)
class PreparedTask:
    task_id: str
    prompt: str | None
    expected_completion: str | None
    status: PreparedStatus
    reason: str | None = None
    prompt_token_count: int | None = None


def prompt_token_count(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
    return _token_id_length(encoded["input_ids"])


def _token_id_length(ids: Any) -> int:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def generation_kwargs(decoding: DecodingContract, tokenizer: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": decoding.max_new_tokens,
        "do_sample": decoding.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if decoding.do_sample:
        kwargs["temperature"] = decoding.temperature
        kwargs["top_p"] = decoding.top_p
    return kwargs


def apply_context_window(
    task: PreparedTask,
    tokenizer: Any,
    *,
    context_window: int,
    truncation_policy: str,
) -> PreparedTask:
    if task.status != "ready" or task.prompt is None:
        return task
    count = prompt_token_count(tokenizer, task.prompt)
    task.prompt_token_count = count
    if count <= context_window:
        return task
    return _overflow_prepared_task(task, count, context_window, truncation_policy)


def _overflow_prepared_task(
    task: PreparedTask, count: int, context_window: int, truncation_policy: str
) -> PreparedTask:
    if truncation_policy == "mark_unsupported":
        return PreparedTask(
            task.task_id,
            task.prompt,
            task.expected_completion,
            "unsupported",
            reason="prompt exceeds context_window",
            prompt_token_count=count,
        )
    if truncation_policy == "reject":
        raise ValueError(
            f"prompt for {task.task_id} has {count} tokens; exceeds context_window {context_window}"
        )
    raise ValueError(
        f"truncation_policy {truncation_policy!r} is not supported by the code-repair canary"
    )


def generate_completions(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[PreparedTask],
    decoding: DecodingContract,
) -> tuple[GenerationRecord, ...]:
    set_seed(decoding.seed)
    model.eval()
    kwargs = generation_kwargs(decoding, tokenizer)
    records: list[GenerationRecord] = []
    for task in tasks:
        if task.status != "ready" or task.prompt is None:
            records.append(_non_ok_record(task))
            continue
        records.append(_generate_one(model, tokenizer, task, kwargs))
    return tuple(records)


def load_arm_model(
    arm: EvaluationArm,
    *,
    artifact_root: str | None,
    trust_remote_code: bool,
    device_map: str = "auto",
) -> tuple[Any, Any]:
    if arm.role == "causal_base":
        return _load_pretrained(
            arm.model_repository, arm.model_revision, trust_remote_code, device_map
        )
    if artifact_root is None:
        raise ValueError("causal_sft arm requires an artifact directory")
    if arm.artifact is not None and arm.artifact.kind == "peft_adapter":
        return _peft_adapter(arm, artifact_root, trust_remote_code, device_map)
    return _load_pretrained(artifact_root, None, trust_remote_code, device_map)


def _load_pretrained(
    repository: str, revision: str | None, trust_remote_code: bool, device_map: str
) -> tuple[Any, Any]:
    assert_no_unsafe_weight_bins(repository, recursive=True)
    return load_base_model(
        repository,
        trust_remote_code=trust_remote_code,
        quant_config=None,
        bf16=True,
        revision=revision,
        device_map=device_map,
        local_files_only=revision is None,
    )


def _peft_adapter(
    arm: EvaluationArm, artifact_root: str, trust_remote_code: bool, device_map: str
) -> tuple[Any, Any]:
    assert_no_unsafe_weight_bins(artifact_root, recursive=True)
    model, tokenizer = _load_pretrained(
        arm.model_repository, arm.model_revision, trust_remote_code, device_map
    )
    return PeftModel.from_pretrained(model, artifact_root), tokenizer


def require_tokenizer_matches(tokenizer: Any, arm: EvaluationArm) -> TokenizerBinding:
    return TokenizerBinding(
        implementation=tokenizer,
        expected_tokenizer_id=arm.tokenizer_repository,
        expected_tokenizer_revision=arm.tokenizer_revision,
        expected_tokenizer_sha256=arm.tokenizer_sha256,
    )


def _generate_one(
    model: Any,
    tokenizer: Any,
    task: PreparedTask,
    kwargs: dict[str, Any],
) -> GenerationRecord:
    if task.prompt is None:
        raise ValueError(f"ready task {task.task_id} is missing a prompt")
    device = next(model.parameters()).device
    inputs = _inputs_on_device(
        tokenizer(task.prompt, return_tensors="pt", truncation=False), device
    )
    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[-1]
    with torch.no_grad():
        outputs = model.generate(**inputs, **kwargs)
    completion_ids = outputs[0][prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    token_count = task.prompt_token_count
    if token_count is None:
        token_count = prompt_len
    return GenerationRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        expected_completion=task.expected_completion,
        raw_text=raw_text,
        completion=completion,
        prompt_token_count=token_count,
        completion_token_count=int(completion_ids.shape[-1]),
        status="ok",
    )


def _non_ok_record(task: PreparedTask) -> GenerationRecord:
    status: Literal["unsupported", "invalid"] = (
        "invalid" if task.status == "invalid" else "unsupported"
    )
    return GenerationRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        expected_completion=task.expected_completion,
        prompt_token_count=task.prompt_token_count,
        status=status,
        reason=task.reason,
    )


def _inputs_on_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }
