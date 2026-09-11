"""Executable base-vs-SFT held-out evaluation for the first code-repair canary."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..split_contract import SerializerBinding, bind_frozen_split, iter_frozen_records
from ..train.completion import _completion_boundary
from .bundle import ArmBundle, EvalBundle, publish_eval_bundle
from .compare import compare_arms
from .contract import (
    COMPARABLE_ARM_FIELDS,
    EvaluationArm,
    PairedEvaluationContract,
    compose_evaluation_contract,
)
from .generate import (
    PreparedTask,
    apply_context_window,
    generate_completions,
    load_arm_model,
    require_tokenizer_matches,
)
from .score import OBJECTIVE_SCORING_VERSION, GenerationRecord, score_arm
from .serializers import code_repair_prompt

ArmGenerator = Callable[[EvaluationArm, Sequence[PreparedTask]], tuple[GenerationRecord, ...]]


@dataclass(frozen=True)
class HeldOutEvalRuntime:
    generator: ArmGenerator | None = None
    tokenizer: Any | None = None
    trust_remote_code: bool = False
    device_map: str = "auto"


def run_held_out_eval(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    arms: tuple[EvaluationArm, EvaluationArm],
    runtime: HeldOutEvalRuntime | None = None,
) -> Path:
    """Evaluate one pinned base and one SFT artifact on frozen held-out membership."""

    options = runtime or HeldOutEvalRuntime()
    base, sft = arms
    destination = Path(output_dir).expanduser()
    contract = compose_evaluation_contract(
        manifest_path=manifest_path,
        contract_path=destination / "contract.json",
        base=base,
        sft=sft,
    )
    _require_canary_contract(contract)
    tasks = _prepare_held_out(manifest_path, contract, options.tokenizer)
    generate = options.generator or _transformers_generator(
        destination,
        trust_remote_code=options.trust_remote_code,
        device_map=options.device_map,
        tokenizer=options.tokenizer,
    )
    base_generations = generate(contract.base, tasks)
    sft_generations = generate(contract.sft, tasks)
    _require_generation_identity(contract.logical_task_ids, base_generations, sft_generations)
    return publish_eval_bundle(
        destination, _bundle_for_arms(contract, base_generations, sft_generations)
    )


def _prepare_held_out(
    manifest_path: str | Path,
    contract: PairedEvaluationContract,
    tokenizer: Any | None,
) -> tuple[PreparedTask, ...]:
    serializer = SerializerBinding(
        implementation=code_repair_prompt,
        expected_serializer_id=contract.base.serializer_id,
        expected_serializer_version=contract.base.serializer_version,
        expected_serializer_sha256=contract.base.serializer_sha256,
    )
    tasks = prepare_tasks(_load_held_out_records(manifest_path, contract), serializer)
    if tokenizer is None:
        return tasks
    return apply_truncation(tasks, tokenizer, contract.base)


def _bundle_for_arms(
    contract: PairedEvaluationContract,
    base_generations: tuple[GenerationRecord, ...],
    sft_generations: tuple[GenerationRecord, ...],
) -> EvalBundle:
    base_scores, base_metrics = score_arm(
        "causal_base", base_generations, scoring_version=contract.base.scoring_version
    )
    sft_scores, sft_metrics = score_arm(
        "causal_sft", sft_generations, scoring_version=contract.sft.scoring_version
    )
    comparison = compare_arms(
        base_scores, sft_scores, scoring_version=contract.base.scoring_version
    )
    return EvalBundle(
        contract=contract,
        comparison=comparison,
        base=ArmBundle(base_generations, base_metrics, base_scores),
        sft=ArmBundle(sft_generations, sft_metrics, sft_scores),
    )


def prepare_tasks(
    records: Sequence[Mapping[str, Any]], serializer: SerializerBinding
) -> tuple[PreparedTask, ...]:
    tasks: list[PreparedTask] = []
    for row in records:
        task_id = str(row["canonical_id"])
        try:
            _completion_boundary(row)
            prompt = serializer(row)
        except (KeyError, TypeError, ValueError) as exc:
            tasks.append(
                PreparedTask(
                    task_id,
                    None,
                    None,
                    "invalid",
                    reason=str(exc),
                )
            )
            continue
        expected = str(row["text"])[int(row["completion_start_char"]) :]
        tasks.append(PreparedTask(task_id, prompt, expected, "ready"))
    return tuple(tasks)


def apply_truncation(
    tasks: Sequence[PreparedTask], tokenizer: Any, arm: EvaluationArm
) -> tuple[PreparedTask, ...]:
    return tuple(
        apply_context_window(
            task,
            tokenizer,
            context_window=arm.context_window,
            truncation_policy=arm.truncation_policy,
        )
        for task in tasks
    )


def _require_canary_contract(contract: PairedEvaluationContract) -> None:
    if contract.base.scoring_version != OBJECTIVE_SCORING_VERSION:
        raise ValueError(
            f"paired contract scoring_version must be {OBJECTIVE_SCORING_VERSION}; "
            f"got {contract.base.scoring_version!r}"
        )
    if contract.base.truncation_policy not in {"reject", "mark_unsupported"}:
        raise ValueError("code-repair canary supports truncation_policy reject or mark_unsupported")
    drift = [
        field
        for field in COMPARABLE_ARM_FIELDS
        if getattr(contract.base, field) != getattr(contract.sft, field)
    ]
    if drift:
        raise ValueError(f"paired evaluation arms are non-comparable; drift in: {drift}")


def _load_held_out_records(
    manifest_path: str | Path, contract: PairedEvaluationContract
) -> tuple[dict[str, Any], ...]:
    binding = bind_frozen_split(manifest_path, "held_out")
    if binding.manifest_sha256 != contract.split_manifest_sha256:
        raise ValueError("held-out eval split-manifest SHA-256 mismatch")
    if binding.split_sha256 != contract.held_out_split_sha256:
        raise ValueError("held-out eval split SHA-256 mismatch")
    records = tuple(iter_frozen_records(manifest_path, "held_out"))
    task_ids = tuple(str(row["canonical_id"]) for row in records)
    if task_ids != contract.logical_task_ids:
        raise ValueError("held-out task IDs differ from the frozen held-out manifest")
    return records


def _require_generation_identity(
    task_ids: tuple[str, ...],
    base_generations: tuple[GenerationRecord, ...],
    sft_generations: tuple[GenerationRecord, ...],
) -> None:
    base_ids = tuple(record.task_id for record in base_generations)
    sft_ids = tuple(record.task_id for record in sft_generations)
    if base_ids != task_ids or sft_ids != task_ids:
        raise ValueError("generated task IDs drifted from frozen held-out membership")


def _transformers_generator(
    destination: Path,
    *,
    trust_remote_code: bool,
    device_map: str,
    tokenizer: Any | None,
) -> ArmGenerator:
    def generate(arm: EvaluationArm, tasks: Sequence[PreparedTask]) -> tuple[GenerationRecord, ...]:
        artifact_root = _artifact_root(arm, destination)
        model, loaded_tokenizer = load_arm_model(
            arm,
            artifact_root=artifact_root,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
        bound_tokenizer = tokenizer if tokenizer is not None else loaded_tokenizer
        require_tokenizer_matches(bound_tokenizer, arm)
        windowed = apply_truncation(tasks, bound_tokenizer, arm)
        try:
            return generate_completions(model, bound_tokenizer, windowed, arm.decoding)
        finally:
            del model

    return generate


def _artifact_root(arm: EvaluationArm, destination: Path) -> str | None:
    if arm.artifact is None:
        return None
    index = Path(os.path.normpath(destination / arm.artifact.artifact_index_path))
    if not index.is_file():
        raise FileNotFoundError(f"SFT artifact index not found: {index}")
    return str(index.parent)
