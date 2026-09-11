"""Helpers for the executable held-out canary tests."""

from pathlib import Path

from agoge_forger.eval.score import OBJECTIVE_SCORING_VERSION, GenerationRecord
from agoge_forger.eval.serializers import code_repair_prompt
from agoge_forger.split_contract import SerializerBinding
from tests.evaluation_contract_cases import evaluation_case


def canary_evaluation_case(tmp_path: Path):
    manifest_path, manifest, base, sft = evaluation_case(tmp_path)
    serializer = SerializerBinding(implementation=code_repair_prompt)
    updates = {
        "serializer_id": serializer.serializer_id,
        "serializer_version": serializer.serializer_version,
        "serializer_sha256": serializer.serializer_sha256,
        "scoring_version": OBJECTIVE_SCORING_VERSION,
    }
    return (
        manifest_path,
        manifest,
        base.model_copy(update=updates),
        sft.model_copy(update=updates),
    )


def scripted_generator(pattern: str = "cycle"):
    """Return a generator that never loads a model.

    ``cycle`` maps held-out order onto improved / regressed / tied / invalid.
    """

    def generate(arm, tasks):
        records = []
        role = arm.role
        for index, task in enumerate(tasks):
            records.append(_scripted_record(role, task, index, pattern))
        return tuple(records)

    return generate


def _scripted_record(role: str, task, index: int, pattern: str) -> GenerationRecord:
    if task.status != "ready":
        return _scripted_not_ready(task)
    expected = task.expected_completion or ""
    if pattern == "all-correct":
        return _scripted_ok(task, expected)
    return _scripted_cycle(role, task, index % 5, expected)


def _scripted_not_ready(task) -> GenerationRecord:
    status = "invalid" if task.status == "invalid" else "unsupported"
    return GenerationRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        expected_completion=task.expected_completion,
        status=status,
        reason=task.reason,
    )


def _scripted_cycle(role: str, task, lane: int, expected: str) -> GenerationRecord:
    if lane == 4:
        return GenerationRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            expected_completion=task.expected_completion,
            status="unsupported",
            reason="scripted unsupported",
            prompt_token_count=task.prompt_token_count,
        )
    return _scripted_ok(task, _scripted_completion(role, lane, expected))


def _scripted_completion(role: str, lane: int, expected: str) -> str:
    if lane == 0:
        return expected
    if lane == 1 and role == "causal_sft":
        return expected
    if lane == 2 and role == "causal_base":
        return expected
    return "WRONG"


def _scripted_ok(task, completion: str) -> GenerationRecord:
    return GenerationRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        expected_completion=task.expected_completion,
        raw_text=f"{task.prompt}{completion}",
        completion=completion,
        prompt_token_count=task.prompt_token_count,
        status="ok",
    )
