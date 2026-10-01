"""Run G0 before G1 and publish the Granite first-SFT artifact tree."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .._atomic_directory import rename_noreplace, require_rename_noreplace_support
from .._atomic_file import publish_bytes_noreplace, write_fsynced_bytes
from .._source_snapshot import nearest_existing_output_ancestor
from .._token_provenance import SerializerBinding
from ..eval.compare import compare_arms
from ..eval.contract import held_out_task_ids
from ..eval.score import ArmMetrics, ExampleScore, GenerationRecord, score_arm
from ..eval.serializers import code_repair_prompt
from ..path_safety import resolve_existing_path
from ..split_schema import canonical_json_bytes, sha256_bytes, sha256_file
from ..split_validation import validate_split_manifest
from .qualification import load_qualification_report
from .report import (
    publish_terminal_files,
    render_failure_report,
    render_success_report,
    scored_comparison,
    unscored_comparison,
)
from .schema import (
    GraniteFirstSftContract,
    ReloadObservation,
    TrainingResults,
    contract_json_bytes,
    load_experiment_contract,
)

ExperimentStatus = Literal["scored", "g0_failed", "training_failed", "reload_failed", "g1_failed"]
ArmRole = Literal["causal_base", "causal_sft"]


@dataclass(frozen=True)
class ExperimentOutcome:
    experiment_dir: Path
    status: ExperimentStatus


@dataclass(frozen=True)
class ExperimentPhases:
    """Injected G0, train, reload, and G1 steps. G0 does not receive an adapter."""

    evaluate_base: Callable[[GraniteFirstSftContract], tuple[GenerationRecord, ...]]
    train: Callable[[GraniteFirstSftContract, Path], TrainingResults]
    reload: Callable[[GraniteFirstSftContract, Path], ReloadObservation]
    evaluate_sft: Callable[[GraniteFirstSftContract, Path], tuple[GenerationRecord, ...]]


def execute_experiment(
    contract_path: str | Path,
    *,
    agoge_commit: str,
    worktree_clean: bool,
    phases: ExperimentPhases,
) -> ExperimentOutcome:
    """Evaluate G0, then train and score G1, leaving a diagnosable artifact on failure."""

    path = resolve_existing_path(str(contract_path), must_be_file=True)
    experiment_dir = path.parent
    contract = _load_verified_contract(path)
    _require_commit(contract, agoge_commit, worktree_clean=worktree_clean)
    _require_live_pins(contract)
    _require_fresh_execution(experiment_dir)
    adapter_dir = experiment_dir / "g1-sft" / "adapter"
    try:
        base_records = phases.evaluate_base(contract)
        base_scores, base_metrics = _score_role(contract, base_records, "causal_base")
    except Exception as exc:  # noqa: BLE001 - preserve a diagnosable G0 failure artifact
        _publish_failure(experiment_dir, contract, stage="g0", reason=_reason(exc))
        return ExperimentOutcome(experiment_dir, "g0_failed")
    _publish_g0(experiment_dir / "g0-base", base_records, base_scores, base_metrics)
    training = _train(phases, contract, adapter_dir)
    if training.status != "completed":
        _publish_failure(
            experiment_dir,
            contract,
            stage="train",
            reason=training.failure_reason or "G1 training failed",
            training=training,
        )
        return ExperimentOutcome(experiment_dir, "training_failed")
    reload = _reload(phases, contract, adapter_dir)
    _write_new(experiment_dir / "g1-sft" / "reload.json", _model_bytes(reload))
    if reload.status != "reloaded":
        _publish_failure(
            experiment_dir,
            contract,
            stage="reload",
            reason=reload.failure_reason or "clean reload failed",
            training=training,
        )
        return ExperimentOutcome(experiment_dir, "reload_failed")
    try:
        sft_records = phases.evaluate_sft(contract, adapter_dir)
        sft_scores, sft_metrics = _score_role(contract, sft_records, "causal_sft")
    except Exception as exc:  # noqa: BLE001 - preserve a diagnosable G1 failure artifact
        _publish_failure(
            experiment_dir,
            contract,
            stage="g1",
            reason=_reason(exc),
            training=training,
        )
        return ExperimentOutcome(experiment_dir, "g1_failed")
    _write_arm(experiment_dir / "g1-sft", sft_records, sft_scores, sft_metrics)
    comparison = scored_comparison(
        contract,
        base_metrics,
        sft_metrics,
        compare_arms(base_scores, sft_scores),
    )
    publish_terminal_files(
        experiment_dir,
        contract=contract,
        comparison=comparison,
        training=training,
        report=render_success_report(contract, comparison, training),
    )
    return ExperimentOutcome(experiment_dir, "scored")


def _load_verified_contract(path: Path) -> GraniteFirstSftContract:
    payload = path.read_bytes()
    contract = load_experiment_contract(path)
    if payload != contract_json_bytes(contract):
        raise ValueError("experiment contract is not canonical JSON")
    digest_path = path.parent / "experiment-contract.sha256"
    recorded = digest_path.read_text(encoding="utf-8").split()[0]
    if recorded != sha256_bytes(payload):
        raise ValueError("experiment contract SHA-256 mismatch")
    return contract


def _require_commit(
    contract: GraniteFirstSftContract, agoge_commit: str, *, worktree_clean: bool
) -> None:
    if not worktree_clean:
        raise ValueError("measured execution requires a clean Agoge worktree")
    if agoge_commit != contract.agoge_commit:
        raise ValueError(
            f"Agoge commit {agoge_commit} does not match frozen commit {contract.agoge_commit}"
        )


def _require_live_pins(contract: GraniteFirstSftContract) -> None:
    SerializerBinding(
        implementation=code_repair_prompt,
        expected_serializer_id=contract.serializer_id,
        expected_serializer_version=contract.serializer_version,
        expected_serializer_sha256=contract.serializer_sha256,
    )
    manifest_path = resolve_existing_path(contract.split.split_manifest_path, must_be_file=True)
    if sha256_file(manifest_path) != contract.split.split_manifest_sha256:
        raise ValueError("split manifest SHA-256 drifted after the contract was frozen")
    manifest = validate_split_manifest(manifest_path)
    if held_out_task_ids(manifest) != contract.split.logical_task_ids:
        raise ValueError("held-out task IDs drifted after the contract was frozen")
    if manifest.splits["train"].sha256 != contract.split.train_sha256:
        raise ValueError("train split SHA-256 drifted after the contract was frozen")
    if manifest.splits["validation"].sha256 != contract.split.validation_sha256:
        raise ValueError("validation split SHA-256 drifted after the contract was frozen")
    if manifest.splits["held_out"].sha256 != contract.split.held_out_sha256:
        raise ValueError("held-out split SHA-256 drifted after the contract was frozen")
    qualification_path = resolve_existing_path(
        contract.blockers.qualification_report_path, must_be_file=True
    )
    qualification, digest = load_qualification_report(qualification_path)
    if digest != contract.blockers.qualification_report_sha256:
        raise ValueError("qualification report SHA-256 drifted after the contract was frozen")
    if qualification.verdict != "pass":
        raise ValueError("qualification verdict is no longer pass")
    if qualification.model_revision != contract.model_revision:
        raise ValueError("qualification model revision drifted after the contract was frozen")
    if qualification.target_modules != contract.budget.target_modules:
        raise ValueError("qualification target modules drifted after the contract was frozen")


def _require_fresh_execution(experiment_dir: Path) -> None:
    if (experiment_dir / "comparison.json").exists():
        raise FileExistsError("experiment comparison already exists")
    if (experiment_dir / "g0-base").exists():
        raise FileExistsError("g0-base already exists; refusing to re-evaluate the baseline")


def _score_role(
    contract: GraniteFirstSftContract,
    records: tuple[GenerationRecord, ...],
    role: ArmRole,
) -> tuple[tuple[ExampleScore, ...], ArmMetrics]:
    task_ids = tuple(record.task_id for record in records)
    if task_ids != contract.split.logical_task_ids:
        raise ValueError("generated task IDs drifted from the frozen held-out membership")
    return score_arm(role, records, scoring_version=contract.decoding.scoring_version)


def _train(
    phases: ExperimentPhases, contract: GraniteFirstSftContract, adapter_dir: Path
) -> TrainingResults:
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    if adapter_dir.exists():
        raise FileExistsError(f"G1 adapter directory already exists: {adapter_dir}")
    try:
        return phases.train(contract, adapter_dir)
    except Exception as exc:  # noqa: BLE001 - training failure stays in the experiment report
        return TrainingResults(status="failed", failure_reason=_reason(exc))


def _reload(
    phases: ExperimentPhases, contract: GraniteFirstSftContract, adapter_dir: Path
) -> ReloadObservation:
    try:
        return phases.reload(contract, adapter_dir)
    except Exception as exc:  # noqa: BLE001 - reload failure stays in the experiment report
        return ReloadObservation(status="failed", failure_reason=_reason(exc))


def _publish_failure(
    experiment_dir: Path,
    contract: GraniteFirstSftContract,
    *,
    stage: str,
    reason: str,
    training: TrainingResults | None = None,
) -> None:
    results = training or TrainingResults(status="failed", failure_reason=reason)
    if stage == "g0":
        _publish_g0_failure(experiment_dir / "g0-base", reason)
    publish_terminal_files(
        experiment_dir,
        contract=contract,
        comparison=unscored_comparison(reason),
        training=results,
        report=render_failure_report(contract, stage=stage, reason=reason),
    )


def _publish_g0(destination: Path, records, scores, metrics) -> None:
    _publish_directory(destination, lambda staged: _write_arm(staged, records, scores, metrics))


def _publish_g0_failure(destination: Path, reason: str) -> None:
    payload = canonical_json_bytes({"status": "failed", "reason": reason}) + b"\n"

    def write_failure(staged: Path) -> None:
        _exclusive_write(staged / "failure.json", payload)

    _publish_directory(destination, write_failure)


def _publish_directory(destination: Path, writer: Callable[[Path], None]) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite {destination}")
    staging_parent = nearest_existing_output_ancestor(destination)
    require_rename_noreplace_support(staging_parent)
    with tempfile.TemporaryDirectory(prefix=".agoge-arm-", dir=staging_parent) as staging:
        staged = Path(staging) / destination.name
        staged.mkdir()
        writer(staged)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rename_noreplace(staged, destination)


def _write_arm(directory: Path, records, scores, metrics: ArmMetrics) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _exclusive_write(directory / "generations.jsonl", _jsonl(records))
    _exclusive_write(directory / "scores.jsonl", _jsonl(scores))
    _exclusive_write(directory / "metrics.json", _model_bytes(metrics))


def _jsonl(rows: tuple[object, ...]) -> bytes:
    chunks: list[bytes] = []
    for row in rows:
        payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
        chunks.append(canonical_json_bytes(payload) + b"\n")
    return b"".join(chunks)


def _model_bytes(model: object) -> bytes:
    if not hasattr(model, "model_dump"):
        raise TypeError("experiment artifact must be a pydantic model")
    return canonical_json_bytes(model.model_dump(mode="json")) + b"\n"


def _write_new(path: Path, payload: bytes) -> None:
    publish_bytes_noreplace(
        path,
        payload,
        refusal="refusing to overwrite experiment artifact",
        writer=write_fsynced_bytes,
    )


def _reason(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


def _exclusive_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
