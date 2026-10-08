"""Machine-readable comparison files and the concise Granite experiment report."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from .._atomic_file import publish_bytes_noreplace, write_fsynced_bytes
from ..eval.compare import ComparisonSummary
from ..eval.score import ArmMetrics, ExampleVerdict
from ..split_schema import canonical_json_bytes
from .schema import (
    REPORT_PARENT_NAME,
    ExperimentComparison,
    GraniteFirstSftContract,
    PairRow,
    TrainingResults,
)

_REFUSAL = "refusing to overwrite experiment artifact"


def scored_comparison(
    contract: GraniteFirstSftContract,
    base_metrics: ArmMetrics,
    sft_metrics: ArmMetrics,
    summary: ComparisonSummary,
) -> ExperimentComparison:
    del base_metrics, sft_metrics
    pairs = tuple(
        PairRow(
            task_id=item.task_id,
            outcome=item.outcome,
            base_verdict=_verdict(item.base_verdict),
            sft_verdict=_verdict(item.sft_verdict),
        )
        for item in summary.outcomes
    )
    return ExperimentComparison(
        status="scored",
        scoring_version=summary.scoring_version,
        g0_accuracy=_comparable_accuracy(pairs, "base_verdict"),
        g1_accuracy=_comparable_accuracy(pairs, "sft_verdict"),
        delta_accuracy=summary.delta_accuracy,
        n_tasks=summary.n_tasks,
        n_improved=summary.n_improved,
        n_regressed=summary.n_regressed,
        n_tied=summary.n_tied,
        n_invalid=summary.n_invalid,
        conclusion=summary.conclusion,
        interpretation_rule_sha256=contract.interpretation_rule_sha256,
        pairs=pairs,
    )


def _comparable_accuracy(pairs: tuple[PairRow, ...], verdict_field: str) -> float | None:
    comparable = tuple(pair for pair in pairs if pair.outcome != "invalid")
    if not comparable:
        return None
    correct = sum(getattr(pair, verdict_field) == "correct" for pair in comparable)
    return correct / len(comparable)


def _verdict(value: str) -> ExampleVerdict:
    if value not in {"correct", "incorrect", "invalid"}:
        raise ValueError(f"unsupported held-out verdict {value!r}")
    return cast(ExampleVerdict, value)


def unscored_comparison(reason: str) -> ExperimentComparison:
    return ExperimentComparison(status="not_run", reason=reason)


def render_success_report(
    contract: GraniteFirstSftContract,
    comparison: ExperimentComparison,
    training: TrainingResults,
) -> str:
    delta = "n/a" if comparison.delta_accuracy is None else f"{comparison.delta_accuracy:+.4f}"
    g0 = "n/a" if comparison.g0_accuracy is None else f"{comparison.g0_accuracy:.4f}"
    g1 = "n/a" if comparison.g1_accuracy is None else f"{comparison.g1_accuracy:.4f}"
    gpu_name = training.environment.gpu_name if training.environment is not None else None
    lines = [
        f"# Granite 4.1 first SFT: {contract.experiment_id}",
        "",
        f"Measured result: **{comparison.conclusion}**.",
        "",
        contract.research_question,
        "",
        "This conclusion is the pre-registered interpretation rule applied to the",
        "paired held-out verdicts. It was not revised after seeing G0 or G1.",
        "",
        "| Arm | Accuracy |",
        "| --- | ---: |",
        f"| G0 base | {g0} |",
        f"| G1 SFT | {g1} |",
        "",
        (
            f"G0 → G1 accuracy delta: {delta}. "
            f"Outcomes: {comparison.n_improved} improved, {comparison.n_regressed} regressed, "
            f"{comparison.n_tied} tied, {comparison.n_invalid} invalid."
        ),
        "",
        (
            "Training loss is diagnostic only: "
            f"train_loss={training.train_loss}, validation_loss={training.validation_loss}."
        ),
        "",
        (
            f"Accepted records: {training.accepted_records}. "
            f"Accepted supervised tokens: {training.accepted_tokens}. "
            f"Wall-clock seconds: {training.wall_clock_seconds}. "
            f"Tokens per second: {training.tokens_per_second}."
        ),
        "",
        (
            f"Peak VRAM bytes: {training.peak_vram_bytes}. "
            f"Adapter bytes: {training.adapter_bytes}. "
            f"Adapter SHA-256: `{training.adapter_sha256}`. "
            f"Checkpoint bytes: {training.checkpoint_bytes}."
        ),
        "",
        f"GPU: `{gpu_name}`.",
        "",
        f"Model revision: `{contract.model_revision}`.",
        f"Tokenizer revision: `{contract.tokenizer_revision}`.",
        f"Split manifest SHA-256: `{contract.split.split_manifest_sha256}`.",
        f"Held-out SHA-256: `{contract.split.held_out_sha256}`.",
        f"Agoge commit: `{contract.agoge_commit}`.",
        f"Interpretation rule SHA-256: `{contract.interpretation_rule_sha256}`.",
        "",
        contract.interpretation_rule,
        "",
    ]
    return "\n".join(lines)


def render_failure_report(
    contract: GraniteFirstSftContract,
    *,
    stage: str,
    reason: str,
) -> str:
    stage_text = {
        "g0": "G0 failed. G1 training did not start.",
        "train": "G0 is preserved. G1 training did not complete.",
        "reload": "G1 training finished. Clean reload failed, so G1 was not scored.",
        "g1": "G1 reloaded. Held-out scoring failed, so no paired conclusion was issued.",
    }[stage]
    lines = [
        f"# Granite 4.1 first SFT: {contract.experiment_id}",
        "",
        "Measured result: the run stopped before a paired held-out conclusion.",
        "",
        stage_text,
        "",
        reason,
        "",
        "The pre-registered interpretation rule was not applied to a partial comparison.",
        "",
        contract.research_question,
        "",
        f"Model revision: `{contract.model_revision}`.",
        f"Agoge commit: `{contract.agoge_commit}`.",
        f"Split manifest SHA-256: `{contract.split.split_manifest_sha256}`.",
        "",
    ]
    return "\n".join(lines)


def render_reproduction(contract: GraniteFirstSftContract, experiment_dir: Path) -> str:
    contract_path = experiment_dir / "experiment-contract.json"
    lines = [
        "# Reproduction",
        "",
        (
            f"Experiment `{contract.experiment_id}` uses budget "
            f"`{contract.budget.budget_id}` and Agoge commit `{contract.agoge_commit}`."
        ),
        "",
        "Freeze the contract before inspecting G0. Do not edit it after results exist.",
        "",
        "```bash",
        "uv run agoge freeze-granite-first-sft \\",
        f"  --experiment-id {contract.experiment_id} \\",
        f"  --output-dir reports/{REPORT_PARENT_NAME}/{contract.experiment_id} \\",
        f"  --split-manifest {contract.split.split_manifest_path} \\",
        f"  --qualification-report {contract.blockers.qualification_report_path}",
        "",
        "uv run agoge run-granite-first-sft \\",
        f"  --contract {contract_path}",
        "```",
        "",
        "`run-granite-first-sft` evaluates untouched G0, then trains G1, then",
        "clean-reloads the adapter, then scores G1 under the same held-out task IDs,",
        "serializer, decoding settings, and scoring version. It requires a local CUDA",
        "GPU. This repository's CI does not execute that measurement.",
        "",
    ]
    return "\n".join(lines)


def publish_terminal_files(
    experiment_dir: Path,
    *,
    contract: GraniteFirstSftContract,
    comparison: ExperimentComparison,
    training: TrainingResults,
    report: str,
) -> None:
    _publish(experiment_dir / "comparison.json", _json_bytes(comparison))
    _publish(experiment_dir / "regressions.jsonl", _regression_bytes(comparison))
    _publish(experiment_dir / "training-results.json", _json_bytes(training))
    _publish(experiment_dir / "report.md", report.encode("utf-8"))
    _publish(
        experiment_dir / "reproduction.md",
        render_reproduction(contract, experiment_dir).encode("utf-8"),
    )


def _json_bytes(model: ExperimentComparison | TrainingResults) -> bytes:
    return canonical_json_bytes(model.model_dump(mode="json")) + b"\n"


def _regression_bytes(comparison: ExperimentComparison) -> bytes:
    chunks: list[bytes] = []
    for pair in comparison.pairs:
        if pair.outcome != "regressed":
            continue
        chunks.append(canonical_json_bytes(pair.model_dump(mode="json")) + b"\n")
    return b"".join(chunks)


def _publish(path: Path, payload: bytes) -> None:
    publish_bytes_noreplace(path, payload, refusal=_REFUSAL, writer=write_fsynced_bytes)
