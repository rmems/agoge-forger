"""Atomic publication of the paired held-out evaluation bundle."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .._atomic_directory import rename_noreplace, require_rename_noreplace_support
from .._source_snapshot import nearest_existing_output_ancestor
from ..split_contract import canonical_json_bytes
from ._artifact_schema import FrozenEvaluationModel
from .compare import ComparisonSummary
from .contract import PairedEvaluationContract
from .score import ArmMetrics, ExampleScore, GenerationRecord

BUNDLE_FILES = (
    "contract.json",
    "base/generations.jsonl",
    "base/metrics.json",
    "sft/generations.jsonl",
    "sft/metrics.json",
    "comparison.json",
    "regressions.jsonl",
    "report.md",
)


def publish_eval_bundle(
    output_dir: Path,
    *,
    contract: PairedEvaluationContract,
    base_generations: tuple[GenerationRecord, ...],
    sft_generations: tuple[GenerationRecord, ...],
    base_metrics: ArmMetrics,
    sft_metrics: ArmMetrics,
    comparison: ComparisonSummary,
    base_scores: tuple[ExampleScore, ...],
    sft_scores: tuple[ExampleScore, ...],
) -> Path:
    destination = output_dir.expanduser()
    if os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite evaluation bundle because output path already exists: {destination}"
        )
    staging_parent = nearest_existing_output_ancestor(destination)
    require_rename_noreplace_support(staging_parent)
    with tempfile.TemporaryDirectory(prefix=".agoge-eval-staging-", dir=staging_parent) as staging:
        staged = Path(staging) / "bundle"
        staged.mkdir()
        _write_bundle(
            staged,
            contract=contract,
            base_generations=base_generations,
            sft_generations=sft_generations,
            base_metrics=base_metrics,
            sft_metrics=sft_metrics,
            comparison=comparison,
            base_scores=base_scores,
            sft_scores=sft_scores,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        rename_noreplace(staged, destination)
    return destination


def render_report(
    contract: PairedEvaluationContract,
    base_metrics: ArmMetrics,
    sft_metrics: ArmMetrics,
    comparison: ComparisonSummary,
) -> str:
    run_name = "held-out"
    lines = [
        f"# Held-out paired evaluation: {run_name}",
        "",
        f"**Conclusion:** {comparison.conclusion}",
        "",
        (
            "Held-out membership is the frozen split digest "
            f"`{contract.held_out_split_sha256}`. This run did not re-split."
        ),
        "",
        (
            f"Objective scorer: `{comparison.scoring_version}`. "
            "Heuristic or model-judge signals were not used."
        ),
        "",
        "| Arm | Tasks | Scored | Correct | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
        _metrics_row("base", base_metrics),
        _metrics_row("sft", sft_metrics),
        "",
        (
            "Paired outcomes: "
            f"{comparison.n_improved} improved, {comparison.n_regressed} regressed, "
            f"{comparison.n_tied} tied, {comparison.n_invalid} invalid."
        ),
        "",
        f"Accuracy delta (SFT − base, comparable tasks): {_format_delta(comparison.delta_accuracy)}",
        "",
        f"Logical task-set SHA-256: `{contract.logical_task_set_sha256}`",
        f"Scoring version: `{contract.base.scoring_version}`",
        f"Decoding seed: `{contract.base.decoding.seed}`",
        "",
    ]
    return "\n".join(lines)


def _write_bundle(
    staged: Path,
    *,
    contract: PairedEvaluationContract,
    base_generations: tuple[GenerationRecord, ...],
    sft_generations: tuple[GenerationRecord, ...],
    base_metrics: ArmMetrics,
    sft_metrics: ArmMetrics,
    comparison: ComparisonSummary,
    base_scores: tuple[ExampleScore, ...],
    sft_scores: tuple[ExampleScore, ...],
) -> None:
    (staged / "base").mkdir()
    (staged / "sft").mkdir()
    _exclusive_write(
        staged / "contract.json", canonical_json_bytes(contract.model_dump(mode="json")) + b"\n"
    )
    _write_jsonl(staged / "base" / "generations.jsonl", base_generations)
    _exclusive_write(
        staged / "base" / "metrics.json",
        canonical_json_bytes(base_metrics.model_dump(mode="json")) + b"\n",
    )
    _write_jsonl(staged / "sft" / "generations.jsonl", sft_generations)
    _exclusive_write(
        staged / "sft" / "metrics.json",
        canonical_json_bytes(sft_metrics.model_dump(mode="json")) + b"\n",
    )
    _exclusive_write(
        staged / "comparison.json",
        canonical_json_bytes(_comparison_payload(comparison)) + b"\n",
    )
    _write_jsonl(
        staged / "regressions.jsonl",
        _regression_rows(comparison, base_scores, sft_scores, base_generations, sft_generations),
    )
    _exclusive_write(
        staged / "report.md",
        render_report(contract, base_metrics, sft_metrics, comparison).encode("utf-8"),
    )


def _comparison_payload(comparison: ComparisonSummary) -> dict[str, object]:
    payload = comparison.model_dump(mode="json")
    payload["outcomes"] = {item.task_id: item.outcome for item in comparison.outcomes}
    payload["pairs"] = [item.model_dump(mode="json") for item in comparison.outcomes]
    return payload


def _regression_rows(
    comparison: ComparisonSummary,
    base_scores: tuple[ExampleScore, ...],
    sft_scores: tuple[ExampleScore, ...],
    base_generations: tuple[GenerationRecord, ...],
    sft_generations: tuple[GenerationRecord, ...],
) -> tuple[dict[str, object], ...]:
    base_by_id = {score.task_id: score for score in base_scores}
    sft_by_id = {score.task_id: score for score in sft_scores}
    base_gen = {record.task_id: record for record in base_generations}
    sft_gen = {record.task_id: record for record in sft_generations}
    rows: list[dict[str, object]] = []
    for item in comparison.outcomes:
        if item.outcome != "regressed":
            continue
        rows.append(
            {
                "task_id": item.task_id,
                "outcome": item.outcome,
                "base": _score_payload(base_by_id[item.task_id], base_gen[item.task_id]),
                "sft": _score_payload(sft_by_id[item.task_id], sft_gen[item.task_id]),
            }
        )
    return tuple(rows)


def _score_payload(score: ExampleScore, generation: GenerationRecord) -> dict[str, object]:
    return {
        "verdict": score.verdict,
        "predicted_completion": score.predicted_completion,
        "expected_completion": score.expected_completion,
        "raw_text": generation.raw_text,
        "reason": score.reason,
    }


def _write_jsonl(path: Path, rows: tuple[object, ...]) -> None:
    chunks: list[bytes] = []
    for row in rows:
        chunks.append(canonical_json_bytes(_row_payload(row)) + b"\n")
    _exclusive_write(path, b"".join(chunks))


def _row_payload(row: object) -> object:
    if isinstance(row, FrozenEvaluationModel):
        return row.model_dump(mode="json")
    return row


def _exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _metrics_row(label: str, metrics: ArmMetrics) -> str:
    accuracy = "n/a" if metrics.accuracy is None else f"{metrics.accuracy:.4f}"
    return (
        f"| {label} | {metrics.n_tasks} | {metrics.n_scored} | {metrics.n_correct} | {accuracy} |"
    )


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.4f}"
