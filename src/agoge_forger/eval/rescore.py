"""Post-hoc re-scoring of a published paired evaluation bundle.

``agoge rescore-held-out`` re-reads ``base/generations.jsonl`` and
``sft/generations.jsonl`` from an existing bundle and scores them with another
registered scorer. Nothing is regenerated and nothing in the bundle is modified; the
result is one new JSON file (``comparison.<scoring_version>.json``), so several
scorers can coexist next to the original ``comparison.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..split_contract import canonical_json_bytes, sha256_file
from .bundle import _comparison_payload
from .compare import compare_arms
from .score import (
    ExampleScore,
    GenerationRecord,
    require_supported_scoring_version,
    score_arm,
)

RESCORE_NOTE = (
    "Post-hoc re-score of already-published generations with a scorer that was not named in "
    "the experiment contract; it does not replace the pre-registered conclusion."
)
_GENERATION_FILES = {"base": "base/generations.jsonl", "sft": "sft/generations.jsonl"}


def read_generations(path: Path) -> tuple[GenerationRecord, ...]:
    records: list[GenerationRecord] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(GenerationRecord.model_validate(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not a generation record: {exc}") from exc
    return tuple(records)


def rescore_paired_bundle(
    bundle_dir: Path, *, scoring_version: str, output: Path | None = None
) -> Path:
    """Score ``bundle_dir``'s base and SFT generations with ``scoring_version``.

    Returns the path of the new comparison file. An existing file is never replaced.
    """

    require_supported_scoring_version(scoring_version)
    paths = {arm: bundle_dir / relative for arm, relative in _GENERATION_FILES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"generations not found: {path}")
    base_scores, base_metrics = score_arm(
        "causal_base", read_generations(paths["base"]), scoring_version=scoring_version
    )
    sft_scores, sft_metrics = score_arm(
        "causal_sft", read_generations(paths["sft"]), scoring_version=scoring_version
    )
    comparison = compare_arms(base_scores, sft_scores, scoring_version=scoring_version)
    payload: dict[str, Any] = _comparison_payload(comparison)
    payload["post_hoc"] = RESCORE_NOTE
    payload["rescored_from"] = {
        _GENERATION_FILES[arm]: sha256_file(path) for arm, path in paths.items()
    }
    payload["arms"] = {
        "base": base_metrics.model_dump(mode="json"),
        "sft": sft_metrics.model_dump(mode="json"),
    }
    payload["scores"] = {
        "base": [_score_row(score) for score in base_scores],
        "sft": [_score_row(score) for score in sft_scores],
    }
    destination = output or bundle_dir / f"comparison.{scoring_version}.json"
    _write_new_file(destination, canonical_json_bytes(payload) + b"\n")
    return destination


def _score_row(score: ExampleScore) -> dict[str, object]:
    row: dict[str, object] = {"task_id": score.task_id, "verdict": score.verdict}
    for key in ("reason", "n_examples", "n_failed", "error"):
        value = getattr(score, key)
        if value is not None:
            row[key] = value
    return row


def _write_new_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:  # refuses to overwrite an existing result
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
