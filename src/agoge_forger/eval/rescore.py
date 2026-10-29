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
import tempfile
from pathlib import Path
from typing import Any, Literal

from .._atomic_directory import rename_noreplace
from ..split_contract import canonical_json_bytes, sha256_bytes
from .bundle import _comparison_payload
from .compare import compare_arms
from .score import (
    ArmMetrics,
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
_ARM_ROLES: dict[str, Literal["causal_base", "causal_sft"]] = {
    "base": "causal_base",
    "sft": "causal_sft",
}
_CONTRACT_FILE = "contract.json"


def read_generations(path: Path) -> tuple[GenerationRecord, ...]:
    return _parse_generations(path.read_bytes(), path)


def _parse_generations(content: bytes, path: Path) -> tuple[GenerationRecord, ...]:
    records: list[GenerationRecord] = []
    for number, line in enumerate(content.decode("utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            records.append(GenerationRecord.model_validate(json.loads(line)))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: not a generation record: {exc}") from exc
    return tuple(records)


def _read_generation_files(bundle_dir: Path) -> dict[str, bytes]:
    """Read each generations file exactly once; the same bytes are hashed and scored."""

    contents: dict[str, bytes] = {}
    for arm, relative in _GENERATION_FILES.items():
        path = bundle_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"generations not found: {path}")
        contents[arm] = path.read_bytes()
    return contents


def _require_contract_tasks(
    bundle_dir: Path, records: dict[str, tuple[GenerationRecord, ...]]
) -> None:
    """Fail closed unless both arms cover exactly the contract's frozen task set."""

    contract_path = bundle_dir / _CONTRACT_FILE
    if not contract_path.is_file():
        raise FileNotFoundError(f"bundle contract not found: {contract_path}")
    expected = sorted(json.loads(contract_path.read_text(encoding="utf-8"))["logical_task_ids"])
    for arm, arm_records in records.items():
        if sorted(record.task_id for record in arm_records) != expected:
            raise ValueError(
                f"{_GENERATION_FILES[arm]} does not cover exactly the {len(expected)} task(s) "
                f"in {contract_path.name}; refusing to re-score a partial or altered bundle"
            )


def rescore_paired_bundle(
    bundle_dir: Path, *, scoring_version: str, output: Path | None = None
) -> Path:
    """Score ``bundle_dir``'s base and SFT generations with ``scoring_version``.

    Both arms must cover exactly the task set in the bundle's ``contract.json``. Returns
    the path of the new comparison file. An existing file is never replaced.
    """

    require_supported_scoring_version(scoring_version)
    contents = _read_generation_files(bundle_dir)
    records = {
        arm: _parse_generations(contents[arm], bundle_dir / _GENERATION_FILES[arm])
        for arm in _GENERATION_FILES
    }
    _require_contract_tasks(bundle_dir, records)
    scored = {
        arm: score_arm(_ARM_ROLES[arm], records[arm], scoring_version=scoring_version)
        for arm in _GENERATION_FILES
    }
    comparison = compare_arms(scored["base"][0], scored["sft"][0], scoring_version=scoring_version)
    payload: dict[str, Any] = _comparison_payload(comparison)
    payload.update(_post_hoc_fields(contents, scored))
    destination = output or bundle_dir / f"comparison.{scoring_version}.json"
    _write_new_file(destination, canonical_json_bytes(payload) + b"\n")
    return destination


def _post_hoc_fields(
    contents: dict[str, bytes], scored: dict[str, tuple[tuple[ExampleScore, ...], ArmMetrics]]
) -> dict[str, Any]:
    return {
        "post_hoc": RESCORE_NOTE,
        "rescored_from": {
            _GENERATION_FILES[arm]: sha256_bytes(content) for arm, content in contents.items()
        },
        "arms": {arm: metrics.model_dump(mode="json") for arm, (_, metrics) in scored.items()},
        "scores": {
            arm: [_score_row(score) for score in scores] for arm, (scores, _) in scored.items()
        },
    }


def _score_row(score: ExampleScore) -> dict[str, object]:
    row: dict[str, object] = {"task_id": score.task_id, "verdict": score.verdict}
    for key in ("reason", "n_examples", "n_failed", "error"):
        value = getattr(score, key)
        if value is not None:
            row[key] = value
    return row


def _write_new_file(path: Path, content: bytes) -> None:
    """Publish ``content`` at ``path`` atomically; never replace an existing file.

    The bytes go to a temporary file beside the destination and are renamed into place
    without replacing, so a failed write leaves nothing behind that would block a retry.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        rename_noreplace(Path(temporary), path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
