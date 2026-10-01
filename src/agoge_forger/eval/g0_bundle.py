"""Atomic publication of a G0-only held-out base evaluation bundle."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .._atomic_directory import rename_noreplace, require_rename_noreplace_support
from .._source_snapshot import nearest_existing_output_ancestor
from ..split_contract import canonical_json_bytes
from .experiment_contract import G0EvaluationContract
from .score import ArmMetrics, ExampleScore, GenerationRecord

G0_BUNDLE_FILES = (
    "g0-contract.json",
    "g0-base/generations.jsonl",
    "g0-base/metrics.json",
)


@dataclass(frozen=True)
class G0EvalBundle:
    contract: G0EvaluationContract
    generations: tuple[GenerationRecord, ...]
    metrics: ArmMetrics
    scores: tuple[ExampleScore, ...]


def publish_g0_eval_bundle(output_dir: Path, bundle: G0EvalBundle) -> Path:
    destination = output_dir.expanduser()
    if os.path.lexists(destination):
        raise FileExistsError(
            "refusing to overwrite G0 evaluation bundle because output path already exists: "
            f"{destination}"
        )
    staging_parent = nearest_existing_output_ancestor(destination)
    require_rename_noreplace_support(staging_parent)
    with tempfile.TemporaryDirectory(prefix=".agoge-g0-staging-", dir=staging_parent) as staging:
        staged = Path(staging) / "bundle"
        staged.mkdir()
        _write_bundle(staged, bundle)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rename_noreplace(staged, destination)
    return destination


def _write_bundle(staged: Path, bundle: G0EvalBundle) -> None:
    (staged / "g0-base").mkdir()
    _exclusive_write(
        staged / "g0-contract.json",
        canonical_json_bytes(bundle.contract.model_dump(mode="json")) + b"\n",
    )
    _write_jsonl(staged / "g0-base" / "generations.jsonl", bundle.generations)
    _exclusive_write(
        staged / "g0-base" / "metrics.json",
        canonical_json_bytes(bundle.metrics.model_dump(mode="json")) + b"\n",
    )


def _write_jsonl(path: Path, rows: tuple[GenerationRecord, ...]) -> None:
    chunks = [canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows]
    _exclusive_write(path, b"".join(chunks))


def _exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
