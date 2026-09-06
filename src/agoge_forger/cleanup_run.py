"""Reclaim disk from a finished training run by pruning `checkpoint-*` trees.

`agoge cleanup-run adapters/<run_name>` removes the trainer recovery snapshots
that are dead weight once a run has produced a final adapter or a merged model.
It never touches anything outside `checkpoint-*`, so the final adapter, the
tokenizer files, `artifact_index.json`, `runs/<run_name>/` manifests, and
`merged/<run_name>` are out of reach by construction rather than by denylist.

Planning and deletion are separate: `plan_cleanup` decides and accounts without
touching the filesystem (that is exactly what `--dry-run` runs), and
`execute_cleanup` is the only function here that removes anything.
"""

from __future__ import annotations

import re
import shutil
from enum import Enum
from pathlib import Path
from typing import Any

from .artifacts.producer_provenance import producer_provenance_from_adapter
from .artifacts.safetensors_io import write_artifact_index
from .logging import logger
from .path_safety import resolve_existing_path
from .run_status import build_run_status
from .train.checkpoints import CHECKPOINT_RE, checkpoint_step, list_valid_checkpoints
from .train.preflight import BYTES_PER_GB, directory_size_bytes

CLEANUP_SCHEMA_VERSION = 1

_ARTIFACT_INDEX_NAME = "artifact_index.json"

# A checkpoint directory that fails validation is still reclaimable garbage, so
# candidates are matched on the directory name alone. `list_valid_checkpoints`
# is used only to choose which checkpoints are safe to *keep*.
_CANDIDATE_RE: re.Pattern[str] = CHECKPOINT_RE


class CleanupFormat(str, Enum):
    """Supported `agoge cleanup-run --format` renderings."""

    json = "json"
    table = "table"


def _candidate_dirs(run_dir: Path) -> list[Path]:
    """Every `checkpoint-N` entry under the run directory, ascending by step."""
    candidates = [
        entry
        for entry in run_dir.iterdir()
        if _CANDIDATE_RE.match(entry.name) and (entry.is_dir() or entry.is_symlink())
    ]
    candidates.sort(key=checkpoint_step)
    return candidates


def _keep_set(run_dir: Path, keep_latest: int, *, allow_unsafe: bool) -> set[Path]:
    """The N highest-step checkpoints worth keeping as a resume point.

    Only *valid* checkpoints qualify: keeping a half-written snapshot would
    reclaim less disk while leaving nothing you could actually resume from.
    """
    if keep_latest <= 0:
        return set()
    valid = list_valid_checkpoints(run_dir, allow_unsafe=allow_unsafe)
    return {path.resolve() for path in valid[-keep_latest:]}


def _guard(report: dict[str, Any]) -> bool:
    """True when a final artifact survives cleanup, so pruning is recoverable."""
    return bool(report["final_adapter"]["present"] or report["merged_model"]["present"])


def plan_cleanup(
    run_dir: str,
    *,
    keep_latest: int = 0,
    allow_unsafe: bool = False,
    force: bool = False,
    merged_dir: str | None = None,
) -> dict[str, Any]:
    """Decide what cleanup would remove, without removing anything.

    Raises ValueError when the run directory is a symlink, or when no final
    artifact exists and ``force`` was not passed.
    """
    if keep_latest < 0:
        raise ValueError(f"--keep-latest must not be negative: {keep_latest}")

    logical_run_dir = Path(run_dir).expanduser()
    # path_safety guards '..' but explicitly not symlink escape (see
    # tests/test_path_safety.py), and it resolves the link away, so a
    # destructive command has to refuse the link itself before resolving.
    if logical_run_dir.is_symlink():
        raise ValueError(f"Refusing to clean a symlinked run directory: {logical_run_dir}")

    resolved_run_dir = resolve_existing_path(run_dir, must_be_dir=True)
    # Pass the caller's path so merged/<run_name> sibling discovery keeps the
    # documented adapters/<run_name> layout, matching build_run_status.
    report = build_run_status(run_dir, merged_dir=merged_dir, allow_unsafe=allow_unsafe)

    recoverable = _guard(report)
    if not recoverable and not force:
        raise ValueError(
            f"No final adapter or merged model found under {resolved_run_dir}; "
            "the checkpoints are the only recoverable artifact. "
            "Re-run with --force to remove them anyway."
        )

    keep = _keep_set(resolved_run_dir, keep_latest, allow_unsafe=allow_unsafe)
    removable: list[dict[str, Any]] = []
    kept: list[str] = []
    skipped: list[dict[str, str]] = []

    for candidate in _candidate_dirs(resolved_run_dir):
        if candidate.is_symlink():
            skipped.append({"path": str(candidate), "reason": "symlink"})
            continue
        if candidate.resolve() in keep:
            kept.append(str(candidate))
            continue
        removable.append(
            {
                "path": str(candidate),
                "step": checkpoint_step(candidate),
                "bytes": directory_size_bytes(str(candidate)),
            }
        )

    return {
        "schema_version": CLEANUP_SCHEMA_VERSION,
        "run_dir": str(resolved_run_dir),
        "run_name": logical_run_dir.name or resolved_run_dir.name,
        "dry_run": True,
        "keep_latest": keep_latest,
        "allow_unsafe_serialization": allow_unsafe,
        "guard": {"final_artifact": recoverable, "forced": bool(force and not recoverable)},
        "removed": removable,
        "kept": kept,
        "skipped": skipped,
        "failed": [],
        "bytes_reclaimed": sum(entry["bytes"] for entry in removable),
        "artifact_index_rewritten": False,
    }


def _refresh_artifact_index(run_dir: Path, provenance: Any) -> bool:
    """Rebuild `artifact_index.json` over whatever survived the deletion.

    The index written at the end of training hashes every file under the run
    directory, checkpoints included, and the evaluation contract requires the
    index and the live file set to match exactly
    (`eval/_artifact_snapshot.py::_require_complete_index`). Leaving a stale
    index behind would therefore break contract validation outright.
    """
    if provenance is None:
        return False
    try:
        write_artifact_index(str(run_dir), producer_provenance=provenance)
    except (OSError, ValueError):
        return False
    return True


def _sealed_provenance(run_dir: Path) -> Any:
    """Provenance sealed into the run's index, or None when there is no index."""
    if not (run_dir / _ARTIFACT_INDEX_NAME).is_file():
        return None
    try:
        return producer_provenance_from_adapter(run_dir)
    except (OSError, ValueError):
        # An unreadable or provenance-less index is not something cleanup should
        # fail on; it just means there is no index worth maintaining.
        return None


def execute_cleanup(plan: dict[str, Any]) -> dict[str, Any]:
    """Remove the checkpoints named by ``plan`` and report what actually went.

    Deletion is per-checkpoint: one failure is recorded and the rest still
    proceed, and the returned report accounts only for directories that were
    genuinely removed. Nothing here logs — every human-facing line comes from
    `log_cleanup`, so a caller emitting JSON keeps stdout parseable.
    """
    run_dir = Path(plan["run_dir"])
    provenance = _sealed_provenance(run_dir)

    removed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for entry in plan["removed"]:
        target = Path(entry["path"])
        try:
            shutil.rmtree(target)
        except OSError as exc:
            failed.append({"path": entry["path"], "error": str(exc)})
            continue
        removed.append(entry)

    report = dict(plan)
    report["dry_run"] = False
    report["removed"] = removed
    report["failed"] = failed
    report["bytes_reclaimed"] = sum(item["bytes"] for item in removed)
    # Rebuild from the state that actually exists, not the state that was
    # planned, so a partial failure still leaves a truthful index.
    report["artifact_index_rewritten"] = (
        _refresh_artifact_index(run_dir, provenance) if removed else False
    )
    return report


def log_cleanup(report: dict[str, Any]) -> None:
    """Log the operator-facing account of a planned or completed cleanup.

    This is the only place in the module that writes to the logger. The shared
    handler renders to stdout, so a caller emitting the JSON report must skip
    this or the payload stops being parseable.
    """
    prefix = "[dry-run] " if report["dry_run"] else ""
    removed = report["removed"]
    gib = report["bytes_reclaimed"] / BYTES_PER_GB
    verb = "would remove" if report["dry_run"] else "removed"

    if not removed:
        logger.info("%snothing to remove under %s", prefix, report["run_dir"])
    else:
        logger.info(
            "%s%s %d checkpoint director%s under %s (~%.2f GiB)",
            prefix,
            verb,
            len(removed),
            "y" if len(removed) == 1 else "ies",
            report["run_dir"],
            gib,
        )
        for entry in removed:
            logger.info("%s  %s (%.2f GiB)", prefix, entry["path"], entry["bytes"] / BYTES_PER_GB)

    for path in report["kept"]:
        logger.info("%skept %s", prefix, path)
    for entry in report["skipped"]:
        logger.info("%sskipped %s (%s)", prefix, entry["path"], entry["reason"])
    for entry in report["failed"]:
        logger.error("failed %s: %s", entry["path"], entry["error"])

    if report["artifact_index_rewritten"]:
        logger.warning(
            "Rewrote %s over the surviving files; its sha256 changed, so publish any "
            "evaluation contract for this run after cleanup, not before.",
            _ARTIFACT_INDEX_NAME,
        )

    if report["guard"]["forced"]:
        logger.warning(
            "--force removed the only recoverable artifact under %s; this run can no "
            "longer be resumed or exported.",
            report["run_dir"],
        )


def format_cleanup_table(report: dict[str, Any]) -> str:
    """Render a cleanup report as an aligned `key: value` block."""
    rows: list[tuple[str, str]] = [
        ("schema_version", str(report["schema_version"])),
        ("run_name", str(report["run_name"])),
        ("run_dir", str(report["run_dir"])),
        ("dry_run", "yes" if report["dry_run"] else "no"),
        ("keep_latest", str(report["keep_latest"])),
        ("final_artifact", "yes" if report["guard"]["final_artifact"] else "no"),
        ("forced", "yes" if report["guard"]["forced"] else "no"),
        ("checkpoints_removed", str(len(report["removed"]))),
        ("checkpoints_kept", str(len(report["kept"]))),
        ("checkpoints_skipped", str(len(report["skipped"]))),
        ("checkpoints_failed", str(len(report["failed"]))),
        ("bytes_reclaimed", str(report["bytes_reclaimed"])),
        ("gib_reclaimed", f"{report['bytes_reclaimed'] / BYTES_PER_GB:.2f}"),
        ("artifact_index_rewritten", "yes" if report["artifact_index_rewritten"] else "no"),
    ]
    width = max(len(label) for label, _ in rows) + 1
    return "\n".join(f"{label + ':':<{width}} {value}" for label, value in rows)
