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

import json
import os
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .artifacts.producer_provenance import producer_provenance_from_adapter
from .artifacts.safetensors_io import sha256_file, write_artifact_index
from .logging import logger
from .path_safety import resolve_existing_path
from .run_status import _escape_controls, build_run_status
from .train.checkpoints import CHECKPOINT_RE, checkpoint_step
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


@dataclass(frozen=True)
class CleanupOptions:
    """Everything `plan_cleanup` needs beyond the run directory itself.

    Bundled rather than passed as five keywords, matching how the CLI already
    groups wide option sets (`_FreezeSplitOptions`, `_SmokeEnvInputs`).
    """

    keep_latest: int = 0
    allow_unsafe: bool = False
    force: bool = False
    merged_dir: str | None = None


def _skip_reason(candidate: Path) -> str | None:
    """Why this checkpoint must not be deleted or counted as a keep slot."""
    if candidate.is_symlink():
        return "symlink"
    if contains_mount(candidate):
        return "mount"
    return None


def _keep_set(candidates: list[Path], steps: list[int], keep_latest: int) -> set[Path]:
    """The N highest-step checkpoints worth keeping as a resume point.

    Selected from the directories actually on disk, matched to the valid steps
    `build_run_status` already scanned for, so the keep set costs no extra
    directory walk. The paths must come from the real entries rather than being
    rebuilt from a step number: `checkpoint-001` parses as step 1, and a
    reconstructed `checkpoint-1` would match nothing, so `--keep-latest 1` would
    delete the very snapshot it was asked to keep.

    Only valid, non-skippable checkpoints qualify. A symlink or mount can still
    parse as a valid step (the link target looks like a resume point) but is
    later skipped; if it consumed a keep slot, `--keep-latest 1` would delete
    every real checkpoint. Callers must pass already-filtered candidates.
    """
    if keep_latest <= 0:
        return set()
    valid = {step for step in steps}
    keepable = [path for path in candidates if checkpoint_step(path) in valid]
    return {path.resolve() for path in keepable[-keep_latest:]}


def _guard(report: dict[str, Any]) -> bool:
    """True when a *usable* final artifact survives cleanup.

    `final_adapter.present` only says the expected filenames exist, so a run
    interrupted after `save_pretrained` but before the tokenizer or the index was
    written still looks finished. `export.ready` is the validated answer, but on
    its own it is not enough: with no run-root adapter it reports the latest
    *checkpoint* as the export source, which is precisely what is about to be
    deleted. So the guard requires a ready export whose source is the final
    adapter -- something that survives the prune.

    A merged model counts only when its sealed provenance matches the run's.
    `merged/<run_name>` is a naming convention, so a leftover directory from an
    older experiment of the same name would otherwise satisfy a destructive
    guard for a run it has nothing to do with.
    """
    export = report["export"]
    adapter_ready = bool(
        export["ready"]
        and export["source_kind"] == "final_adapter"
        # `export.ready` validates the adapter config and weights but not the
        # sealed index, while `export-final-model` unconditionally reads
        # provenance off the adapter. Without this the guard can pass on a run
        # the official export path would refuse -- deleting the checkpoints of a
        # run interrupted after the weights but before artifact_index.json.
        and report["final_adapter"]["has_provenance"]
    )
    return bool(adapter_ready or report["merged_model"]["matches_run"])


def _require_recoverable(recoverable: bool, run_dir: Path, *, force: bool) -> None:
    """Stop unless something usable survives the prune, or the operator insisted."""
    if recoverable or force:
        return
    raise ValueError(
        f"No final adapter or merged model found under {run_dir}; "
        "the checkpoints are the only recoverable artifact. "
        "Re-run with --force to remove them anyway."
    )


def _has_provenance(path: str | Path) -> bool:
    """True when a sealed producer provenance can be read off `path`."""
    try:
        producer_provenance_from_adapter(path)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _merged_matches_run(run_dir: Path, merged_path: str | None) -> bool:
    """True when the merged model was sealed from this run's adapter.

    Compares the producer provenance on both sides. A merged directory with no
    provenance, or one carrying a different run's, does not count: the point of
    the guard is that something recoverable *from this run* survives.
    """
    if merged_path is None:
        return False
    try:
        merged = producer_provenance_from_adapter(merged_path)
        run = producer_provenance_from_adapter(run_dir)
    except (OSError, ValueError, TypeError):
        return False
    return bool(merged == run)


def _resolved_run_dir(run_dir: str) -> tuple[Path, Path]:
    """Return the (logical, resolved) run directory, refusing a symlinked one.

    path_safety guards '..' but explicitly not symlink escape (see
    tests/test_path_safety.py), and it resolves the link away, so a destructive
    command has to refuse the link itself before resolving.
    """
    logical = Path(run_dir).expanduser()
    # Check every component, not just the leaf: a symlinked *parent* redirects
    # the whole walk just as effectively as a symlinked run directory.
    probe = logical if logical.is_absolute() else Path.cwd() / logical
    for parent in (probe, *probe.parents):
        if parent.is_symlink():
            raise ValueError(f"Refusing to clean through a symlinked path: {parent}")
    return logical, resolve_existing_path(run_dir, must_be_dir=True)


def contains_mount(root: Path) -> bool:
    """True when `root` or anything beneath it is a mount point.

    `rmtree` descends into a mounted subdirectory and deletes its contents before
    failing on the busy mount itself, so testing only the checkpoint root would
    still let `checkpoint-100/cache` take data from another filesystem with it.
    """
    if os.path.ismount(root):
        return True
    for parent, dirs, _ in os.walk(root):
        if any(os.path.ismount(os.path.join(parent, name)) for name in dirs):
            return True
    return False


def _partition_candidates(
    candidates: list[Path], keep: set[Path]
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    """Split `checkpoint-*` entries into removable, kept, and skipped."""
    removable: list[dict[str, Any]] = []
    kept: list[str] = []
    skipped: list[dict[str, str]] = []
    for candidate in candidates:
        reason = _skip_reason(candidate)
        if reason is not None:
            skipped.append({"path": str(candidate), "reason": reason})
        elif candidate.resolve() in keep:
            kept.append(str(candidate))
        else:
            removable.append(
                {
                    "path": str(candidate),
                    "step": checkpoint_step(candidate),
                    "bytes": directory_size_bytes(str(candidate), follow_symlinks=False),
                }
            )
    return removable, kept, skipped


def plan_cleanup(run_dir: str, options: CleanupOptions | None = None) -> dict[str, Any]:
    """Decide what cleanup would remove, without removing anything.

    Raises ValueError when the run directory is a symlink, or when no final
    artifact exists and ``force`` was not passed.
    """
    options = options or CleanupOptions()
    if options.keep_latest < 0:
        raise ValueError(f"--keep-latest must not be negative: {options.keep_latest}")

    logical_run_dir, resolved_run_dir = _resolved_run_dir(run_dir)
    # Pass the caller's path so merged/<run_name> sibling discovery keeps the
    # documented adapters/<run_name> layout, matching build_run_status.
    report = build_run_status(
        run_dir, merged_dir=options.merged_dir, allow_unsafe=options.allow_unsafe
    )

    report["final_adapter"]["has_provenance"] = _has_provenance(resolved_run_dir)
    report["merged_model"]["matches_run"] = _merged_matches_run(
        resolved_run_dir, report["merged_model"]["path"]
    )
    recoverable = _guard(report)
    _require_recoverable(recoverable, resolved_run_dir, force=options.force)

    candidates = _candidate_dirs(resolved_run_dir)
    # Symlinks and mounts are skipped later; they must not consume keep slots.
    keepable = [path for path in candidates if _skip_reason(path) is None]
    keep = _keep_set(keepable, report["checkpoints"]["steps"], options.keep_latest)
    removable, kept, skipped = _partition_candidates(candidates, keep)

    return {
        "schema_version": CLEANUP_SCHEMA_VERSION,
        "run_dir": str(resolved_run_dir),
        "run_name": logical_run_dir.name or resolved_run_dir.name,
        "dry_run": True,
        "keep_latest": options.keep_latest,
        "allow_unsafe_serialization": options.allow_unsafe,
        "guard": {
            "final_artifact": recoverable,
            "forced": bool(options.force and not recoverable),
        },
        "removed": removable,
        "kept": kept,
        "skipped": skipped,
        "failed": [],
        "bytes_reclaimed": sum(entry["bytes"] for entry in removable),
        "artifact_index_rewritten": False,
    }


def _refresh_artifact_index(run_dir: Path, provenance: Any) -> tuple[bool, str | None]:
    """Rebuild `artifact_index.json` over whatever survived the deletion.

    The index written at the end of training hashes every file under the run
    directory, checkpoints included, and the evaluation contract requires the
    index and the live file set to match exactly
    (`eval/_artifact_snapshot.py::_require_complete_index`). Leaving a stale
    index behind would therefore break contract validation outright.

    Returns `(rewritten, error)`. A rewrite that fails is an error, not a quiet
    False: the checkpoints are already gone at that point, so the index on disk
    now lists files that do not exist and the caller has to report it.
    """
    if not (run_dir / _ARTIFACT_INDEX_NAME).is_file():
        # Nothing to maintain, and cleanup should not invent an index a run
        # never had.
        return False, None
    try:
        write_artifact_index(str(run_dir), producer_provenance=provenance)
    except (OSError, ValueError) as exc:
        return False, str(exc)
    return True, None


def _index_artifact_entries(
    run_dir: Path,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Load the old index's artifact list, or explain why it cannot be verified.

    Returns ``(entries, error)``. ``entries`` is None when there is no mapping
    to verify against (missing index, non-object JSON, no ``artifacts`` key).
    ``error`` is set when the index exists but its artifact list is the wrong
    shape, so a sealed provenance must not be carried forward blindly.
    """
    index_path = run_dir / _ARTIFACT_INDEX_NAME
    if not index_path.is_file():
        return None, None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    raw = payload.get("artifacts")
    if raw is None:
        return None, None
    if not isinstance(raw, list):
        return None, f"{_ARTIFACT_INDEX_NAME} artifacts is not a list"
    entries: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            return None, f"{_ARTIFACT_INDEX_NAME} contains a non-object artifact entry"
        entries.append(entry)
    return entries, None


def _unsafe_indexed_path(run_dir: Path, name: str) -> str | None:
    """Reject index paths that would hash a file outside the run directory."""
    if not name or Path(name).is_absolute() or ".." in Path(name).parts:
        return f"refuses to follow artifact path {name!r} outside {run_dir}"
    try:
        (run_dir / name).resolve().relative_to(run_dir.resolve())
    except (OSError, ValueError):
        return f"refuses to follow artifact path {name!r} outside {run_dir}"
    return None


def _unindexed_survivor(run_dir: Path, indexed: set[str]) -> str | None:
    """First non-checkpoint file on disk that the old index never recorded."""
    for root, dirs, files in os.walk(run_dir):
        dirs[:] = [name for name in dirs if not _CANDIDATE_RE.match(name)]
        for filename in files:
            if filename == _ARTIFACT_INDEX_NAME and Path(root) == run_dir:
                continue
            relative = Path(os.path.relpath(os.path.join(root, filename), run_dir)).as_posix()
            if relative not in indexed:
                return str(run_dir / relative)
    return None


def _verify_surviving_entries(run_dir: Path, entries: list[dict[str, Any]]) -> str | None:
    """Confirm survivors still match the old index, and that nothing extra appeared.

    Cleanup carries the old producer provenance into the rewritten index. A
    modified survivor, a missing non-checkpoint file, or a file added after
    sealing would otherwise be stamped with that attestation.
    """
    indexed: set[str] = set()
    for entry in entries:
        name = str(entry.get("file", ""))
        unsafe = _unsafe_indexed_path(run_dir, name)
        if unsafe is not None:
            return unsafe
        first = Path(name).parts[0] if Path(name).parts else ""
        target = run_dir / name
        if not _CANDIDATE_RE.match(first):
            indexed.add(Path(name).as_posix())
        if not target.is_file():
            if _CANDIDATE_RE.match(first):
                continue
            return f"{target} is missing but {_ARTIFACT_INDEX_NAME} records it"
        try:
            if sha256_file(str(target)) != entry.get("sha256"):
                return f"{target} changed since {_ARTIFACT_INDEX_NAME} was written"
        except OSError as exc:
            return f"could not verify {target}: {exc}"
    extra = _unindexed_survivor(run_dir, indexed)
    if extra is not None:
        return f"{extra} appeared after {_ARTIFACT_INDEX_NAME} was written"
    return None


def _sealed_provenance(run_dir: Path) -> Any:
    """Provenance sealed into the run's index, or None when there is no index."""
    if not (run_dir / _ARTIFACT_INDEX_NAME).is_file():
        return None
    try:
        return producer_provenance_from_adapter(run_dir)
    except (OSError, ValueError, TypeError):
        # An unreadable or provenance-less index is not something cleanup should
        # fail on; it just means there is no index worth maintaining. TypeError
        # is in the list because _load_artifact_index_payload raises it — not
        # ValueError — for an index that parses as valid JSON but is not an
        # object (`[]`, `"text"`, `3`).
        return None


def _remove_checkpoints(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Delete each planned checkpoint, returning what went and what refused.

    One failure does not abort the rest: a checkpoint held open by another
    process should not strand the others on disk.
    """
    removed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for entry in entries:
        path = Path(entry["path"])
        # Re-check at delete time: a symlink or mount swapped in after planning
        # would otherwise go straight to rmtree.
        reason = _skip_reason(path)
        if reason is not None:
            failed.append({"path": entry["path"], "error": f"became a {reason} after planning"})
            continue
        try:
            shutil.rmtree(path)
        except OSError as exc:
            failed.append({"path": entry["path"], "error": str(exc)})
        else:
            removed.append(entry)
    return removed, failed


def _reseal_index(
    run_dir: Path,
    provenance: Any,
    *,
    touched: bool,
) -> tuple[bool, str | None]:
    """Rebuild the artifact index over the state that actually exists.

    Keyed off whether deletion was attempted — including a failed ``rmtree``
    that may have removed some files — so a partial failure still leaves a
    truthful index. Nothing was touched means nothing to reseal.
    """
    if not touched:
        return False, None
    entries, shape_error = _index_artifact_entries(run_dir)
    if provenance is not None:
        if shape_error is not None:
            return False, shape_error
        if entries is None:
            return False, (
                f"{_ARTIFACT_INDEX_NAME} has sealed provenance but no verifiable artifacts list"
            )
        tampered = _verify_surviving_entries(run_dir, entries)
        if tampered is not None:
            return False, tampered
    return _refresh_artifact_index(run_dir, provenance)


def execute_cleanup(plan: dict[str, Any]) -> dict[str, Any]:
    """Remove the checkpoints named by ``plan`` and report what actually went.

    Deletion is per-checkpoint: one failure is recorded and the rest still
    proceed, and the returned report accounts only for directories that were
    genuinely removed. Nothing here logs — every human-facing line comes from
    `log_cleanup`, so a caller emitting JSON keeps stdout parseable.
    """
    run_dir = Path(plan["run_dir"])
    provenance = _sealed_provenance(run_dir)

    removed, failed = _remove_checkpoints(plan["removed"])
    rewritten, index_error = _reseal_index(run_dir, provenance, touched=bool(removed or failed))
    if index_error is not None:
        # The checkpoints are gone but the index still lists them, so the run's
        # own metadata is now wrong. That is a failed cleanup, not a success.
        failed.append({"path": str(run_dir / _ARTIFACT_INDEX_NAME), "error": index_error})

    report = dict(plan)
    report["dry_run"] = False
    report["removed"] = removed
    report["failed"] = failed
    report["bytes_reclaimed"] = sum(item["bytes"] for item in removed)
    report["artifact_index_rewritten"] = rewritten
    return report


def log_cleanup(report: dict[str, Any]) -> None:
    """Log the operator-facing account of a planned or completed cleanup.

    This is the only place in the module that writes to the logger. The shared
    handler renders to stdout, so a caller emitting the JSON report must skip
    this or the payload stops being parseable.
    """
    prefix = "[dry-run] " if report["dry_run"] else ""
    _log_removals(report, prefix)
    _log_untouched(report, prefix)
    _log_warnings(report)


def _log_path(value: object) -> str:
    return _escape_controls(str(value))


def _log_removals(report: dict[str, Any], prefix: str) -> None:
    removed = report["removed"]
    if not removed:
        logger.info("%snothing to remove under %s", prefix, _log_path(report["run_dir"]))
        return
    logger.info(
        "%s%s %d checkpoint director%s under %s (~%.2f GiB)",
        prefix,
        "would remove" if report["dry_run"] else "removed",
        len(removed),
        "y" if len(removed) == 1 else "ies",
        _log_path(report["run_dir"]),
        report["bytes_reclaimed"] / BYTES_PER_GB,
    )
    for entry in removed:
        logger.info(
            "%s  %s (%.2f GiB)",
            prefix,
            _log_path(entry["path"]),
            entry["bytes"] / BYTES_PER_GB,
        )


def _log_untouched(report: dict[str, Any], prefix: str) -> None:
    for path in report["kept"]:
        logger.info("%skept %s", prefix, _log_path(path))
    for entry in report["skipped"]:
        logger.info("%sskipped %s (%s)", prefix, _log_path(entry["path"]), entry["reason"])
    for entry in report["failed"]:
        logger.error("failed %s: %s", _log_path(entry["path"]), entry["error"])


def _force_left_nothing_usable(report: dict[str, Any]) -> bool:
    """True when --force deleted the last resume/export point."""
    if not report["guard"]["forced"]:
        return False
    if report["kept"]:
        return False
    return bool(report["removed"])


def _log_warnings(report: dict[str, Any]) -> None:
    if report["artifact_index_rewritten"]:
        logger.warning(
            "Rewrote %s over the surviving files; its sha256 changed, so publish any "
            "evaluation contract for this run after cleanup, not before.",
            _ARTIFACT_INDEX_NAME,
        )
    if _force_left_nothing_usable(report):
        logger.warning(
            "--force removed the only recoverable artifact under %s; this run can no "
            "longer be resumed or exported.",
            _log_path(report["run_dir"]),
        )


def format_cleanup_table(report: dict[str, Any]) -> str:
    """Render a cleanup report as an aligned `key: value` block."""
    rows: list[tuple[str, str]] = [
        ("schema_version", str(report["schema_version"])),
        # A run directory name can carry ANSI escapes; run-status sanitizes the
        # same two fields for the same reason.
        ("run_name", _escape_controls(str(report["run_name"]))),
        ("run_dir", _escape_controls(str(report["run_dir"]))),
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
