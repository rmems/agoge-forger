"""Operator-facing readiness report for a training run directory.

`agoge run-status adapters/<run_name>` answers the three questions an operator
asks about a run without loading a single model weight: can I resume it, can I
export it, and has it already been merged?

Every discovery rule lives in `train/checkpoints.py` (what counts as a valid
checkpoint, which artifact `export-final-model` would pick, which base model an
adapter was trained from) and in `artifacts/safetensors_io.py`. This module only
assembles their answers into one stable JSON document, so a readiness report can
never disagree with what training and export actually do.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .path_safety import resolve_existing_path
from .train.checkpoints import (
    _checkpoint_step,
    infer_base_model_from_adapter,
    infer_base_revision_from_adapter,
    is_adapter_artifact,
    list_valid_checkpoints,
    resolve_export_source,
)

SCHEMA_VERSION = 1

PathLike = str | Path

# A malformed or unreadable adapter_config.json must degrade to "unknown base
# model", never crash a status report. json.JSONDecodeError is a ValueError
# subclass; it is named here for the reader's benefit. AttributeError covers a
# file that parses as valid JSON but is not an object (`[]`, `"text"`, `3`), on
# which the checkpoint helpers' `.get(...)` call would otherwise raise.
_ADAPTER_CONFIG_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    json.JSONDecodeError,
)


class RunStatusFormat(str, Enum):
    """Supported `agoge run-status --format` renderings."""

    json = "json"
    table = "table"


def is_merged_model_dir(path: PathLike) -> bool:
    """Return True when `path` looks like an exported merged model directory.

    A merged model is a full `save_pretrained` tree: a `config.json` plus at
    least one safetensors weight file *directly in the directory*. The root-level
    requirement is what distinguishes a real merge from a tree that merely
    contains adapters further down (`checkpoint-10/adapter_model.safetensors`),
    which a recursive search would misreport as an already-merged model. Sharded
    exports still match, since `save_pretrained` writes every
    `model-0000N-of-0000M.safetensors` shard at the root alongside its index.
    """
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    if not (candidate / "config.json").is_file():
        return False
    return any(entry.is_file() for entry in candidate.glob("*.safetensors"))


def find_merged_model_dir(run_dir: Path, merged_dir: str | None = None) -> Path | None:
    """Locate the merged model exported from `run_dir`, if there is one.

    With `merged_dir` given, exactly that path is checked. Otherwise the
    conventional sibling layout is probed: `adapters/<run_name>` pairs with
    `merged/<run_name>`. A merged model that has not been exported yet is an
    expected answer rather than an error, so a missing directory yields None.
    """
    if merged_dir is not None:
        try:
            candidate = resolve_existing_path(merged_dir, must_be_dir=True)
        except (FileNotFoundError, ValueError):
            # Missing, not-a-dir, empty, or '..' traversal: no merge, not a crash.
            # Library callers get the same "absent" answer the CLI treats as
            # non-fatal for an explicit --merged-dir that does not resolve.
            return None
        return candidate if is_merged_model_dir(candidate) else None

    # Use the caller-supplied path, not a symlink-resolved one. If
    # adapters/<run> points at external storage, resolve() would probe
    # <target-grandparent>/merged/<target-basename> and miss the documented
    # sibling merged/<run_name>.
    conventional = run_dir.parent.parent / "merged" / run_dir.name
    return conventional if is_merged_model_dir(conventional) else None


def _as_str(value: PathLike | None) -> str | None:
    """Render a path for JSON: `str`, or None when there is nothing to report."""
    return None if value is None else str(value)


def _resolve_export(run_dir: Path, *, allow_unsafe: bool) -> tuple[str | None, str | None]:
    """Return the (source_path, source_kind) `export-final-model` would use."""
    try:
        source = resolve_export_source(run_dir=str(run_dir), allow_unsafe=allow_unsafe)
    except (ValueError, FileNotFoundError):
        return None, None
    kind = "final_adapter" if Path(source) == run_dir else "checkpoint"
    return source, kind


def _infer_base(adapter_path: PathLike | None) -> tuple[str | None, str | None]:
    """Read the base model id and pinned revision off an adapter, tolerantly."""
    if adapter_path is None:
        return None, None

    try:
        base_model: str | None = infer_base_model_from_adapter(adapter_path)
    except _ADAPTER_CONFIG_ERRORS:
        base_model = None

    try:
        base_revision = infer_base_revision_from_adapter(adapter_path)
    except _ADAPTER_CONFIG_ERRORS:
        base_revision = None

    return base_model, base_revision


def _adapter_config_usable(adapter_path: PathLike | None) -> bool:
    """True when adapter_config.json is a JSON object export can parse.

    A valid object that simply omits `base_model_name_or_path` is still
    exportable. A missing, unreadable, or non-object file is not:
    `export-final-model` will fail on that same file, so `export.ready`
    must be false.
    """
    if adapter_path is None:
        return False
    config_path = Path(adapter_path) / "adapter_config.json"
    try:
        payload = json.loads(config_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict)


def build_run_status(
    run_dir: str,
    *,
    merged_dir: str | None = None,
    allow_unsafe: bool = False,
) -> dict[str, Any]:
    """Build the JSON-serializable readiness report for one run directory.

    The path is re-resolved here even though `cli.py` already validated it, so
    direct library callers get the same traversal and existence guarantees the
    CLI boundary enforces (the same defense in depth as `merge_adapter`).

    Every key of the schema is always present; unknown values are None.
    """
    resolved_run_dir = resolve_existing_path(run_dir, must_be_dir=True)

    checkpoints = list_valid_checkpoints(resolved_run_dir, allow_unsafe=allow_unsafe)
    # Exactly the selection `resolve_resume_checkpoint` makes when
    # `resume_from_latest_checkpoint` is set: `find_latest_valid_checkpoint` is
    # the last element of this very list. Taking it from the list already in
    # hand — rather than rescanning — keeps `steps`, `valid_count`,
    # `latest_step` and `latest_path` describing one single observation of the
    # directory, so a checkpoint written mid-report cannot produce a report
    # whose `latest_step` is missing from its own `steps`.
    latest_checkpoint = checkpoints[-1] if checkpoints else None
    latest_step = None if latest_checkpoint is None else _checkpoint_step(latest_checkpoint)

    final_adapter_present = is_adapter_artifact(resolved_run_dir, allow_unsafe=allow_unsafe)
    export_source, export_kind = _resolve_export(resolved_run_dir, allow_unsafe=allow_unsafe)
    base_model, base_revision = _infer_base(export_source or latest_checkpoint)
    # Conventional merged/<run_name> is relative to the logical adapters/
    # parent, which resolve() would lose if the run dir is a symlink.
    logical_run_dir = Path(run_dir).expanduser()
    merged_model = find_merged_model_dir(logical_run_dir, merged_dir)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(resolved_run_dir),
        "run_name": resolved_run_dir.name,
        "allow_unsafe_serialization": allow_unsafe,
        "checkpoints": {
            "valid_count": len(checkpoints),
            "steps": [_checkpoint_step(path) for path in checkpoints],
            "latest_step": latest_step,
            "latest_path": _as_str(latest_checkpoint),
        },
        "final_adapter": {
            "present": final_adapter_present,
            "path": str(resolved_run_dir) if final_adapter_present else None,
        },
        "merged_model": {
            "present": merged_model is not None,
            "path": _as_str(merged_model),
        },
        "base_model": base_model,
        "base_revision": base_revision,
        "resume": {
            "ready": latest_checkpoint is not None,
            "checkpoint_path": _as_str(latest_checkpoint),
        },
        "export": {
            "ready": export_source is not None and _adapter_config_usable(export_source),
            "source_path": export_source,
            "source_kind": export_kind,
        },
    }


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _or_dash(value: Any) -> str:
    return "-" if value is None else str(value)


def format_run_status_table(report: dict[str, Any]) -> str:
    """Render a report from `build_run_status` as an aligned `key: value` block."""
    checkpoints = report["checkpoints"]
    steps = checkpoints["steps"]
    rows: list[tuple[str, str]] = [
        ("schema_version", str(report["schema_version"])),
        ("run_name", report["run_name"]),
        ("run_dir", report["run_dir"]),
        ("allow_unsafe_serialization", _yes_no(report["allow_unsafe_serialization"])),
        ("valid_checkpoints", str(checkpoints["valid_count"])),
        ("checkpoint_steps", ", ".join(str(step) for step in steps) if steps else "-"),
        ("latest_checkpoint_step", _or_dash(checkpoints["latest_step"])),
        ("latest_checkpoint_path", _or_dash(checkpoints["latest_path"])),
        ("final_adapter", _yes_no(report["final_adapter"]["present"])),
        ("final_adapter_path", _or_dash(report["final_adapter"]["path"])),
        ("merged_model", _yes_no(report["merged_model"]["present"])),
        ("merged_model_path", _or_dash(report["merged_model"]["path"])),
        ("base_model", _or_dash(report["base_model"])),
        ("base_revision", _or_dash(report["base_revision"])),
        ("resume_ready", _yes_no(report["resume"]["ready"])),
        ("resume_checkpoint_path", _or_dash(report["resume"]["checkpoint_path"])),
        ("export_ready", _yes_no(report["export"]["ready"])),
        ("export_source_kind", _or_dash(report["export"]["source_kind"])),
        ("export_source_path", _or_dash(report["export"]["source_path"])),
    ]
    width = max(len(label) for label, _ in rows) + 1
    return "\n".join(f"{label + ':':<{width}} {value}" for label, value in rows)
