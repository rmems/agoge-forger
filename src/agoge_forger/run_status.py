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
import unicodedata
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


def _merged_config_is_object(candidate: Path) -> bool:
    """True when config.json exists and parses as a JSON object."""
    config_path = candidate / "config.json"
    if not config_path.is_file():
        return False
    try:
        payload = json.loads(config_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict)


def _shard_filenames(weight_map: dict[str, Any]) -> set[str] | None:
    """Unique shard filenames, or None if any mapping is not a string."""
    shards: set[str] = set()
    for name in weight_map.values():
        if not isinstance(name, str):
            return None
        shards.add(name)
    return shards or None


def _has_complete_sharded_weights(candidate: Path) -> bool:
    """True when every shard named in model.safetensors.index.json exists."""
    index_path = candidate / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards = _shard_filenames(weight_map)
    if shards is None:
        return False
    return all((candidate / name).is_file() for name in shards)


def _has_complete_merged_weights(candidate: Path) -> bool:
    """True for unsharded model.safetensors or a complete shard index set."""
    if (candidate / "model.safetensors").is_file():
        return True
    return _has_complete_sharded_weights(candidate)


def is_merged_model_dir(path: PathLike) -> bool:
    """Return True when `path` looks like an exported merged model directory.

    A merged model is a full `save_pretrained` tree: a parseable object
    `config.json` plus either `model.safetensors` or every shard named in
    `model.safetensors.index.json`, all at the directory root. Nested
    adapter weights or a leftover `adapter_model.safetensors` do not count.
    """
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    return _merged_config_is_object(candidate) and _has_complete_merged_weights(candidate)


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
    if not is_merged_model_dir(conventional):
        return None
    # Discovery used the logical path; emit an absolute one so
    # merged_model.path does not depend on the caller's cwd.
    return conventional.resolve()


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
    # infer_base_model_from_adapter returns the raw truthy field. A list, dict,
    # bool, or int would leak into the JSON report; only a real string is a
    # usable model id. Sibling infer_base_revision_from_adapter already str().
    if not isinstance(base_model, str):
        base_model = None

    try:
        base_revision = infer_base_revision_from_adapter(adapter_path)
    except _ADAPTER_CONFIG_ERRORS:
        base_revision = None

    return base_model, base_revision


def _adapter_config_usable(adapter_path: PathLike | None) -> bool:
    """True when adapter_config.json is a JSON object export can parse.

    A valid object is not enough: the default `export-final-model --run-dir`
    path calls `infer_base_model_from_adapter` and raises unless
    `base_model_name_or_path` is a non-empty string. `run-status` has no
    `--base-model` override, so that field must be present for ready.
    """
    if adapter_path is None:
        return False
    config_path = Path(adapter_path) / "adapter_config.json"
    try:
        payload = json.loads(config_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    base = payload.get("base_model_name_or_path")
    return isinstance(base, str) and bool(base)


def _trainer_state_usable(checkpoint: PathLike | None) -> bool:
    """True when trainer_state.json parses as a JSON object.

    `list_valid_checkpoints` only requires the file to exist. Trainer.train
    deserializes it, so a truncated or non-object state is not resume-ready.
    """
    if checkpoint is None:
        return False
    state_path = Path(checkpoint) / "trainer_state.json"
    try:
        payload = json.loads(state_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict)


_SAFETENSORS_HEADER_MAX = 100 * 1024 * 1024


def _safetensors_header_len_ok(header_len: int) -> bool:
    """True when header length is aligned and within the read budget."""
    if header_len < 8:
        return False
    if header_len % 8 != 0:
        return False
    return header_len <= _SAFETENSORS_HEADER_MAX


def _safetensors_header_usable(path: Path) -> bool:
    """True when `path` has a parseable, 8-byte-aligned safetensors JSON header."""
    try:
        with path.open("rb") as handle:
            size_bytes = handle.read(8)
            if len(size_bytes) != 8:
                return False
            header_len = int.from_bytes(size_bytes, "little")
            if not _safetensors_header_len_ok(header_len):
                return False
            header = handle.read(header_len)
            if len(header) != header_len:
                return False
            payload = json.loads(header)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict)


def _adapter_weights_usable(adapter_path: PathLike | None, *, allow_unsafe: bool = False) -> bool:
    """True when adapter weight files pass a lightweight validity check.

    `is_adapter_artifact` only checks the filename. export-final-model then
    fails when PEFT/safetensors opens an empty or truncated file.
    """
    if adapter_path is None:
        return False
    adapter_dir = Path(adapter_path)
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.is_file() and _safetensors_header_usable(safetensors_path):
        return True
    if allow_unsafe:
        legacy = adapter_dir / "adapter_model.bin"
        try:
            return legacy.is_file() and legacy.stat().st_size > 0
        except OSError:
            return False
    return False


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
            "ready": latest_checkpoint is not None and _trainer_state_usable(latest_checkpoint),
            "checkpoint_path": _as_str(latest_checkpoint),
        },
        "export": {
            "ready": (
                export_source is not None
                and _adapter_config_usable(export_source)
                and _adapter_weights_usable(export_source, allow_unsafe=allow_unsafe)
            ),
            "source_path": export_source,
            "source_kind": export_kind,
        },
    }


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _escape_controls(text: str) -> str:
    """Escape Unicode Cc controls so table cells cannot drive the terminal.

    Adapter metadata and paths can carry ANSI or other control bytes into
    `format_run_status_table`. Render those as backslash-uXXXX escapes
    instead of emitting them raw (CWE-150 / CodeRabbit on #96).
    """
    return "".join(f"\\u{ord(ch):04x}" if unicodedata.category(ch) == "Cc" else ch for ch in text)


def _or_dash(value: Any) -> str:
    return "-" if value is None else _escape_controls(str(value))


def format_run_status_table(report: dict[str, Any]) -> str:
    """Render a report from `build_run_status` as an aligned `key: value` block."""
    checkpoints = report["checkpoints"]
    steps = checkpoints["steps"]
    rows: list[tuple[str, str]] = [
        ("schema_version", str(report["schema_version"])),
        ("run_name", _escape_controls(str(report["run_name"]))),
        ("run_dir", _escape_controls(str(report["run_dir"]))),
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
