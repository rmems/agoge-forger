"""Agoge consumer contract for Prometheus-derived JSONL.

Loads ``messages`` and instruction/input/output rows through the current
dataset parser, writes a provenance sidecar, and rejects future-event
leakage. The path is CPU-only: no model download and no GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._atomic_file import publish_bytes_replace
from .datasets import normalize_row
from .path_safety import resolve_existing_path, resolve_output_directory

SIDECAR_SCHEMA = "agoge.consumer-sidecar.v1"
PARSER_NAME = "agoge_forger.datasets.normalize_row"


class ConsumerContractError(ValueError):
    """A consumer-contract input or runtime precondition failed."""


def _cuda_is_available() -> bool:
    # Call `torch.cuda.is_available` directly. `getattr(..., None)` made Qodana
    # treat the later call as "'None' object is not callable".
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, TypeError):
        return False


def refuse_gpu() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "-1"):
        raise ConsumerContractError("consumer-contract must run with no GPU (CUDA_VISIBLE_DEVICES)")
    if "torch" in sys.modules and _cuda_is_available():
        raise ConsumerContractError("consumer-contract imported torch with CUDA available")


def _parse_timestamp(value: str) -> datetime:
    iso = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    parsed = datetime.fromisoformat(iso)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _prometheus_timestamps(record: dict[str, Any]) -> list[str]:
    meta = record.get("_prometheus")
    if not isinstance(meta, dict):
        return []
    raw = meta.get("event_timestamps") or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _event_payload_timestamps(record: dict[str, Any]) -> list[str]:
    events = record.get("events")
    if not isinstance(events, list):
        return []
    return [
        event["timestamp"]
        for event in events
        if isinstance(event, dict) and isinstance(event.get("timestamp"), str)
    ]


def _ordering_errors(stamps: list[str]) -> list[str]:
    errors: list[str] = []
    last: datetime | None = None
    for stamp in stamps:
        try:
            current = _parse_timestamp(stamp)
        except (ValueError, TypeError, OverflowError):
            errors.append(f"unparseable event timestamp {stamp}")
            continue
        if last is not None and current < last:
            errors.append(f"future-event leakage / events not ordered ({stamp})")
        last = current
    return errors


def future_event_errors(record: dict[str, Any]) -> list[str]:
    stamps = _prometheus_timestamps(record) + _event_payload_timestamps(record)
    return _ordering_errors(stamps)


def _format_name(record: dict[str, Any]) -> str:
    if "messages" in record:
        return "messages"
    if "instruction" in record:
        return "instruction"
    if "text" in record:
        return "text"
    return "unknown"


def _load_jsonl_object(source_name: str, line_number: int, line: str) -> dict[str, Any] | str:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        return f"{source_name}:{line_number} Invalid JSON: {exc}"
    if not isinstance(record, dict):
        return f"{source_name}:{line_number} JSONL row must be an object"
    return record


def _consume_record(
    source_name: str, line_number: int, record: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    prefix = f"{source_name}:{line_number}"
    errors = [f"{prefix} {message}" for message in future_event_errors(record)]
    payload = {key: value for key, value in record.items() if key != "_prometheus"}
    try:
        normalized = normalize_row(payload, None, index=line_number)
    except ValueError as exc:
        return None, [*errors, f"{prefix} {exc}"]
    text = normalized.get("text") if isinstance(normalized, dict) else None
    if not isinstance(text, str):
        return None, [*errors, f"{prefix} parser did not return a text row"]
    row = {
        "line": line_number,
        "format": _format_name(payload),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return row, errors


def consume_file(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    source_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {
            "schema_version": SIDECAR_SCHEMA,
            "source_path": str(path),
            "source_sha256": source_sha256,
            "parser": PARSER_NAME,
            "row_count": 0,
            "rows": [],
            "errors": [f"{path.name} is not valid UTF-8: {exc}"],
            "ok": False,
        }
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        loaded = _load_jsonl_object(path.name, line_number, line)
        if isinstance(loaded, str):
            errors.append(loaded)
            continue
        row, row_errors = _consume_record(path.name, line_number, loaded)
        errors.extend(row_errors)
        if row is not None:
            rows.append(row)
    return {
        "schema_version": SIDECAR_SCHEMA,
        "source_path": str(path),
        "source_sha256": source_sha256,
        "parser": PARSER_NAME,
        "row_count": len(rows),
        "rows": rows,
        "errors": errors,
        "ok": not errors,
    }


def write_sidecar(sidecar: dict[str, Any], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.sidecar.json"
    payload = json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n"
    publish_bytes_replace(out_path, payload.encode("utf-8"))
    return out_path


def _sidecar_stem(path: Path, used: set[str]) -> str:
    stem = path.stem
    if stem not in used:
        used.add(stem)
        return stem
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    unique = f"{stem}-{digest}"
    used.add(unique)
    return unique


def _consume_input(raw_path: str, out_dir: Path, used_stems: set[str]) -> dict[str, Any]:
    path = resolve_existing_path(raw_path, must_be_file=True)
    sidecar = consume_file(path)
    write_sidecar(sidecar, out_dir, _sidecar_stem(path, used_stems))
    return sidecar


def run_consumer_contract(inputs: list[str], out_dir: str) -> list[dict[str, Any]]:
    refuse_gpu()
    if not inputs:
        raise ConsumerContractError("consumer-contract requires at least one JSONL input")
    safe_out = resolve_output_directory(out_dir)
    used_stems: set[str] = set()
    sidecars = [_consume_input(raw_path, safe_out, used_stems) for raw_path in inputs]
    failures = [error for sidecar in sidecars for error in sidecar["errors"]]
    if failures:
        raise ConsumerContractError("; ".join(failures))
    return sidecars
