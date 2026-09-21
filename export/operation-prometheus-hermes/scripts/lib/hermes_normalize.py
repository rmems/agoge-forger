"""Normalize local Hermes traces into deterministic trajectory v1.1 records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from validate_jsonl import HOME_PATH_RE, load_schema

from .hermes_sanitize import (
    SECRET_QUERY_KEYS,
    UnsafeUrlError,
    contains_header_secret,
    hidden_markup_remains,
    is_hidden_key,
    sanitize_query_secrets,
    strip_hidden_reasoning,
)
from .secrets import find_secrets
from .source_inventory_common import sha256_json

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised at CLI startup
    jsonschema = None

NORMALIZER_VERSION = "hermes-normalize/0.1.0"
SCHEMA_VERSION = "1.1"
REPORT_SCHEMA_VERSION = "hermes_normalize_report_v1"
COLLECTION_POLICY = "isolated-local-hermes"
PROVIDER_ID = "hermes-agent"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
V1_1_SCHEMA_PATH = REPO_ROOT / "schemas" / "trajectory_v1_1.schema.json"
MANIFEST_SCHEMA_PATH = REPO_ROOT / "schemas" / "hermes_run_manifest.schema.json"
RAW_SCHEMA_PATH = REPO_ROOT / "schemas" / "hermes_raw_trace.schema.json"
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
ALLOWED_OLLAMA_CONFIG = frozenset({"num_ctx", "keep_alive"})
_REPO_FIELDS = ("owner", "name", "url", "base_oid", "head_oid")
_WORKSPACE_FIELDS = ("id", "kind")
_VERIFIER_FIELDS = ("identity", "version", "outcome", "artifact_sha256")

_SUCCESS_OUTCOMES = frozenset(
    {"pass", "passed", "success", "successful", "verified", "ok"}
)
_FAILED_OUTCOMES = frozenset({"fail", "failed", "error", "falsified"})
_INTERRUPTED_OUTCOMES = frozenset({"interrupted", "timeout", "killed", "incomplete"})
_TERMINAL_EVIDENCE = frozenset({"successful", "failed", "interrupted"})
_ACTOR_TYPES = {
    "user": "human",
    "human": "human",
    "assistant": "agent",
    "agent": "agent",
    "tool": "application",
    "application": "application",
    "system": "application",
}


class LocalFileBoundary:
    """Read-only local-file source. Tests may inject frozen bytes for replay."""

    def __init__(self, opener: Callable[[Path], bytes] | None = None) -> None:
        self._opener = opener

    def read_bytes(self, path: Path | str) -> bytes:
        raw = str(path)
        if _is_remote(raw):
            raise ValueError(f"refusing non-local file path: {raw}")
        resolved = Path(raw).expanduser()
        if self._opener is not None:
            return self._opener(resolved)
        return resolved.read_bytes()


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_files(
    *,
    input_path: Path,
    run_manifest_path: Path,
    model_admission_path: Path,
    output_path: Path,
    report_path: Path,
    source: LocalFileBoundary | None = None,
) -> dict[str, Any]:
    """Normalize one Hermes JSONL file. Returns the decision report."""
    _refuse_collisions(
        [input_path, run_manifest_path, model_admission_path],
        output_path,
        report_path,
    )
    boundary = source or LocalFileBoundary()
    input_bytes = boundary.read_bytes(input_path)
    manifest_bytes = boundary.read_bytes(run_manifest_path)
    admission_bytes = boundary.read_bytes(model_admission_path)
    report = _normalize_bytes(
        input_bytes=input_bytes,
        manifest_bytes=manifest_bytes,
        admission_bytes=admission_bytes,
        input_path=str(input_path),
    )
    admitted = [row["output_line"] for row in report.pop("_admitted_lines")]
    admitted_text = ("\n".join(admitted) + "\n") if admitted else ""
    _atomic_write_bytes(output_path, admitted_text.encode("utf-8"))
    _atomic_write_bytes(report_path, (canonical_dumps(report) + "\n").encode("utf-8"))
    return report


def _is_remote(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered.startswith(("http://", "https://", "ftp://")):
        return True
    parsed = urlsplit(raw)
    return bool(parsed.scheme) and parsed.scheme not in {"", "file"}


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _refuse_collisions(
    inputs: list[Path], output_path: Path, report_path: Path
) -> None:
    if _same_file(output_path, report_path):
        raise RuntimeError("output and report paths must be different")
    for source in inputs:
        if _same_file(source, output_path):
            raise RuntimeError(f"refusing to overwrite source file: {source}")
        if _same_file(source, report_path):
            raise RuntimeError(
                f"refusing to overwrite source file with the report: {source}"
            )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _normalize_bytes(
    *,
    input_bytes: bytes,
    manifest_bytes: bytes,
    admission_bytes: bytes,
    input_path: str,
) -> dict[str, Any]:
    input_digest = sha256_bytes(input_bytes)
    manifest_digest = sha256_bytes(manifest_bytes)
    admission_digest = sha256_bytes(admission_bytes)
    file_reasons: list[str] = []
    manifest, manifest_reason = _parse_json_object(manifest_bytes)
    admission, admission_reason = _parse_json_object(admission_bytes)
    if manifest is None:
        file_reasons.append(manifest_reason or "invalid_json")
    else:
        file_reasons.extend(_manifest_schema_errors(manifest))
        file_reasons.extend(_safety_reasons(manifest))
    if admission is None:
        file_reasons.append(admission_reason or "invalid_json")
    else:
        file_reasons.extend(_safety_reasons(admission))
    if manifest is not None:
        file_reasons.extend(_workspace_errors(manifest))
    if manifest is not None and admission is not None:
        file_reasons.extend(_binding_errors(manifest, admission, admission_digest))

    decisions: list[dict[str, Any]] = []
    admitted_lines: list[str] = []
    seen_keys: dict[tuple[str, str, str, str], int] = {}
    validator = _v1_1_validator()
    raw_validator = _raw_validator()
    for line_no, raw_line in _iter_jsonl_lines(input_bytes):
        payload = raw_line.rstrip(b"\r\n")
        if not payload.strip():
            continue
        decision = _normalize_line(
            payload,
            line_no=line_no,
            input_path=input_path,
            input_digest=input_digest,
            manifest_digest=manifest_digest,
            admission_digest=admission_digest,
            manifest=manifest,
            file_reasons=file_reasons,
            seen_keys=seen_keys,
            validator=validator,
            raw_validator=raw_validator,
        )
        decisions.append(decision["report_row"])
        if decision["output_line"] is not None:
            admitted_lines.append(decision["output_line"])

    if not decisions:
        empty_reasons = [*file_reasons, "empty_input"]
        decisions.append(
            _decision(
                line_no=0,
                status="rejected",
                reasons=empty_reasons,
                source_sha256=input_digest,
                trajectory_id=None,
                output_line=None,
            )["report_row"]
        )

    report = {
        "accepted_count": sum(1 for row in decisions if row["status"] == "accepted"),
        "admission_report_sha256": admission_digest,
        "input_sha256": input_digest,
        "quarantined_count": sum(
            1 for row in decisions if row["status"] == "quarantined"
        ),
        "records": decisions,
        "rejected_count": sum(1 for row in decisions if row["status"] == "rejected"),
        "run_manifest_sha256": manifest_digest,
        "schema_version": REPORT_SCHEMA_VERSION,
        "_admitted_lines": [{"output_line": line} for line in admitted_lines],
    }
    return report


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _object_fields(value: Any, fields: tuple[str, ...]) -> bool:
    if not isinstance(value, dict):
        return False
    return all(_nonempty_str(value.get(field)) for field in fields)


_MODEL_BIND_FIELDS = (
    "name",
    "tag",
    "ollama_digest",
    "quantization",
    "upstream_revision",
)
_OPTIONAL_MODEL_FIELDS = frozenset({"quantization", "upstream_revision"})
_REQUIRED_INPUT_DIGESTS = ("admissions", "rights", "probe")


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _evidence_digest_ok(admission: dict[str, Any]) -> bool:
    declared = admission.get("evidence_digest")
    if not _is_sha256(declared):
        return False
    body = {key: value for key, value in admission.items() if key != "evidence_digest"}
    return declared == sha256_json(body)


def _fallback_disproved(admission: dict[str, Any]) -> bool:
    provider = admission.get("provider")
    config = provider.get("config") if isinstance(provider, dict) else None
    fallback = admission.get("fallback_evidence")
    if not isinstance(config, dict) or not isinstance(fallback, dict):
        return False
    return (
        admission.get("cloud_fallback_allowed") is False
        and config.get("no_cloud") is True
        and config.get("cloud_fallback_allowed") is False
        and fallback.get("no_cloud") is True
        and fallback.get("cloud_fallback_allowed") is False
        and fallback.get("unsanitized_keys") == []
        and fallback.get("remote_endpoints") == []
    )


def _workspace_errors(manifest: dict[str, Any]) -> list[str]:
    workspace = manifest.get("workspace")
    if not isinstance(workspace, dict):
        return ["workspace_not_isolated"]
    if not _nonempty_str(workspace.get("id")) or workspace.get("kind") != "isolated":
        return ["workspace_not_isolated"]
    return []


def _binding_errors(
    manifest: dict[str, Any], admission: dict[str, Any], admission_digest: str
) -> list[str]:
    reasons: list[str] = []
    expected = str(manifest.get("admission_report_sha256") or "")
    if expected != admission_digest:
        reasons.append("admission_digest_mismatch")
    if admission.get("schema_version") != "local_model_admission_v1":
        reasons.append("admission_schema")
    if admission.get("decision") != "accepted":
        reasons.append("admission_rejected")
    if admission.get("reasons") != []:
        reasons.append("admission_reasons")
    if not _evidence_digest_ok(admission):
        reasons.append("evidence_digest")
    model = admission.get("model")
    runtime = admission.get("runtime")
    rights = admission.get("rights")
    provider = admission.get("provider")
    probe = admission.get("probe")
    digests = admission.get("input_digests")
    if not _object_fields(model, ("name", "tag", "ollama_digest", "quantization")):
        reasons.append("admission_schema")
    if (
        isinstance(model, dict)
        and "upstream_revision" in model
        and model.get("upstream_revision") is not None
        and not _nonempty_str(model.get("upstream_revision"))
    ):
        reasons.append("admission_schema")
    if not _object_fields(runtime, ("name", "version", "endpoint")):
        reasons.append("admission_schema")
    if not _object_fields(rights, ("identifier", "terms_source", "terms_sha256")):
        reasons.append("admission_schema")
    if not isinstance(provider, dict) or provider.get("name") != "hermes-agent":
        reasons.append("provider_mismatch")
    if not _object_fields(probe, ("timestamp", "endpoint")):
        reasons.append("admission_schema")
    if not isinstance(digests, dict) or any(
        not _is_sha256(digests.get(name)) for name in _REQUIRED_INPUT_DIGESTS
    ):
        reasons.append("admission_schema")
    if not _fallback_disproved(admission):
        reasons.append("cloud_fallback")
    reasons.extend(_cross_bind_errors(manifest, admission))
    return reasons


def _cross_bind_errors(manifest: dict[str, Any], admission: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    manifest_model = (
        manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    )
    admission_model = (
        admission.get("model") if isinstance(admission.get("model"), dict) else {}
    )
    for field in _MODEL_BIND_FIELDS:
        left = manifest_model.get(field)
        right = admission_model.get(field)
        if field in _OPTIONAL_MODEL_FIELDS and not _present(left) and not _present(right):
            continue
        if left != right:
            reasons.append("model_mismatch")
            break
    ollama = manifest.get("ollama") if isinstance(manifest.get("ollama"), dict) else {}
    runtime = admission.get("runtime") if isinstance(admission.get("runtime"), dict) else {}
    if (
        runtime.get("name") != ollama.get("runtime")
        or runtime.get("version") != ollama.get("version")
        or runtime.get("endpoint") != ollama.get("endpoint")
    ):
        reasons.append("runtime_mismatch")
    probe = admission.get("probe") if isinstance(admission.get("probe"), dict) else {}
    if probe.get("endpoint") != runtime.get("endpoint") or probe.get(
        "endpoint"
    ) != ollama.get("endpoint"):
        reasons.append("probe_endpoint_mismatch")
    producer = manifest.get("producer") if isinstance(manifest.get("producer"), dict) else {}
    provider = admission.get("provider") if isinstance(admission.get("provider"), dict) else {}
    if provider.get("name") != producer.get("name"):
        reasons.append("provider_mismatch")
    return reasons


def _schema_reason(document: Any, schema_path: Path, code: str) -> list[str]:
    if jsonschema is None:
        return ["jsonschema_missing"]
    validator = jsonschema.Draft7Validator(
        load_schema(schema_path),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )
    errors = sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    if errors:
        return [code]
    return []


def _manifest_schema_errors(manifest: dict[str, Any]) -> list[str]:
    return _schema_reason(manifest, MANIFEST_SCHEMA_PATH, "invalid_manifest")


def _raw_record_errors(record: dict[str, Any], validator: Any) -> list[str]:
    if validator is None:
        return ["jsonschema_missing"]
    errors = sorted(validator.iter_errors(record), key=lambda item: list(item.path))
    if errors:
        return ["invalid_hermes_record"]
    return []


def _v1_1_validator() -> Any:
    if jsonschema is None:
        return None
    return jsonschema.Draft7Validator(
        load_schema(V1_1_SCHEMA_PATH),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )


def _raw_validator() -> Any:
    if jsonschema is None:
        return None
    return jsonschema.Draft7Validator(
        load_schema(RAW_SCHEMA_PATH),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )


def _normalize_line(
    payload: bytes,
    *,
    line_no: int,
    input_path: str,
    input_digest: str,
    manifest_digest: str,
    admission_digest: str,
    manifest: dict[str, Any] | None,
    file_reasons: list[str],
    seen_keys: dict[tuple[str, str, str, str], int],
    validator: Any,
    raw_validator: Any,
) -> dict[str, Any]:
    source_digest = sha256_bytes(payload)
    try:
        return _normalize_line_inner(
            payload,
            line_no=line_no,
            input_path=input_path,
            input_digest=input_digest,
            manifest_digest=manifest_digest,
            admission_digest=admission_digest,
            manifest=manifest,
            file_reasons=file_reasons,
            seen_keys=seen_keys,
            validator=validator,
            raw_validator=raw_validator,
            source_digest=source_digest,
        )
    except UnsafeUrlError as exc:
        reason = "secret_leakage" if "credential" in str(exc) else "invalid_record"
        return _decision(
            line_no=line_no,
            status="rejected",
            reasons=[*file_reasons, reason],
            source_sha256=source_digest,
            trajectory_id=None,
            output_line=None,
        )
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
        return _decision(
            line_no=line_no,
            status="rejected",
            reasons=[*file_reasons, "invalid_record"],
            source_sha256=source_digest,
            trajectory_id=None,
            output_line=None,
        )


def _normalize_line_inner(
    payload: bytes,
    *,
    line_no: int,
    input_path: str,
    input_digest: str,
    manifest_digest: str,
    admission_digest: str,
    manifest: dict[str, Any] | None,
    file_reasons: list[str],
    seen_keys: dict[tuple[str, str, str, str], int],
    validator: Any,
    raw_validator: Any,
    source_digest: str,
) -> dict[str, Any]:
    parsed, parse_reason = _parse_json_object(payload)
    reasons = list(file_reasons)
    if parsed is None:
        reasons.append(parse_reason or "invalid_json")
        return _decision(
            line_no=line_no,
            status="rejected",
            reasons=reasons,
            source_sha256=source_digest,
            trajectory_id=None,
            output_line=None,
        )
    reasons.extend(_raw_record_errors(parsed, raw_validator))
    identity: tuple[str, str, str, str] | None = None
    if "invalid_hermes_record" not in reasons:
        identity = _identity_tuple(parsed)
        first = seen_keys.get(identity)
        if first is None:
            seen_keys[identity] = line_no
        else:
            reasons.append("duplicate_record")
    reasons.extend(_tool_payload_reasons(parsed))
    reasons.extend(_safety_reasons(parsed, allow_hidden=True))
    sanitized = strip_hidden_reasoning(sanitize_query_secrets(parsed))
    reasons.extend(_safety_reasons(sanitized, allow_hidden=False))
    if manifest is not None:
        reasons.extend(_trace_conflicts(parsed, manifest))
        terminal, verifier_reasons = _independent_terminal(manifest)
        reasons.extend(verifier_reasons)
    else:
        terminal = None
        reasons.append("invalid_manifest")
    if terminal is None:
        status = "quarantined"
        reasons.append("unverified")
    elif reasons:
        status = "rejected"
    else:
        status = "accepted"
    if status != "accepted" or manifest is None:
        if status == "accepted":
            status = "rejected"
            reasons.append("invalid_manifest")
        if status == "quarantined" and any(
            code not in {"unverified"} for code in reasons
        ):
            status = "rejected"
        return _decision(
            line_no=line_no,
            status=status,
            reasons=_unique_reasons(reasons),
            source_sha256=source_digest,
            trajectory_id=_trajectory_id(parsed) if identity is not None else None,
            output_line=None,
        )
    record = _emit_record(
        sanitized,
        manifest=manifest,
        input_digest=input_digest,
        manifest_digest=manifest_digest,
        admission_digest=admission_digest,
        raw_trace_sha256=source_digest,
        terminal=terminal,
    )
    reasons.extend(_safety_reasons(record, allow_hidden=False))
    schema_errors = _schema_errors(record, validator)
    if schema_errors or reasons:
        extra = ["schema"] if schema_errors else []
        return _decision(
            line_no=line_no,
            status="rejected",
            reasons=_unique_reasons([*reasons, *extra]),
            source_sha256=source_digest,
            trajectory_id=record.get("trajectory_id"),
            output_line=None,
        )
    return _decision(
        line_no=line_no,
        status="accepted",
        reasons=[],
        source_sha256=source_digest,
        trajectory_id=record["trajectory_id"],
        output_line=canonical_dumps(record),
        extra={"input_path": input_path},
    )


def _decision(
    *,
    line_no: int,
    status: str,
    reasons: list[str],
    source_sha256: str,
    trajectory_id: str | None,
    output_line: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del extra
    row = {
        "line": line_no,
        "reason_codes": _unique_reasons(reasons),
        "source_sha256": source_sha256,
        "status": status,
        "trajectory_id": trajectory_id,
    }
    return {"output_line": output_line, "report_row": row}


def _unique_reasons(reasons: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for reason in reasons:
        if not reason or reason in seen:
            continue
        seen.add(reason)
        out.append(reason)
    return out


def _identity_tuple(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["run_id"]),
        str(record["session_id"]),
        str(record["task_id"]),
        str(record["raw_trace_id"]),
    )


def _trajectory_id(record: dict[str, Any]) -> str:
    encoded = "/".join(f"{len(part)}:{part}" for part in _identity_tuple(record))
    return f"hermes:{encoded}"


def _mapping_conflicts(
    trace_obj: Any, manifest_obj: Any, fields: tuple[str, ...], reason: str
) -> list[str]:
    if not isinstance(trace_obj, dict):
        return []
    if not isinstance(manifest_obj, dict):
        return [reason]
    for field in fields:
        if field in trace_obj and trace_obj.get(field) != manifest_obj.get(field):
            return [reason]
    return []


def _trace_conflicts(record: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    reasons.extend(
        _mapping_conflicts(
            record.get("repository"),
            manifest.get("repository"),
            _REPO_FIELDS,
            "repository_conflict",
        )
    )
    reasons.extend(
        _mapping_conflicts(
            record.get("workspace"),
            manifest.get("workspace"),
            _WORKSPACE_FIELDS,
            "workspace_conflict",
        )
    )
    source = record.get("verifier")
    expected = (
        manifest.get("verifier") if isinstance(manifest.get("verifier"), dict) else {}
    )
    if isinstance(source, dict):
        for field in _VERIFIER_FIELDS:
            if field in source and source.get(field) != expected.get(field):
                reasons.append("verifier_conflict")
                break
    elif source is not None:
        reasons.append("verifier_conflict")
    return reasons


def _artifact_hashes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if _is_sha256(item)]


def _independent_terminal(manifest: dict[str, Any]) -> tuple[str | None, list[str]]:
    verifier = manifest.get("verifier")
    if not isinstance(verifier, dict):
        return None, ["verifier_evidence"]
    outcome = verifier.get("outcome")
    if outcome is None or outcome == "":
        return None, []
    terminal = _terminal_disposition(str(outcome).strip().lower())
    if terminal not in _TERMINAL_EVIDENCE:
        return None, []
    producer = ""
    producer_obj = manifest.get("producer")
    if isinstance(producer_obj, dict):
        producer = str(producer_obj.get("name") or "")
    identity = verifier.get("identity")
    version = verifier.get("version")
    hashes = _artifact_hashes(verifier.get("artifact_sha256"))
    if (
        not _nonempty_str(identity)
        or not _nonempty_str(version)
        or not _external_verifier_identity(str(identity), producer)
        or len(hashes) < 1
    ):
        return terminal, ["verifier_evidence"]
    return terminal, []


def _external_verifier_identity(identity: str, producer: str) -> bool:
    folded = identity.casefold()
    blocked = {producer.casefold(), PROVIDER_ID.casefold(), "hermes-agent"}
    return folded not in blocked


def _terminal_disposition(outcome: str | None) -> str | None:
    if outcome is None:
        return None
    if outcome in _SUCCESS_OUTCOMES:
        return "successful"
    if outcome in _FAILED_OUTCOMES:
        return "failed"
    if outcome in _INTERRUPTED_OUTCOMES:
        return "interrupted"
    return None


_CREDENTIAL_KEY_NAMES = {name.replace("-", "_") for name in SECRET_QUERY_KEYS} | {
    "x_api_key",
    "bearer",
}


def _credential_key(key: Any) -> bool:
    return str(key).casefold().replace("-", "_") in _CREDENTIAL_KEY_NAMES


def _json_credential_string(value: str) -> bool:
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    return _contains_credential_key(parsed)


def _contains_secret(value: Any) -> bool:
    if isinstance(value, str):
        return (
            bool(find_secrets(value))
            or contains_header_secret(value)
            or _json_credential_string(value)
        )
    if isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_secret(key) or _contains_secret(item)
            for key, item in value.items()
        )
    return False


def _contains_credential_key(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_credential_key(item) for item in value)
    if isinstance(value, dict):
        for key, item in value.items():
            if _credential_key(key) and item not in (
                None,
                "",
                False,
            ):
                return True
            if _contains_credential_key(item):
                return True
    return False


def _contains_home_path(value: Any) -> bool:
    if isinstance(value, str):
        return HOME_PATH_RE.search(value) is not None
    if isinstance(value, list):
        return any(_contains_home_path(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_home_path(key) or _contains_home_path(item)
            for key, item in value.items()
        )
    return False


def _contains_hidden_reasoning(value: Any) -> bool:
    if isinstance(value, str):
        return hidden_markup_remains(value)
    if isinstance(value, list):
        return any(_contains_hidden_reasoning(item) for item in value)
    if isinstance(value, dict):
        return any(
            is_hidden_key(key) or _contains_hidden_reasoning(item)
            for key, item in value.items()
        )
    return False


def _tool_payload_reasons(record: dict[str, Any]) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        text = content.lstrip()
        if text.startswith("{") or text.startswith("["):
            try:
                json.loads(text)
            except json.JSONDecodeError:
                return ["invalid_tool_payload"]
        elif text.startswith("<") and not _xml_well_formed(text):
            return ["invalid_tool_payload"]
    return []


def _xml_well_formed(text: str) -> bool:
    try:
        ET.fromstring(text)
    except ET.ParseError:
        return False
    return True


def _safety_reasons(value: Any, *, allow_hidden: bool = False) -> list[str]:
    reasons: list[str] = []
    if (
        _contains_secret(value)
        or _contains_credential_key(value)
        or _contains_home_path(value)
    ):
        reasons.append("secret_leakage")
    if not allow_hidden and _contains_hidden_reasoning(value):
        reasons.append("hidden_reasoning")
    return reasons


def _pick(source: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fields:
        if field in source:
            out[field] = source[field]
    return out


def _ollama_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest.get("ollama") if isinstance(manifest.get("ollama"), dict) else {}
    out: dict[str, Any] = {
        "endpoint": sanitize_query_secrets(str(source.get("endpoint") or "")),
        "runtime": source.get("runtime"),
        "version": source.get("version"),
    }
    config = source.get("config")
    if isinstance(config, dict):
        cleaned = {
            key: value for key, value in config.items() if key in ALLOWED_OLLAMA_CONFIG
        }
        if cleaned:
            out["config"] = cleaned
    return out


def _verifier_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    source = (
        manifest.get("verifier") if isinstance(manifest.get("verifier"), dict) else {}
    )
    return {
        "artifact_sha256": _artifact_hashes(source.get("artifact_sha256")),
        "identity": source.get("identity"),
        "outcome": source.get("outcome"),
        "version": source.get("version"),
    }


def _repository_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    source = (
        manifest.get("repository")
        if isinstance(manifest.get("repository"), dict)
        else {}
    )
    out = _pick(source, _REPO_FIELDS)
    cleaned = {key: value for key, value in out.items() if value not in (None, "")}
    url = cleaned.get("url")
    if isinstance(url, str) and url:
        cleaned["url"] = sanitize_query_secrets(url)
    return cleaned


def _workspace_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    source = (
        manifest.get("workspace") if isinstance(manifest.get("workspace"), dict) else {}
    )
    return _pick(source, _WORKSPACE_FIELDS)


def _emit_record(
    record: dict[str, Any],
    *,
    manifest: dict[str, Any],
    input_digest: str,
    manifest_digest: str,
    admission_digest: str,
    raw_trace_sha256: str,
    terminal: str,
) -> dict[str, Any]:
    repository = _repository_from_manifest(manifest)
    workspace = _workspace_from_manifest(manifest)
    producer = _pick(
        manifest.get("producer") if isinstance(manifest.get("producer"), dict) else {},
        ("name", "version", "revision"),
    )
    model = _pick(
        manifest.get("model") if isinstance(manifest.get("model"), dict) else {},
        ("name", "tag", "ollama_digest", "quantization", "upstream_revision"),
    )
    retention = _pick(
        manifest.get("reasoning_retention")
        if isinstance(manifest.get("reasoning_retention"), dict)
        else {},
        ("policy", "strip_tags"),
    )
    events = _events(record, repository, terminal)
    payload = _software_payload(record, terminal)
    run_id, session_id, task_id, raw_trace_id = _identity_tuple(record)
    return {
        "artifacts": [],
        "collection_policy": COLLECTION_POLICY,
        "collector_version": NORMALIZER_VERSION,
        "events": events,
        "execution": {
            "producer_completed": record["completed"],
            "producer_partial": record["partial"],
        },
        "execution_provenance": {
            "admission_report_sha256": admission_digest,
            "input_sha256": input_digest,
            "model": model,
            "ollama": _ollama_from_manifest(manifest),
            "producer": producer,
            "raw_trace_id": raw_trace_id,
            "raw_trace_sha256": raw_trace_sha256,
            "repository": repository,
            "reasoning_retention": retention,
            "run_id": run_id,
            "run_manifest_sha256": manifest_digest,
            "session_id": session_id,
            "task_id": task_id,
            "verifier": _verifier_from_manifest(manifest),
            "workspace": workspace,
        },
        "license": "Apache-2.0",
        "provider_id": PROVIDER_ID,
        "repository": repository,
        "schema_version": SCHEMA_VERSION,
        "software_payload": payload,
        "source_id": raw_trace_id,
        "terminal_disposition": terminal,
        "trajectory_id": _trajectory_id(record),
        "trajectory_type": "software",
    }


def _events(
    record: dict[str, Any], repository: dict[str, Any], terminal: str
) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        timestamp = str(record.get("started_at") or record.get("ended_at") or "")
        return [_event("e1", timestamp, "agent", "run", terminal, repository, "")]
    events: list[dict[str, Any]] = []
    last_index = len(messages) - 1
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("non-object message")
        role = str(message.get("role") or "assistant")
        actor_type = _ACTOR_TYPES.get(role, "agent")
        event_type = "tool_call" if role == "tool" else "message"
        disposition = terminal if index == last_index else "neutral"
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("non-string message content")
        timestamp = str(message.get("timestamp") or record.get("started_at") or "")
        events.append(
            _event(
                f"e{index + 1}",
                timestamp,
                actor_type,
                event_type,
                disposition,
                repository,
                content,
            )
        )
    return events


def _event(
    event_id: str,
    timestamp: str,
    actor_type: str,
    event_type: str,
    disposition: str,
    repository: dict[str, Any],
    content: str,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "actor": {
            "id": PROVIDER_ID if actor_type != "human" else "user",
            "type": actor_type,
        },
        "code_state": {
            "base_oid": repository.get("base_oid"),
            "head_oid": repository.get("head_oid"),
        },
        "disposition": disposition,
        "event_id": event_id,
        "event_type": event_type,
        "timestamp": timestamp,
    }
    if content:
        event["content"] = content
    url = repository.get("url")
    if isinstance(url, str) and url:
        event["evidence_references"] = [url]
    return event


def _software_payload(record: dict[str, Any], terminal: str) -> dict[str, str]:
    issue = str(record.get("issue_statement") or "").strip()
    patch = _patch_from_messages(record)
    outcome = {
        "successful": "pass",
        "failed": "fail",
        "interrupted": "interrupted",
        "inconclusive": "unverified",
    }.get(terminal, terminal)
    payload: dict[str, str] = {"validation_outcome": outcome}
    if issue:
        payload["issue_statement"] = issue
    if patch:
        payload["implementation_patch"] = patch
    return payload


def _patch_from_messages(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" and message.get("name") == "apply_patch":
            return str(message.get("content") or "")
    return ""


def _schema_errors(record: dict[str, Any], validator: Any) -> list[str]:
    if validator is None:
        return ["jsonschema_missing"]
    errors = sorted(validator.iter_errors(record), key=lambda item: list(item.path))
    return [error.message for error in errors]


def _iter_jsonl_lines(data: bytes) -> list[tuple[int, bytes]]:
    rows: list[tuple[int, bytes]] = []
    start = 0
    line_no = 1
    while start < len(data):
        newline = data.find(b"\n", start)
        if newline == -1:
            rows.append((line_no, data[start:]))
            break
        rows.append((line_no, data[start : newline + 1]))
        start = newline + 1
        line_no += 1
    return rows


def _parse_json_object(payload: bytes) -> tuple[dict[str, Any] | None, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None, "invalid_json"
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError:
        return None, "invalid_json"
    except ValueError as exc:
        message = str(exc)
        if "duplicate JSON object key" in message:
            return None, "duplicate_key"
        return None, "invalid_json"
    if _contains_nonfinite(parsed):
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "invalid_json"
    return parsed, ""


def _unique_object(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    seen: set[Any] = set()
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON object key: {key}")
        seen.add(key)
        parsed[key] = value
    return parsed


def _reject_nonfinite(name: str) -> None:
    raise ValueError(f"non-finite JSON constant: {name}")


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float) and not math.isfinite(value):
        return True
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False
