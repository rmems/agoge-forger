"""Synthetic Hermes builders for offline normalize tests (RM-1348).

Fixtures are deliberately invented. They are not personal conversations,
live Hermes traces, or recorded GitHub pages.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "hermes"
ADMISSION_FIXTURE = FIXTURES / "local_model_admission_v1.json"

BASE_OID = "aaa111aaa111aaa111aaa111aaa111aaa111aaa1"
HEAD_OID = "bbb222bbb222bbb222bbb222bbb222bbb222bbb2"
MODEL_DIGEST = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
TERMS_SHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
PRODUCER_REVISION = "c0ffee1c0ffee1c0ffee1c0ffee1c0ffee1c0ffe"
VERIFIER_ARTIFACT = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
SYNTHETIC_GITHUB_TOKEN = "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

FIXTURE_VERIFIER_OUTCOMES: dict[str, str | None] = {
    "completed_true_verifier_failed.jsonl": "fail",
    "duplicate_keys.jsonl": "pass",
    "duplicate_records.jsonl": "pass",
    "hidden_reasoning.jsonl": "pass",
    "interrupted.jsonl": "interrupted",
    "malformed.jsonl": "pass",
    "partial.jsonl": "interrupted",
    "query_secret.jsonl": "pass",
    "secret_leak.jsonl": "pass",
    "successful.jsonl": "pass",
    "truncated.jsonl": "pass",
    "unverified.jsonl": None,
}


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def seal_admission(payload: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != "evidence_digest"}
    sealed = dict(body)
    sealed["evidence_digest"] = sha256_bytes(canonical_dumps(body).encode("utf-8"))
    return sealed


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def default_admission(**overrides: Any) -> dict[str, Any]:
    payload = json.loads(ADMISSION_FIXTURE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("local_model_admission_v1 fixture must be an object")
    return seal_admission(_deep_merge(payload, overrides))


def default_verifier(*, outcome: str | None = "pass") -> dict[str, Any]:
    artifacts = [VERIFIER_ARTIFACT] if outcome else []
    return {
        "artifact_sha256": artifacts,
        "identity": "pytest-local",
        "outcome": outcome,
        "version": "8.4.1",
    }


def default_manifest(
    *, admission_report_sha256: str, **overrides: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "hermes_run_manifest_v1",
        "producer": {
            "name": "hermes-agent",
            "version": "0.4.0",
            "revision": PRODUCER_REVISION,
        },
        "model": {
            "name": "qwen2.5-coder",
            "tag": "7b-instruct",
            "ollama_digest": MODEL_DIGEST,
            "quantization": "q4_K_M",
            "upstream_revision": "synthetic-upstream-1",
        },
        "admission_report_sha256": admission_report_sha256,
        "ollama": {
            "runtime": "ollama",
            "version": "0.11.4",
            "endpoint": "http://127.0.0.1:11434",
            "config": {"num_ctx": 8192, "keep_alive": "5m"},
        },
        "repository": {
            "owner": "rmems",
            "name": "synthetic-loop",
            "url": "https://github.com/rmems/synthetic-loop",
            "base_oid": BASE_OID,
            "head_oid": HEAD_OID,
        },
        "workspace": {
            "id": "ws-isolated-001",
            "kind": "isolated",
        },
        "reasoning_retention": {
            "policy": "strip_hidden",
            "strip_tags": ["think"],
        },
        "verifier": default_verifier(),
    }
    return _deep_merge(payload, overrides)


def hermes_record(
    *,
    run_id: str,
    completed: bool = True,
    partial: bool = False,
    content: str | dict[str, Any] = "I'll inspect loop.py and apply a minimal fix.",
    reasoning: str | None = None,
    extra_message: dict[str, Any] | None = None,
    trace_verifier: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    assistant: dict[str, Any] = {
        "role": "assistant",
        "timestamp": "2026-09-21T12:00:10Z",
        "content": content,
    }
    if reasoning is not None:
        assistant["reasoning"] = reasoning
    messages: list[Any] = [
        {
            "role": "user",
            "timestamp": "2026-09-21T12:00:00Z",
            "content": "Fix the failing empty-list test in loop.py.",
        },
        assistant,
        {
            "role": "tool",
            "name": "apply_patch",
            "timestamp": "2026-09-21T12:00:20Z",
            "content": "--- a/loop.py\n+++ b/loop.py\n@@ -1 +1 @@\n-return None\n+return []\n",
        },
    ]
    if extra_message is not None:
        messages.append(extra_message)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "session_id": f"sess-{run_id}",
        "task_id": f"task-{run_id}",
        "raw_trace_id": f"trace-{run_id}",
        "completed": completed,
        "partial": partial,
        "started_at": "2026-09-21T12:00:00Z",
        "ended_at": "2026-09-21T12:01:00Z",
        "issue_statement": "Empty lists currently return None instead of [].",
        "messages": messages,
        "repository": {
            "owner": "rmems",
            "name": "synthetic-loop",
            "url": "https://github.com/rmems/synthetic-loop",
            "base_oid": BASE_OID,
            "head_oid": HEAD_OID,
        },
        "workspace": {"id": "ws-isolated-001", "kind": "isolated"},
    }
    if trace_verifier is not None:
        payload["verifier"] = trace_verifier
    return _deep_merge(payload, overrides)


def write_json(path: Path, payload: Any) -> bytes:
    raw = (canonical_dumps(payload) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> bytes:
    raw = "".join(canonical_dumps(record) + "\n" for record in records).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def write_scenario(
    tmp_path: Path,
    records: list[dict[str, Any]],
    *,
    admission: dict[str, Any] | None = None,
    manifest_overrides: dict[str, Any] | None = None,
    input_name: str = "input.jsonl",
) -> dict[str, Path]:
    admission_payload = seal_admission(
        default_admission() if admission is None else admission
    )
    admission_path = tmp_path / "model_admission.json"
    admission_bytes = write_json(admission_path, admission_payload)
    manifest = default_manifest(
        admission_report_sha256=sha256_bytes(admission_bytes),
        **(manifest_overrides or {}),
    )
    manifest_path = tmp_path / "run_manifest.json"
    write_json(manifest_path, manifest)
    input_path = tmp_path / input_name
    write_jsonl(input_path, records)
    return {
        "admission": admission_path,
        "input": input_path,
        "manifest": manifest_path,
        "output": tmp_path / "canonical.jsonl",
        "report": tmp_path / "decision-report.json",
    }


def write_fixture_scenario(tmp_path: Path, name: str) -> dict[str, Path]:
    outcome = FIXTURE_VERIFIER_OUTCOMES[name]
    paths = write_scenario(
        tmp_path,
        [],
        manifest_overrides={"verifier": default_verifier(outcome=outcome)},
    )
    paths["input"].write_bytes((FIXTURES / name).read_bytes())
    return paths


def cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--input",
        str(paths["input"]),
        "--run-manifest",
        str(paths["manifest"]),
        "--model-admission",
        str(paths["admission"]),
        "--output",
        str(paths["output"]),
        "--report",
        str(paths["report"]),
    ]


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
