"""Trajectory v1.1 schema presence while v0 / v1.0 stay unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
V0_SCHEMA = ROOT / "schemas" / "pr_trajectory.schema.json"
V1_SCHEMA = ROOT / "schemas" / "trajectory_v1.schema.json"
V1_1_SCHEMA = ROOT / "schemas" / "trajectory_v1_1.schema.json"
MANIFEST_SCHEMA = ROOT / "schemas" / "hermes_run_manifest.schema.json"
RAW_SCHEMA = ROOT / "schemas" / "hermes_raw_trace.schema.json"
V1_SOFTWARE = ROOT / "tests" / "fixtures" / "v1" / "software_valid.jsonl"
V0_EXAMPLE = ROOT / "datasets" / "examples" / "trajectory-v0-example.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v1_0_schema_version_enum_is_unchanged():
    schema = _load(V1_SCHEMA)
    assert schema["properties"]["schema_version"]["enum"] == ["1", "1.0", "v1"]


def test_v0_example_still_matches_v0_schema():
    schema = _load(V0_SCHEMA)
    jsonschema.Draft7Validator(schema).validate(_load(V0_EXAMPLE))


def test_v1_software_fixture_still_matches_v1_schema():
    schema = _load(V1_SCHEMA)
    record = json.loads(V1_SOFTWARE.read_text(encoding="utf-8").splitlines()[0])
    jsonschema.Draft7Validator(
        schema, format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER
    ).validate(record)


def test_v1_1_schema_declares_new_version_and_bindings():
    schema = _load(V1_1_SCHEMA)
    assert schema["properties"]["schema_version"]["enum"] == ["1.1", "v1.1"]
    required = set(schema["required"])
    assert "execution" in required
    assert "execution_provenance" in required
    assert "provenance_binding" not in required
    assert schema.get("additionalProperties") is False
    execution = schema["properties"]["execution"]["required"]
    assert "producer_completed" in execution
    assert "producer_partial" in execution
    binding = schema["properties"]["execution_provenance"]["required"]
    for field in (
        "run_id",
        "session_id",
        "task_id",
        "raw_trace_id",
        "producer",
        "model",
        "ollama",
        "workspace",
        "reasoning_retention",
        "verifier",
        "repository",
        "input_sha256",
        "admission_report_sha256",
        "raw_trace_sha256",
        "run_manifest_sha256",
    ):
        assert field in binding
    digest = schema["properties"]["execution_provenance"]["properties"]["model"][
        "properties"
    ]["ollama_digest"]
    digest_pattern = (
        digest.get("pattern") or schema["definitions"]["modelDigest"]["pattern"]
    )
    assert "sha256:" in digest_pattern
    policy = schema["properties"]["execution_provenance"]["properties"][
        "reasoning_retention"
    ]["properties"]["policy"]
    assert policy.get("const") == "strip_hidden"


def test_hermes_run_manifest_schema_binds_identities():
    schema = _load(MANIFEST_SCHEMA)
    assert schema["properties"]["schema_version"]["const"] == "hermes_run_manifest_v1"
    assert schema.get("additionalProperties") is False
    required = set(schema["required"])
    for field in (
        "producer",
        "model",
        "admission_report_sha256",
        "ollama",
        "repository",
        "workspace",
        "reasoning_retention",
        "verifier",
    ):
        assert field in required
    verifier = schema["properties"]["verifier"]["required"]
    for field in ("identity", "version", "outcome", "artifact_sha256"):
        assert field in verifier


def test_raw_hermes_schema_requires_identities_and_booleans():
    schema = _load(RAW_SCHEMA)
    required = set(schema["required"])
    for field in (
        "run_id",
        "session_id",
        "task_id",
        "raw_trace_id",
        "completed",
        "partial",
        "messages",
    ):
        assert field in required
    assert schema["properties"]["completed"]["type"] == "boolean"
    assert schema["properties"]["partial"]["type"] == "boolean"
    assert schema["properties"]["messages"]["items"]["type"] == "object"
    assert (
        schema["properties"]["messages"]["items"]["properties"]["content"]["type"]
        == "string"
    )
    assert schema["properties"]["messages"]["items"]["additionalProperties"] is False
