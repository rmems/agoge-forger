"""Normalize verified Hermes traces into trajectory v1.1 (Linear RM-1348)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from hermes_fixtures import (
    ADMISSION_FIXTURE,
    BASE_OID,
    HEAD_OID,
    ROOT,
    SYNTHETIC_GITHUB_TOKEN,
    VERIFIER_ARTIFACT,
    cli_args,
    default_admission,
    default_verifier,
    hermes_record,
    sha256_bytes,
    seal_admission,
    write_json,
    write_scenario,
    write_fixture_scenario,
)
from lib.hermes_normalize import LocalFileBoundary, canonical_dumps, normalize_files
from normalize_hermes_trajectories import build_parser, main
from validate_jsonl import load_schema, validate_file

V1_1_SCHEMA = ROOT / "schemas" / "trajectory_v1_1.schema.json"
V0_SCHEMA = ROOT / "schemas" / "pr_trajectory.schema.json"
V1_SCHEMA = ROOT / "schemas" / "trajectory_v1.schema.json"


def _load_jsonl(path: Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [json.loads(line) for line in lines]


def _run(paths: dict[str, Path], extra: list[str] | None = None) -> int:
    argv = cli_args(paths)
    if extra:
        argv.extend(extra)
    return main(argv)


def _scenario_from_fixture(tmp_path: Path, name: str) -> dict[str, Path]:
    return write_fixture_scenario(tmp_path, name)


def _report(paths: dict[str, Path]) -> dict:
    return json.loads(paths["report"].read_text(encoding="utf-8"))


def test_cli_contract_exposes_exact_flags():
    actions = {action.dest for action in build_parser()._actions}
    assert {
        "input",
        "run_manifest",
        "model_admission",
        "output",
        "report",
    }.issubset(actions)
    usage = build_parser().format_help()
    assert "--input" in usage
    assert "--run-manifest" in usage
    assert "--model-admission" in usage
    assert "--output" in usage
    assert "--report" in usage


def test_successful_run_is_accepted_v1_1_and_binds_provenance(tmp_path: Path):
    paths = _scenario_from_fixture(tmp_path, "successful.jsonl")
    assert _run(paths) == 0
    records = _load_jsonl(paths["output"])
    assert len(records) == 1
    record = records[0]
    jsonschema.Draft7Validator(
        load_schema(V1_1_SCHEMA),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    ).validate(record)
    assert record["schema_version"] in ("1.1", "v1.1")
    assert record["terminal_disposition"] == "successful"
    assert record["execution"]["producer_completed"] is True
    assert record["execution"]["producer_partial"] is False
    assert "provenance_binding" not in record
    binding = record["execution_provenance"]
    assert binding["run_id"] == "run-success"
    assert binding["session_id"] == "sess-run-success"
    assert binding["task_id"] == "task-run-success"
    assert binding["raw_trace_id"] == "trace-run-success"
    assert binding["input_sha256"] == sha256_bytes(paths["input"].read_bytes())
    assert binding["run_manifest_sha256"] == sha256_bytes(
        paths["manifest"].read_bytes()
    )
    assert binding["admission_report_sha256"] == sha256_bytes(
        paths["admission"].read_bytes()
    )
    assert binding["raw_trace_sha256"] == sha256_bytes(
        paths["input"].read_bytes().splitlines()[0]
    )
    assert binding["producer"]["name"] == "hermes-agent"
    assert binding["producer"]["revision"]
    assert binding["model"]["ollama_digest"].startswith("sha256:")
    assert binding["model"]["upstream_revision"] == "synthetic-upstream-1"
    assert binding["repository"]["base_oid"] == BASE_OID
    assert binding["repository"]["head_oid"] == HEAD_OID
    assert binding["workspace"]["kind"] == "isolated"
    assert binding["ollama"]["endpoint"].startswith("http://127.0.0.1")
    assert binding["workspace"]["id"] == "ws-isolated-001"
    assert binding["reasoning_retention"]["policy"] == "strip_hidden"
    assert binding["verifier"]["identity"] == "pytest-local"
    assert binding["verifier"]["outcome"] == "pass"
    assert binding["verifier"]["artifact_sha256"] == [VERIFIER_ARTIFACT]
    report = _report(paths)
    assert report["accepted_count"] == 1
    assert report["quarantined_count"] == 0
    assert report["rejected_count"] == 0
    assert report["records"][0]["status"] == "accepted"


def test_completed_true_with_failing_verifier_is_not_successful(tmp_path: Path):
    paths = _scenario_from_fixture(tmp_path, "completed_true_verifier_failed.jsonl")
    assert _run(paths) == 0
    records = _load_jsonl(paths["output"])
    assert len(records) == 1
    assert records[0]["execution"]["producer_completed"] is True
    assert records[0]["terminal_disposition"] == "failed"
    assert records[0]["terminal_disposition"] != "successful"
    assert records[0]["execution_provenance"]["verifier"]["outcome"] == "fail"


def test_partial_and_interrupted_runs_keep_execution_metadata(tmp_path: Path):
    partial = _scenario_from_fixture(tmp_path / "partial", "partial.jsonl")
    interrupted = _scenario_from_fixture(tmp_path / "interrupted", "interrupted.jsonl")
    assert _run(partial) == 0
    assert _run(interrupted) == 0
    partial_record = _load_jsonl(partial["output"])[0]
    interrupted_record = _load_jsonl(interrupted["output"])[0]
    assert partial_record["execution"]["producer_partial"] is True
    assert partial_record["execution"]["producer_completed"] is False
    assert partial_record["terminal_disposition"] == "interrupted"
    assert interrupted_record["terminal_disposition"] == "interrupted"
    assert interrupted_record["execution"]["producer_completed"] is False


def test_unverified_run_is_quarantined(tmp_path: Path):
    paths = _scenario_from_fixture(tmp_path, "unverified.jsonl")
    assert _run(paths) == 0
    assert paths["output"].read_text(encoding="utf-8") == ""
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["quarantined_count"] == 1
    assert report["records"][0]["status"] == "quarantined"
    assert "unverified" in ",".join(report["records"][0]["reason_codes"])


def test_hidden_reasoning_is_stripped_from_trainable_views(tmp_path: Path):
    paths = _scenario_from_fixture(tmp_path, "hidden_reasoning.jsonl")
    assert _run(paths) == 0
    dumped = paths["output"].read_text(encoding="utf-8")
    assert "<think>" not in dumped.lower()
    assert "</think>" not in dumped.lower()
    assert "hide this private chain" not in dumped
    assert "internal plan must not be trainable" not in dumped
    record = _load_jsonl(paths["output"])[0]
    assert "I'll inspect loop.py" in json.dumps(record)
    assert record["execution_provenance"]["reasoning_retention"]["policy"] == (
        "strip_hidden"
    )


def test_secret_leakage_is_rejected_and_not_emitted(tmp_path: Path):
    paths = _scenario_from_fixture(tmp_path, "secret_leak.jsonl")
    original = paths["input"].read_bytes()
    assert _run(paths) == 0
    assert paths["input"].read_bytes() == original
    assert paths["output"].read_text(encoding="utf-8") == ""
    report = _report(paths)
    assert report["rejected_count"] == 1
    assert "secret" in ",".join(report["records"][0]["reason_codes"])
    dumped = paths["report"].read_text(encoding="utf-8")
    assert "ghp_" not in dumped


def test_query_secrets_are_stripped_from_accepted_output(tmp_path: Path):
    paths = _scenario_from_fixture(tmp_path, "query_secret.jsonl")
    assert _run(paths) == 0
    dumped = paths["output"].read_text(encoding="utf-8")
    assert "supersecretvalue1" not in dumped
    assert "access_token" not in dumped
    record = _load_jsonl(paths["output"])[0]
    assert record["schema_version"] in ("1.1", "v1.1")


def test_malformed_and_truncated_jsonl_are_rejected(tmp_path: Path):
    malformed = _scenario_from_fixture(tmp_path / "malformed", "malformed.jsonl")
    truncated = _scenario_from_fixture(tmp_path / "truncated", "truncated.jsonl")
    assert _run(malformed) == 0
    assert _run(truncated) == 0
    malformed_report = _report(malformed)
    truncated_report = _report(truncated)
    assert malformed_report["rejected_count"] == 1
    assert truncated_report["rejected_count"] == 1
    assert malformed["output"].read_text(encoding="utf-8") == ""
    assert truncated["output"].read_text(encoding="utf-8") == ""
    assert "invalid_json" in malformed_report["records"][0]["reason_codes"]
    assert "invalid_json" in truncated_report["records"][0]["reason_codes"]


def test_duplicate_keys_and_duplicate_records_are_rejected(tmp_path: Path):
    keys = _scenario_from_fixture(tmp_path / "keys", "duplicate_keys.jsonl")
    records = _scenario_from_fixture(tmp_path / "records", "duplicate_records.jsonl")
    assert _run(keys) == 0
    assert _run(records) == 0
    keys_report = _report(keys)
    records_report = _report(records)
    assert keys_report["rejected_count"] == 1
    assert "duplicate_key" in ",".join(keys_report["records"][0]["reason_codes"])
    statuses = [row["status"] for row in records_report["records"]]
    assert statuses[0] == "accepted"
    assert statuses[1] == "rejected"
    assert "duplicate" in ",".join(records_report["records"][1]["reason_codes"])
    assert len(_load_jsonl(records["output"])) == 1


def test_admission_digest_mismatch_fails_closed(tmp_path: Path):
    paths = write_scenario(tmp_path, [hermes_record(run_id="run-mismatch")])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["admission_report_sha256"] = "0" * 64
    write_json(paths["manifest"], manifest)
    assert _run(paths) == 0
    assert paths["output"].read_text(encoding="utf-8") == ""
    report = _report(paths)
    assert report["rejected_count"] >= 1
    assert any("digest" in ",".join(row["reason_codes"]) for row in report["records"])


def test_output_and_report_cannot_alias_inputs(tmp_path: Path):
    paths = write_scenario(tmp_path, [hermes_record(run_id="run-alias")])
    original = paths["input"].read_bytes()
    colliding = dict(paths)
    colliding["output"] = paths["input"]
    assert _run(colliding) == 2
    assert paths["input"].read_bytes() == original
    colliding_report = dict(paths)
    colliding_report["report"] = paths["manifest"]
    assert _run(colliding_report) == 2
    colliding_same = dict(paths)
    colliding_same["output"] = paths["report"]
    colliding_same["report"] = paths["report"]
    assert _run(colliding_same) == 2


def test_hard_link_output_collision_fails_closed(tmp_path: Path):
    paths = write_scenario(tmp_path, [hermes_record(run_id="run-link")])
    linked = tmp_path / "linked-output.jsonl"
    linked.hardlink_to(paths["input"])
    colliding = dict(paths)
    colliding["output"] = linked
    original = paths["input"].read_bytes()
    assert _run(colliding) == 2
    assert paths["input"].read_bytes() == original


def test_rerun_is_byte_identical_canonical_json(tmp_path: Path):
    paths = write_scenario(
        tmp_path,
        [hermes_record(run_id="run-stable", content="Stable observable answer.")],
    )
    assert _run(paths) == 0
    first_out = paths["output"].read_bytes()
    first_report = paths["report"].read_bytes()
    assert b"\r" not in first_out
    assert first_out.endswith(b"\n")
    assert first_report.endswith(b"\n")
    assert _run(paths) == 0
    assert paths["output"].read_bytes() == first_out
    assert paths["report"].read_bytes() == first_report
    line = first_out.strip()
    parsed = json.loads(line)
    assert line == canonical_dumps(parsed).encode("utf-8")
    assert list(json.loads(first_report).keys()) == sorted(
        json.loads(first_report).keys()
    )


def test_local_file_boundary_rejects_http_and_allows_injected_replay(tmp_path: Path):
    paths = write_scenario(tmp_path, [hermes_record(run_id="run-replay")])
    frozen = {
        paths["input"].resolve(): paths["input"].read_bytes(),
        paths["manifest"].resolve(): paths["manifest"].read_bytes(),
        paths["admission"].resolve(): paths["admission"].read_bytes(),
    }
    boundary = LocalFileBoundary(opener=lambda path: frozen[path.resolve()])
    report = normalize_files(
        input_path=paths["input"],
        run_manifest_path=paths["manifest"],
        model_admission_path=paths["admission"],
        output_path=paths["output"],
        report_path=paths["report"],
        source=boundary,
    )
    assert report["accepted_count"] == 1
    with pytest.raises(ValueError, match="local file"):
        LocalFileBoundary().read_bytes(Path("https://example.test/trace.jsonl"))


def test_existing_v0_and_v1_validators_still_pass():
    v0 = jsonschema.Draft7Validator(
        load_schema(V0_SCHEMA),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )
    v1 = jsonschema.Draft7Validator(
        load_schema(V1_SCHEMA),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )
    errors = validate_file(
        ROOT / "tests" / "fixtures" / "v1" / "software_valid.jsonl",
        v0,
        v1,
        strict_policy=True,
    )
    assert errors == []


def test_replacement_does_not_ship_github_page_cassettes():
    assert not (ROOT / "schemas" / "github_page_cassette.schema.json").exists()
    assert not (ROOT / "scripts" / "lib" / "github_page_fixtures.py").exists()
    assert not (
        ROOT / "tests" / "fixtures" / "github" / "pages" / "pr89_multipage.json"
    ).exists()


def test_custody_and_reasoning_docs_exist():
    assert (ROOT / "docs" / "hermes-raw-trace-custody.md").is_file()
    assert (ROOT / "docs" / "hermes-reasoning-retention.md").is_file()


def test_nonfinite_json_is_rejected(tmp_path: Path):
    paths = write_scenario(tmp_path, [hermes_record(run_id="run-nan")])
    text = paths["input"].read_text(encoding="utf-8").replace('"run-nan"', "NaN")
    paths["input"].write_text(text, encoding="utf-8")
    assert _run(paths) == 0
    report = _report(paths)
    assert report["rejected_count"] == 1
    assert "invalid_json" in report["records"][0]["reason_codes"]


def test_admission_model_mismatch_fails_closed(tmp_path: Path):
    admission = default_admission()
    admission["model"]["name"] = "other-model"
    paths = write_scenario(
        tmp_path,
        [hermes_record(run_id="run-model")],
        admission=admission,
    )
    assert _run(paths) == 0
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1
    joined = ",".join(report["records"][0]["reason_codes"])
    assert "model" in joined or "mismatch" in joined


def test_shared_admission_fixture_uses_singular_decision_interface():
    payload = json.loads(ADMISSION_FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "local_model_admission_v1"
    assert payload["decision"] == "accepted"
    assert payload["reasons"] == []
    assert payload["model"]["ollama_digest"].startswith("sha256:")
    assert payload["runtime"]["name"] == "ollama"
    assert payload["rights"]["identifier"] == "Apache-2.0"
    assert payload["provider"]["config"]["no_cloud"] is True
    assert payload["provider"]["config"]["cloud_fallback_allowed"] is False
    assert payload["cloud_fallback_allowed"] is False
    assert payload["fallback_evidence"]["cloud_fallback_allowed"] is False
    assert payload["fallback_evidence"]["no_cloud"] is True
    assert payload["probe"]["endpoint"].startswith("http://127.0.0.1")
    assert set(payload["input_digests"]) >= {"admissions", "rights", "probe"}
    assert payload["evidence_digest"] == seal_admission(payload)["evidence_digest"]
    assert default_admission()["decision"] == "accepted"
    assert "disposition" not in default_admission()


def test_legacy_disposition_admission_is_rejected(tmp_path: Path):
    admission = default_admission()
    admission.pop("decision")
    admission["disposition"] = "accepted"
    paths = write_scenario(
        tmp_path,
        [hermes_record(run_id="run-disposition")],
        admission=admission,
    )
    assert _run(paths) == 0
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1
    joined = ",".join(report["records"][0]["reason_codes"])
    assert "admission" in joined


def test_non_accepted_decision_admission_is_rejected(tmp_path: Path):
    admission = default_admission(decision="quarantined")
    paths = write_scenario(
        tmp_path,
        [hermes_record(run_id="run-decision")],
        admission=admission,
    )
    assert _run(paths) == 0
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1
    assert "admission_rejected" in report["records"][0]["reason_codes"]


def test_manifest_verifier_conflict_with_trace_is_rejected(tmp_path: Path):
    record = hermes_record(
        run_id="run-verifier-conflict",
        trace_verifier={
            "artifact_sha256": [
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            ],
            "identity": "trace-self",
            "outcome": "pass",
            "version": "0.0.0",
        },
    )
    paths = write_scenario(
        tmp_path,
        [record],
        manifest_overrides={"verifier": default_verifier(outcome="fail")},
    )
    assert _run(paths) == 0
    assert paths["output"].read_text(encoding="utf-8") == ""
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1
    joined = ",".join(report["records"][0]["reason_codes"])
    assert "verifier" in joined


def test_manifest_repository_and_workspace_conflicts_are_rejected(tmp_path: Path):
    record = hermes_record(run_id="run-repo-conflict")
    record["repository"]["head_oid"] = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    record["workspace"] = {"id": "ws-shared-home", "kind": "shared"}
    paths = write_scenario(tmp_path, [record])
    assert _run(paths) == 0
    dumped = paths["output"].read_text(encoding="utf-8")
    assert dumped == ""
    assert "deadbeef" not in dumped
    assert "ws-shared-home" not in dumped
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1
    joined = ",".join(report["records"][0]["reason_codes"])
    assert "repository" in joined or "workspace" in joined or "conflict" in joined


def test_manifest_ollama_credential_is_rejected_and_not_emitted(tmp_path: Path):
    paths = write_scenario(
        tmp_path,
        [hermes_record(run_id="run-manifest-secret")],
        manifest_overrides={
            "ollama": {"config": {"api_key": SYNTHETIC_GITHUB_TOKEN}},
        },
    )
    assert _run(paths) == 0
    assert paths["output"].read_text(encoding="utf-8") == ""
    report_text = paths["report"].read_text(encoding="utf-8")
    assert "ghp_" not in report_text
    report = json.loads(report_text)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1
    assert "secret" in ",".join(report["records"][0]["reason_codes"])


def test_missing_ids_and_string_booleans_are_rejected(tmp_path: Path):
    record = hermes_record(run_id="run-coerced")
    del record["session_id"]
    del record["task_id"]
    del record["raw_trace_id"]
    record["completed"] = "false"
    record["partial"] = "false"
    paths = write_scenario(tmp_path, [record])
    assert _run(paths) == 0
    assert paths["output"].read_text(encoding="utf-8") == ""
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1
    joined = ",".join(report["records"][0]["reason_codes"])
    assert "identity" in joined or "execution" in joined or "invalid" in joined


def test_empty_input_with_invalid_binding_is_explicitly_rejected(tmp_path: Path):
    paths = write_scenario(tmp_path, [])
    paths["input"].write_bytes(b"")
    write_json(paths["manifest"], {"schema_version": "not-a-manifest"})
    write_json(paths["admission"], {"schema_version": "not-admission"})
    assert _run(paths) == 0
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["quarantined_count"] == 0
    assert report["rejected_count"] >= 1
    assert report["records"]
    assert report["records"][0]["reason_codes"]


def test_empty_input_with_valid_binding_is_explicitly_rejected(tmp_path: Path):
    paths = write_scenario(tmp_path, [])
    paths["input"].write_bytes(b"")
    assert _run(paths) == 0
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["quarantined_count"] == 0
    assert report["rejected_count"] >= 1
    assert "empty_input" in report["records"][0]["reason_codes"]


def test_structured_credential_content_is_rejected(tmp_path: Path):
    record = hermes_record(
        run_id="run-structured-secret",
        content={"api_key": "ordinarysecretvalue123", "answer": "visible"},
    )
    paths = write_scenario(tmp_path, [record])
    assert _run(paths) == 0
    dumped = paths["output"].read_text(encoding="utf-8")
    assert dumped == ""
    assert "ordinarysecretvalue123" not in dumped
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1


def test_credential_url_path_is_rejected(tmp_path: Path):
    record = hermes_record(
        run_id="run-path-secret",
        extra_message={
            "role": "tool",
            "name": "fetch",
            "timestamp": "2026-09-21T12:00:30Z",
            "content": "https://example.test/token/ordinarysecretvalue123",
        },
    )
    paths = write_scenario(tmp_path, [record])
    assert _run(paths) == 0
    dumped = paths["output"].read_text(encoding="utf-8")
    assert dumped == ""
    assert "ordinarysecretvalue123" not in dumped
    report = _report(paths)
    assert report["rejected_count"] >= 1


def test_casefolded_hidden_reasoning_and_manifest_hidden_fields_are_rejected(
    tmp_path: Path,
):
    record = hermes_record(run_id="run-reason-case")
    record["messages"][1]["Reasoning"] = "case-folded private plan"
    paths = write_scenario(
        tmp_path,
        [record],
        manifest_overrides={
            "reasoning_retention": {"hidden_reasoning": "keep this private"},
        },
    )
    assert _run(paths) == 0
    dumped = paths["output"].read_text(encoding="utf-8")
    assert "case-folded private plan" not in dumped
    assert "keep this private" not in dumped
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1


def test_malformed_identity_and_invalid_url_do_not_abort_batch(tmp_path: Path):
    bad_id = hermes_record(run_id="run-list-id")
    bad_id["run_id"] = []
    bad_url = hermes_record(
        run_id="run-bad-url",
        extra_message={
            "role": "tool",
            "name": "fetch",
            "timestamp": "2026-09-21T12:00:30Z",
            "content": "http://example.test:notaport/path",
        },
    )
    good = hermes_record(run_id="run-after-bad")
    paths = write_scenario(tmp_path, [bad_id, bad_url, good])
    assert _run(paths) == 0
    report = _report(paths)
    assert len(report["records"]) == 3
    assert report["records"][0]["status"] == "rejected"
    assert report["records"][1]["status"] == "rejected"
    assert report["records"][2]["status"] == "accepted"
    assert report["accepted_count"] == 1
    assert len(_load_jsonl(paths["output"])) == 1


def test_successful_disposition_requires_independent_verifier_evidence(tmp_path: Path):
    paths = write_scenario(
        tmp_path,
        [hermes_record(run_id="run-self-verify")],
        manifest_overrides={
            "verifier": {
                "artifact_sha256": [],
                "identity": "hermes-agent",
                "outcome": "pass",
                "version": "0.4.0",
            }
        },
    )
    assert _run(paths) == 0
    assert paths["output"].read_text(encoding="utf-8") == ""
    report = _report(paths)
    assert report["accepted_count"] == 0
    joined = ",".join(report["records"][0]["reason_codes"])
    assert "verifier" in joined


def test_normalized_output_passes_strict_jsonl_validator(tmp_path: Path):
    paths = write_scenario(tmp_path, [hermes_record(run_id="run-validate")])
    assert _run(paths) == 0
    v0 = jsonschema.Draft7Validator(
        load_schema(V0_SCHEMA),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )
    v1 = jsonschema.Draft7Validator(
        load_schema(V1_SCHEMA),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )
    errors = validate_file(paths["output"], v0, v1, strict_policy=True)
    assert errors == []
    v0_errors = validate_file(
        ROOT / "tests" / "fixtures" / "v1" / "software_valid.jsonl",
        v0,
        v1,
        strict_policy=True,
    )
    assert v0_errors == []


def test_trajectory_id_uses_full_identity_tuple(tmp_path: Path):
    first = hermes_record(run_id="same")
    second = hermes_record(run_id="same")
    second["session_id"] = "sess-other"
    second["task_id"] = "task-other"
    second["raw_trace_id"] = "trace-other"
    paths = write_scenario(tmp_path, [first, second])
    assert _run(paths) == 0
    records = _load_jsonl(paths["output"])
    assert len(records) == 2
    assert records[0]["trajectory_id"] != records[1]["trajectory_id"]
    assert records[0]["trajectory_id"] != "hermes:same"
    assert "sess-same" in records[0]["trajectory_id"]
    assert "sess-other" in records[1]["trajectory_id"]


def test_non_object_message_is_rejected(tmp_path: Path):
    record = hermes_record(run_id="run-msg")
    record["messages"].append("not-an-object")
    paths = write_scenario(tmp_path, [record])
    assert _run(paths) == 0
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert report["rejected_count"] >= 1
    assert paths["output"].read_text(encoding="utf-8") == ""


def test_ipv6_loopback_endpoint_keeps_brackets(tmp_path: Path):
    paths = write_scenario(
        tmp_path,
        [hermes_record(run_id="run-ipv6")],
        manifest_overrides={"ollama": {"endpoint": "http://[::1]:11434"}},
        admission=default_admission(),
    )
    admission = json.loads(paths["admission"].read_text(encoding="utf-8"))
    admission["runtime"]["endpoint"] = "http://[::1]:11434"
    admission["probe"]["endpoint"] = "http://[::1]:11434"
    write_json(paths["admission"], seal_admission(admission))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["admission_report_sha256"] = sha256_bytes(paths["admission"].read_bytes())
    write_json(paths["manifest"], manifest)
    assert _run(paths) == 0
    record = _load_jsonl(paths["output"])[0]
    assert record["execution_provenance"]["ollama"]["endpoint"] == "http://[::1]:11434"
    assert "http://::1" not in paths["output"].read_text(encoding="utf-8")


def test_accepted_admission_with_reasons_is_rejected(tmp_path: Path):
    admission = default_admission(reasons=["rights_conflict"])
    paths = write_scenario(tmp_path, [hermes_record(run_id="run-reasons")], admission=admission)
    assert _run(paths) == 0
    assert paths["output"].read_text(encoding="utf-8") == ""
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert "admission_reasons" in report["records"][0]["reason_codes"]


def test_runtime_and_probe_conflicts_are_rejected(tmp_path: Path):
    admission = default_admission()
    admission["runtime"]["version"] = "9.9.9"
    admission["runtime"]["endpoint"] = "http://127.0.0.1:1"
    admission["probe"]["endpoint"] = "http://127.0.0.1:2"
    paths = write_scenario(
        tmp_path, [hermes_record(run_id="run-runtime")], admission=admission
    )
    assert _run(paths) == 0
    report = _report(paths)
    assert report["accepted_count"] == 0
    joined = ",".join(report["records"][0]["reason_codes"])
    assert "runtime" in joined or "probe" in joined


def test_provider_config_without_fallback_flag_is_rejected(tmp_path: Path):
    admission = default_admission()
    admission["provider"]["config"].pop("cloud_fallback_allowed")
    paths = write_scenario(
        tmp_path, [hermes_record(run_id="run-fallback")], admission=admission
    )
    assert _run(paths) == 0
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert "cloud_fallback" in report["records"][0]["reason_codes"]


def test_fake_evidence_digest_is_rejected(tmp_path: Path):
    admission = default_admission()
    admission["evidence_digest"] = "f" * 64
    paths = write_scenario(
        tmp_path,
        [hermes_record(run_id="run-digest")],
        admission=admission,
    )
    # write_scenario reseals; overwrite the sealed file with a fake digest.
    stored = json.loads(paths["admission"].read_text(encoding="utf-8"))
    stored["evidence_digest"] = "f" * 64
    write_json(paths["admission"], stored)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["admission_report_sha256"] = sha256_bytes(paths["admission"].read_bytes())
    write_json(paths["manifest"], manifest)
    assert _run(paths) == 0
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert "evidence_digest" in report["records"][0]["reason_codes"]


def test_shared_or_missing_workspace_kind_is_rejected(tmp_path: Path):
    record = hermes_record(run_id="run-shared-ws")
    record.pop("workspace")
    shared = write_scenario(
        tmp_path / "shared",
        [record],
        manifest_overrides={"workspace": {"kind": "shared"}},
    )
    missing = write_scenario(tmp_path / "missing", [record])
    missing_manifest = json.loads(missing["manifest"].read_text(encoding="utf-8"))
    missing_manifest["workspace"].pop("kind")
    write_json(missing["manifest"], missing_manifest)
    assert _run(shared) == 0
    assert _run(missing) == 0
    for paths in (shared, missing):
        assert paths["output"].read_text(encoding="utf-8") == ""
        report = _report(paths)
        assert report["accepted_count"] == 0
        assert report["rejected_count"] >= 1
        assert "workspace" in ",".join(report["records"][0]["reason_codes"])


def test_casefolded_hermes_verifier_identity_is_not_external(tmp_path: Path):
    paths = write_scenario(
        tmp_path,
        [hermes_record(run_id="run-self-case")],
        manifest_overrides={
            "verifier": {
                "artifact_sha256": [VERIFIER_ARTIFACT],
                "identity": "Hermes-Agent",
                "outcome": "pass",
                "version": "0.4.0",
            }
        },
    )
    assert _run(paths) == 0
    dumped = paths["output"].read_text(encoding="utf-8")
    assert dumped == ""
    assert "successful" not in dumped
    report = _report(paths)
    assert report["accepted_count"] == 0
    assert "verifier" in ",".join(report["records"][0]["reason_codes"])


def test_hidden_equivalent_tags_are_stripped_and_unclosed_are_rejected(tmp_path: Path):
    closed = hermes_record(
        run_id="run-hidden-tags",
        content=(
            "<analysis>private chain</analysis>"
            "<reasoning>private chain</reasoning>"
            "<scratchpad>private chain</scratchpad>"
            "Visible answer."
        ),
    )
    paths = write_scenario(tmp_path / "closed", [closed])
    assert _run(paths) == 0
    dumped = paths["output"].read_text(encoding="utf-8")
    assert "private chain" not in dumped
    assert "Visible answer." in dumped
    unclosed = hermes_record(
        run_id="run-unclosed",
        content="<analysis>private chain",
    )
    rejected = write_scenario(tmp_path / "open", [unclosed])
    assert _run(rejected) == 0
    open_dump = rejected["output"].read_text(encoding="utf-8")
    assert open_dump == ""
    assert "private chain" not in open_dump
    assert _report(rejected)["rejected_count"] >= 1


def test_repository_url_query_credential_is_not_emitted(tmp_path: Path):
    secret_url = (
        "https://github.com/rmems/synthetic-loop?access_token=ordinarysecretvalue123"
    )
    omitted = hermes_record(run_id="run-url-omit")
    omitted.pop("repository")
    agreed = hermes_record(run_id="run-url-agree")
    agreed["repository"]["url"] = secret_url
    for name, record in (("omit", omitted), ("agree", agreed)):
        paths = write_scenario(
            tmp_path / name,
            [record],
            manifest_overrides={"repository": {"url": secret_url}},
        )
        assert _run(paths) == 0
        dumped = paths["output"].read_text(encoding="utf-8") + paths["report"].read_text(
            encoding="utf-8"
        )
        assert "ordinarysecretvalue123" not in dumped
        assert "access_token" not in dumped


def test_header_and_json_credential_content_is_rejected(tmp_path: Path):
    bearer = hermes_record(
        run_id="run-bearer",
        content="Authorization: Bearer ordinarysecretvalue123",
    )
    header = hermes_record(
        run_id="run-header",
        content='{"X-API-Key":"ordinarysecretvalue123"}',
    )
    for name, record in (("bearer", bearer), ("header", header)):
        paths = write_scenario(tmp_path / name, [record])
        assert _run(paths) == 0
        dumped = paths["output"].read_text(encoding="utf-8")
        assert dumped == ""
        assert "ordinarysecretvalue123" not in dumped
        assert _report(paths)["rejected_count"] >= 1


def test_trajectory_id_encoding_resists_delimiter_collision(tmp_path: Path):
    first = hermes_record(run_id="a/b")
    first["session_id"] = "c"
    first["task_id"] = "d"
    first["raw_trace_id"] = "e"
    second = hermes_record(run_id="a")
    second["session_id"] = "b"
    second["task_id"] = "c"
    second["raw_trace_id"] = "d/e"
    paths = write_scenario(tmp_path, [first, second])
    assert _run(paths) == 0
    records = _load_jsonl(paths["output"])
    assert len(records) == 2
    assert records[0]["trajectory_id"] != records[1]["trajectory_id"]
    assert records[0]["trajectory_id"] != "hermes:a/b/c/d/e"
    assert records[0]["execution_provenance"]["run_id"] == "a/b"
    assert records[1]["execution_provenance"]["raw_trace_id"] == "d/e"


def test_malformed_tool_payload_is_rejected(tmp_path: Path):
    record = hermes_record(
        run_id="run-tool",
        extra_message={
            "role": "tool",
            "name": "fetch",
            "timestamp": "2026-09-21T12:00:30Z",
            "content": '{"unterminated":',
        },
    )
    paths = write_scenario(tmp_path, [record])
    assert _run(paths) == 0
    assert paths["output"].read_text(encoding="utf-8") == ""
    report = _report(paths)
    assert report["rejected_count"] >= 1
    assert "invalid_tool_payload" in report["records"][0]["reason_codes"]


def test_unknown_message_field_is_rejected(tmp_path: Path):
    record = hermes_record(run_id="run-extra-field")
    record["messages"][1]["private_notes"] = "producer private container"
    paths = write_scenario(tmp_path, [record])
    assert _run(paths) == 0
    dumped = paths["output"].read_text(encoding="utf-8")
    assert "producer private container" not in dumped
    assert _report(paths)["accepted_count"] == 0


def test_canonical_dumps_rejects_nonfinite_numbers():
    with pytest.raises(ValueError):
        canonical_dumps({"value": float("nan")})
