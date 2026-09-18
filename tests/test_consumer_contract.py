from pathlib import Path

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.consumer_contract import (
    ConsumerContractError,
    consume_file,
    future_event_errors,
    refuse_gpu,
    run_consumer_contract,
)

ROOT = Path(__file__).resolve().parents[1]
GOOD = ROOT / "tests" / "fixtures" / "consumer" / "messages_and_instruction.jsonl"
LEAK = ROOT / "tests" / "fixtures" / "consumer" / "negative" / "future_leakage.jsonl"


def test_consumer_contract_parses_messages_and_instruction():
    sidecar = consume_file(GOOD)
    assert sidecar["ok"]
    assert sidecar["schema_version"] == "agoge.consumer-sidecar.v1"
    assert sidecar["parser"] == "agoge_forger.datasets.normalize_row"
    assert {row["format"] for row in sidecar["rows"]} == {"messages", "instruction"}
    assert sidecar["source_sha256"]


def test_consumer_contract_detects_future_event_leakage():
    record = {"_prometheus": {"event_timestamps": ["2026-02-02T00:00:00Z", "2026-02-01T00:00:00Z"]}}
    assert any("future-event leakage" in error for error in future_event_errors(record))
    sidecar = consume_file(LEAK)
    assert sidecar["ok"] is False


def test_run_consumer_contract_writes_sidecar(tmp_path):
    out_dir = tmp_path / "sidecars"
    sidecars = run_consumer_contract([str(GOOD)], str(out_dir))
    assert len(sidecars) == 1
    assert (out_dir / "messages_and_instruction.sidecar.json").is_file()


def test_run_consumer_contract_refuses_gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ConsumerContractError, match="no GPU"):
        run_consumer_contract([str(GOOD)], str(tmp_path / "out"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    refuse_gpu()


def test_consumer_contract_cli(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "consumer-contract",
            "--input",
            str(GOOD),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "messages_and_instruction.sidecar.json").is_file()


def test_consume_file_records_invalid_json_instead_of_crashing(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json}\n")
    sidecar = consume_file(path)
    assert sidecar["ok"] is False
    assert any("Invalid JSON" in error for error in sidecar["errors"])


def test_consumer_contract_cli_rejects_future_leakage(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "consumer-contract",
            "--input",
            str(LEAK),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "future-event leakage" in result.output


def test_same_stem_inputs_keep_distinct_sidecars(tmp_path):
    first = tmp_path / "a" / "dup.jsonl"
    second = tmp_path / "b" / "dup.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('{"text": "hello"}\n')
    second.write_text('{"text": "world"}\n')
    out_dir = tmp_path / "out"
    sidecars = run_consumer_contract([str(first), str(second)], str(out_dir))
    names = {path.name for path in out_dir.glob("*.sidecar.json")}
    assert len(names) == 2
    assert {sidecar["source_path"] for sidecar in sidecars} == {
        str(first.resolve()),
        str(second.resolve()),
    }


def test_pr_workflows_never_receive_hf_token():
    workflows = ROOT / ".github" / "workflows"
    for path in workflows.glob("*.yml"):
        text = path.read_text()
        if "pull_request:" not in text:
            continue
        assert "secrets.HF_TOKEN" not in text, (
            f"{path.name} is a PR workflow and must not receive secrets.HF_TOKEN"
        )
