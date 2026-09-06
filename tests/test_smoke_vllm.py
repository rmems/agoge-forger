import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.providers.chat_completions import ChatCompletionsConfig
from agoge_forger.serving.smoke import run_vllm_smoke

NON_STREAM_BODY = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Paris is the capital of France.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
}


def _mock_post_response(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def test_smoke_vllm_success_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("agoge_forger.providers.chat_completions.httpx.post") as mock_post:
        mock_post.return_value = _mock_post_response(NON_STREAM_BODY)

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        run_dir, results = run_vllm_smoke(cfg, run_name="smoke_ok", prompt="Hello")

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].result.response_text == "Paris is the capital of France."
    assert results[0].result.error == ""
    assert results[0].result.input_tokens == 10
    assert results[0].result.output_tokens == 5

    run_path = Path(run_dir)
    assert (run_path / "smoke_vllm_result.json").is_file()
    assert (run_path / "results.jsonl").is_file()
    assert (run_path / "summary.md").is_file()

    raw_path = results[0].result.raw_response_path
    assert raw_path and os.path.exists(raw_path)

    summary = (run_path / "summary.md").read_text(encoding="utf-8")
    assert "vLLM Compatibility Smoke Summary" in summary
    assert "**OK:** 1" in summary
    assert "**Errors:** 0" in summary


def test_smoke_vllm_streaming_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks = [
        {"id": "s", "choices": [{"delta": {"content": "Paris"}, "finish_reason": None}]},
        {
            "id": "s",
            "choices": [{"delta": {"content": " is"}, "finish_reason": None}],
        },
        {
            "id": "s",
            "choices": [{"delta": {"content": " the capital."}, "finish_reason": None}],
        },
        {
            "id": "s",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
    ]
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.iter_lines.return_value = iter(lines)
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("agoge_forger.providers.chat_completions.httpx.stream") as mock_stream:
        mock_stream.return_value = mock_ctx
        cfg = ChatCompletionsConfig(model="test-model", stream=True)
        _run_dir, results = run_vllm_smoke(cfg, run_name="smoke_stream", prompt="Hello")

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].result.response_text == "Paris is the capital."
    assert results[0].result.input_tokens == 8
    assert results[0].result.output_tokens == 4

    raw_path = results[0].result.raw_response_path
    assert raw_path and os.path.exists(raw_path)


def test_smoke_vllm_failure_still_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("agoge_forger.providers.chat_completions.httpx.post") as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        run_dir, results = run_vllm_smoke(cfg, run_name="smoke_fail", prompt="Hello")

    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].result.error == "Connection refused"

    run_path = Path(run_dir)
    assert (run_path / "smoke_vllm_result.json").is_file()
    assert (run_path / "results.jsonl").is_file()
    assert (run_path / "summary.md").is_file()

    summary = (run_path / "summary.md").read_text(encoding="utf-8")
    assert "**Errors:** 1" in summary


def test_smoke_vllm_dry_run_writes_artifacts_and_exits_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = ChatCompletionsConfig(model="test-model", stream=False)
    run_dir, results = run_vllm_smoke(cfg, run_name="smoke_dry", prompt="Hello", dry_run=True)

    assert len(results) == 1
    assert results[0].status == "dry_run"
    assert results[0].result.response_text == "Dry-run response."

    run_path = Path(run_dir)
    assert (run_path / "smoke_vllm_result.json").is_file()

    data = json.loads((run_path / "results.jsonl").read_text(encoding="utf-8"))
    assert data["status"] == "dry_run"
    assert data["workload"] == "smoke"


def test_smoke_vllm_loads_prompt_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text("system: You are a test assistant.\nprompts:\n  - One\n  - Two\n")

    cfg = ChatCompletionsConfig(model="m", stream=False)
    _run_dir, results = run_vllm_smoke(
        cfg, run_name="ps", prompt_set=str(prompt_path), dry_run=True
    )

    assert len(results) == 2
    assert {r.prompt for r in results} == {"One", "Two"}


def test_smoke_vllm_cli_success(runner: CliRunner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("agoge_forger.providers.chat_completions.httpx.post") as mock_post:
        mock_post.return_value = _mock_post_response(NON_STREAM_BODY)

        result = runner.invoke(
            app,
            [
                "smoke-vllm",
                "--model",
                "test-model",
                "--prompt",
                "Hello",
                "--run-name",
                "cli_ok",
            ],
        )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "runs" / "cli_ok" / "smoke_vllm_result.json").is_file()


def test_smoke_vllm_cli_failure_returns_non_zero(runner: CliRunner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("agoge_forger.providers.chat_completions.httpx.post") as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        result = runner.invoke(
            app,
            [
                "smoke-vllm",
                "--model",
                "test-model",
                "--prompt",
                "Hello",
                "--run-name",
                "cli_fail",
            ],
        )

    assert result.exit_code == 1, result.output
    assert (tmp_path / "runs" / "cli_fail" / "smoke_vllm_result.json").is_file()


def test_smoke_vllm_cli_requires_model(runner: CliRunner):
    result = runner.invoke(app, ["smoke-vllm"])
    assert result.exit_code != 0
    assert "Model is required" in result.output


def test_effective_smoke_stream_defaults():
    from agoge_forger.cli import _effective_smoke_stream

    assert _effective_smoke_stream(True, None) is True
    assert _effective_smoke_stream(False, "config.yaml") is False
    assert _effective_smoke_stream(None, None) is False
    assert _effective_smoke_stream(None, "config.yaml") is None


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()
