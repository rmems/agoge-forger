from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.serving.config import ServingConfig
from agoge_forger.serving.serve import build_serve_command, serve_vllm


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_serve_vllm_dry_run(runner: CliRunner, caplog):
    caplog.set_level("INFO")
    with patch("agoge_forger.serving.serve.get_vllm_version") as version_mock:
        result = runner.invoke(app, ["serve-vllm", "--model", "m", "--dry-run"])
    assert result.exit_code == 0
    assert not version_mock.called
    assert "-m vllm serve m" in caplog.text


def test_serve_vllm_fails_when_vllm_not_installed(runner: CliRunner, caplog):
    with patch("agoge_forger.serving.serve.get_vllm_version", return_value=None):
        result = runner.invoke(app, ["serve-vllm", "--model", "m"])
    assert result.exit_code == 1
    assert "vLLM is not installed" in caplog.text


def test_serve_vllm_forces_python_frontend_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "1")
    monkeypatch.setenv("AGOGE_SERVING_TEST_KEY", "preserved")
    cfg = ServingConfig(model="example/model")

    with (
        patch("agoge_forger.serving.serve.get_vllm_version", return_value="test"),
        patch("agoge_forger.serving.serve.subprocess.run") as run_mock,
    ):
        run_mock.return_value.returncode = 0
        result = serve_vllm(cfg)

    assert result == 0
    run_mock.assert_called_once()
    assert run_mock.call_args.args == (build_serve_command(cfg),)
    assert run_mock.call_args.kwargs["check"] is False
    assert run_mock.call_args.kwargs["shell"] is False
    serve_env = run_mock.call_args.kwargs["env"]
    assert serve_env is not os.environ
    assert serve_env["VLLM_USE_RUST_FRONTEND"] == "0"
    assert serve_env["AGOGE_SERVING_TEST_KEY"] == "preserved"
    assert os.environ["VLLM_USE_RUST_FRONTEND"] == "1"


def test_serve_vllm_requires_model(runner: CliRunner, caplog):
    result = runner.invoke(app, ["serve-vllm", "--dry-run"])
    assert result.exit_code != 0
    assert "Model ID is required" in result.output or "Model ID is required" in caplog.text


def test_removed_rust_frontend_setting_is_rejected():
    with pytest.raises(ValidationError, match="frontend"):
        ServingConfig.model_validate({"model": "example/model", "frontend": "rust"})
