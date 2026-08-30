from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app


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


def test_serve_vllm_requires_model(runner: CliRunner, caplog):
    result = runner.invoke(app, ["serve-vllm", "--dry-run"])
    assert result.exit_code != 0
    assert "Model ID is required" in result.output or "Model ID is required" in caplog.text
