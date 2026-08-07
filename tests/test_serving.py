from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.serving.diagnostics import get_environment


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_prompt_set(path: Path) -> Path:
    prompt_path = path / "prompts.yaml"
    prompt_path.write_text("system: You are a test assistant.\nprompts:\n  - Hello\n  - World\n")
    return prompt_path


def test_serve_vllm_dry_run_python_frontend(runner: CliRunner, caplog):
    caplog.set_level("INFO")
    with patch("agoge_forger.serving.serve.get_vllm_version") as version_mock:
        result = runner.invoke(
            app, ["serve-vllm", "--model", "m", "--frontend", "python", "--dry-run"]
        )
    assert result.exit_code == 0
    assert not version_mock.called
    assert "-m vllm serve m" in caplog.text
    assert "VLLM_USE_RUST_FRONTEND=0" in caplog.text


def test_serve_vllm_dry_run_rust_frontend(runner: CliRunner, caplog):
    caplog.set_level("INFO")
    with (
        patch("agoge_forger.serving.serve.get_vllm_version") as version_mock,
        patch(
            "agoge_forger.serving.serve.has_rust_frontend_support",
            return_value=(True, "/fake/vllm-rs"),
        ) as rust_mock,
    ):
        result = runner.invoke(
            app, ["serve-vllm", "--model", "m", "--frontend", "rust", "--dry-run"]
        )
    assert result.exit_code == 0
    assert not version_mock.called
    assert not rust_mock.called
    assert "-m vllm serve m" in caplog.text
    assert "VLLM_USE_RUST_FRONTEND=1" in caplog.text


def test_serve_vllm_fails_when_vllm_not_installed(runner: CliRunner, caplog):
    with patch("agoge_forger.serving.serve.get_vllm_version", return_value=None):
        result = runner.invoke(app, ["serve-vllm", "--model", "m", "--frontend", "rust"])
    assert result.exit_code == 1
    assert "vLLM is not installed" in caplog.text


def test_serve_vllm_fails_when_rust_frontend_unavailable(runner: CliRunner, caplog):
    with (
        patch("agoge_forger.serving.serve.get_vllm_version", return_value="0.11.0"),
        patch(
            "agoge_forger.serving.serve.has_rust_frontend_support",
            return_value=(False, "no binary"),
        ),
    ):
        result = runner.invoke(app, ["serve-vllm", "--model", "m", "--frontend", "rust"])
    assert result.exit_code == 1
    assert "Rust frontend requested but not available" in caplog.text


def test_bench_vllm_frontend_dry_run_creates_artifacts(runner: CliRunner, tmp_path: Path):
    prompt_path = _write_prompt_set(tmp_path)
    out_dir = tmp_path / "bench_out"
    result = runner.invoke(
        app,
        [
            "bench-vllm-frontend",
            "--model",
            "m",
            "--prompt-set",
            str(prompt_path),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "results.jsonl").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "comparison.csv").exists()

    lines = (out_dir / "results.jsonl").read_text().strip().split("\n")
    assert len(lines) == 4  # 2 prompts x 2 frontends
    frontends = {json.loads(line)["frontend"] for line in lines}
    assert frontends == {"python", "rust"}

    summary = (out_dir / "summary.md").read_text()
    assert "Python vs Rust comparison" in summary
    assert "vLLM version" in summary
    assert "CUDA version" in summary
    assert "GPU name" in summary

    with open(out_dir / "comparison.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert {row["frontend"] for row in rows} == {"python", "rust"}


def test_bench_vllm_frontend_single_frontend_dry_run(runner: CliRunner, tmp_path: Path):
    prompt_path = _write_prompt_set(tmp_path)
    out_dir = tmp_path / "bench_single"
    result = runner.invoke(
        app,
        [
            "bench-vllm-frontend",
            "--model",
            "m",
            "--prompt-set",
            str(prompt_path),
            "--out-dir",
            str(out_dir),
            "--frontend",
            "rust",
            "--stream",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = (out_dir / "results.jsonl").read_text().strip().split("\n")
    assert all(json.loads(line)["frontend"] == "rust" for line in lines)
    assert all(json.loads(line)["stream"] is True for line in lines)


def test_bench_vllm_frontend_requires_prompt_set(runner: CliRunner, caplog):
    result = runner.invoke(app, ["bench-vllm-frontend", "--model", "m", "--dry-run"])
    assert result.exit_code != 0
    assert "Prompt set is required" in result.output or "Prompt set is required" in caplog.text


def test_serve_vllm_requires_model(runner: CliRunner, caplog):
    result = runner.invoke(app, ["serve-vllm", "--frontend", "python", "--dry-run"])
    assert result.exit_code != 0
    assert "Model ID is required" in result.output or "Model ID is required" in caplog.text


def test_get_environment_excludes_vllm_api_key(monkeypatch):
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "1")
    monkeypatch.setenv("VLLM_API_KEY", "secret-token")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    env = get_environment()
    assert env["VLLM_USE_RUST_FRONTEND"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "VLLM_API_KEY" not in env
    assert "HF_TOKEN" not in env
