from pathlib import Path

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from tests.held_out_eval_cases import canary_evaluation_case

pytestmark = pytest.mark.usefixtures("cached_test_base_config")


def test_held_out_eval_cli_wires_library_path(tmp_path, monkeypatch):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    seen: dict[str, object] = {}

    def fake_arms(**kwargs):
        seen["arms"] = kwargs
        return base, sft, object()

    def fake_run(**kwargs):
        seen["run"] = kwargs
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        (output / "report.md").write_text("ok\n")
        return output

    monkeypatch.setattr("agoge_forger._cli_eval._evaluation_arms", fake_arms)
    monkeypatch.setattr("agoge_forger._cli_eval.run_held_out_eval", fake_run)
    runner = CliRunner()
    output = tmp_path / "eval" / "cli"
    result = runner.invoke(
        app,
        [
            "held-out-eval",
            "--split-manifest",
            str(manifest_path),
            "--sft-artifact",
            str(tmp_path / "adapter"),
            "--output-dir",
            str(output),
            "--base-model-id",
            "example/base-model",
            "--base-revision",
            "abcdef0123456789abcdef0123456789abcdef01",
            "--context-window",
            "4096",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert seen["run"]["base"] == base
    assert seen["run"]["sft"] == sft
    assert seen["run"]["trust_remote_code"] is False
