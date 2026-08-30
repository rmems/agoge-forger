from unittest.mock import patch

from typer.testing import CliRunner

from agoge_forger.cli import app

REVISION = "156170697656c48f69915b33a2fb44110242187c"
MODEL_ID = "openbmb/MiniCPM5-1B-Base"


def test_model_metadata_forwards_immutable_revision():
    with patch("agoge_forger.cli.get_model_config_metadata", return_value={}) as inspect:
        result = CliRunner().invoke(
            app,
            ["model-metadata", "--model-id", MODEL_ID, "--revision", REVISION],
        )

    assert result.exit_code == 0
    inspect.assert_called_once_with(MODEL_ID, False, REVISION)


def test_model_inspection_forwards_immutable_revision():
    with patch("agoge_forger.cli._inspect_model") as inspect:
        result = CliRunner().invoke(
            app,
            ["inspect-model", "--model-id", MODEL_ID, "--revision", REVISION],
        )

    assert result.exit_code == 0
    inspect.assert_called_once_with(MODEL_ID, False, REVISION)


def test_lora_target_inspection_forwards_immutable_revision(tmp_path):
    output = tmp_path / "targets.json"
    with patch("agoge_forger.cli._inspect_lora_targets") as inspect:
        result = CliRunner().invoke(
            app,
            [
                "inspect-lora-targets",
                "--model-id",
                MODEL_ID,
                "--revision",
                REVISION,
                "--out",
                str(output),
            ],
        )

    assert result.exit_code == 0
    inspect.assert_called_once_with(MODEL_ID, False, str(output), REVISION)
