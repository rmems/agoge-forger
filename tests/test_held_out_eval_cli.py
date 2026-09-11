import subprocess  # nosec B404 - isolated interpreter import-cycle regression
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agoge_forger._token_provenance import (
    _fingerprint_mapping,
    attach_pinned_tokenizer_revision,
    derive_tokenizer_provenance,
)
from agoge_forger.cli import app
from agoge_forger.eval._tensor_schema import TensorSchemaEntry, require_matching_tensor_schema
from tests.held_out_eval_cases import canary_evaluation_case

pytestmark = pytest.mark.usefixtures("cached_test_base_config")


def test_attach_pinned_tokenizer_revision_recovers_transformers5_gap():
    class DummyTokenizer:
        name_or_path = "ibm-granite/granite-4.1-3b-base"

        def __init__(self) -> None:
            self.init_kwargs: dict[str, object] = {}

    tokenizer = DummyTokenizer()
    revision = "dacb9cb9157bec98e99b09f285c92a4d58405c96"
    attach_pinned_tokenizer_revision(tokenizer, revision)
    assert derive_tokenizer_provenance(tokenizer) == (tokenizer.name_or_path, revision)


def test_fingerprint_mapping_accepts_int_keys():
    assert _fingerprint_mapping({100256: "pad"}) == {"int:100256": "pad"}


def test_adapter_schema_allows_empty_init_lora_dtype_mismatch():
    actual = {"w": TensorSchemaEntry((16, 2560), "BF16")}
    expected = {"w": TensorSchemaEntry((16, 2560), "F32")}
    require_matching_tensor_schema(actual, expected, label="adapter", compare_dtypes=False)
    with pytest.raises(ValueError, match="wrong_dtypes"):
        require_matching_tensor_schema(actual, expected, label="adapter")


def _import_in_fresh_interpreter(script: str) -> None:
    result = subprocess.run(  # nosec B603 - fixed interpreter and import-only script
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_cli_import_survives_fresh_interpreter():
    _import_in_fresh_interpreter("from agoge_forger.cli import app")


def test_eval_generate_import_survives_fresh_interpreter():
    _import_in_fresh_interpreter("from agoge_forger.eval.generate import load_arm_model")


def test_safetensors_io_source_does_not_import_eval():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agoge_forger"
        / "artifacts"
        / "safetensors_io.py"
    )
    text = source.read_text(encoding="utf-8")
    assert "from ..eval" not in text
    assert "agoge_forger.eval" not in text


def test_safetensors_io_does_not_import_eval_package():
    """safetensors_io must not import agoge_forger.eval at module load.

    eval.generate imports assert_no_unsafe_weight_bins from this module. If
    safetensors_io also imports agoge_forger.eval, CLI startup raises ImportError.
    """
    _import_in_fresh_interpreter(
        "import sys\n"
        "from agoge_forger.artifacts import safetensors_io\n"
        "assert 'agoge_forger.eval' not in sys.modules, sorted(sys.modules)\n"
        "from agoge_forger.eval.generate import load_arm_model\n"
        "from agoge_forger.cli import app\n"
        "assert callable(safetensors_io.assert_no_unsafe_weight_bins)\n"
        "assert callable(load_arm_model)\n"
        "assert app is not None\n"
    )


def test_held_out_eval_cli_wires_library_path(tmp_path, monkeypatch):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    seen: dict[str, object] = {}

    def fake_arms(inputs):
        seen["arms"] = inputs
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
    assert seen["run"]["arms"] == (base, sft)
    runtime = seen["run"]["runtime"]
    assert runtime.trust_remote_code is False
