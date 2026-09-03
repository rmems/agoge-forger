"""PEFT adapter and serialization readiness tests for run-status."""

import json
import os
import subprocess  # nosec B404 - required for the isolated deadline regression
import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from test_run_status import (
    SKIP_IF_ROOT,
    TINY_LLAMA_CONFIG,
    _deny_read_access_or_skip,
    _make_run_dir,
    _safetensors_with_shapes,
    _safetensors_with_tensors,
    _test_base_model_path,
    _write_adapter_config,
    _write_final_adapter,
    _write_legacy_bin_adapter,
)
from transformers import AutoModelForCausalLM, GPT2Config
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.run_status import build_run_status


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_lora_config(run_dir: Path, **overrides) -> None:
    payload = {
        "base_model_name_or_path": str(_test_base_model_path(run_dir)),
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj"],
        **overrides,
    }
    (run_dir / "adapter_config.json").write_text(json.dumps(payload))


def _write_lora_weights(
    run_dir: Path,
    shapes: dict[str, tuple[int, ...]],
    *,
    legacy: bool,
) -> None:
    if legacy:
        torch.save(
            {key: torch.zeros(shape) for key, shape in shapes.items()},
            run_dir / "adapter_model.bin",
        )
        return
    (run_dir / "adapter_model.safetensors").write_bytes(_safetensors_with_shapes(shapes))


# --------------------------------------------------------------------------
# 7. Safetensors policy
# --------------------------------------------------------------------------


def test_legacy_bin_adapter_is_rejected_by_default(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_legacy_bin_adapter(run_dir)

    report = build_run_status(str(run_dir))

    assert report["allow_unsafe_serialization"] is False
    assert report["final_adapter"] == {"present": False, "path": None}
    assert report["export"] == {"ready": False, "source_path": None, "source_kind": None}
    assert report["base_model"] is None


def test_legacy_bin_adapter_is_accepted_with_allow_unsafe(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_legacy_bin_adapter(run_dir)

    report = build_run_status(str(run_dir), allow_unsafe=True)

    assert report["allow_unsafe_serialization"] is True
    assert report["final_adapter"] == {"present": True, "path": str(run_dir.resolve())}
    assert report["export"]["ready"] is True
    assert report["export"]["source_kind"] == "final_adapter"
    assert report["base_model"] == str(run_dir / ".test-base-model")

    result = runner.invoke(app, ["run-status", str(run_dir), "--allow-unsafe-serialization"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == report


def test_malformed_legacy_bin_is_not_export_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_legacy_bin_adapter(run_dir)
    (run_dir / "adapter_model.bin").write_bytes(b"not a torch archive")

    assert build_run_status(str(run_dir), allow_unsafe=True)["export"]["ready"] is False


def test_whitespace_base_model_is_not_export_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir, base_model="   ")

    report = build_run_status(str(run_dir))

    assert report["base_model"] is None
    assert report["export"]["ready"] is False


def test_unrelated_safetensor_keys_are_not_lora_weights(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(_safetensors_with_tensors("foreign.weight"))
    _write_adapter_config(run_dir)

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_lora_key_names_must_use_recognized_segments(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(
        _safetensors_with_tensors("fake_lora_A_extra", "fake_lora_B_extra")
    )
    _write_adapter_config(run_dir)

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_nonpositive_lora_rank_is_not_export_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-0.5B",
                "peft_type": "LORA",
                "r": 0,
            }
        )
    )

    assert build_run_status(str(run_dir))["export"]["ready"] is False


@pytest.mark.parametrize("legacy", [False, True], ids=["safetensors", "legacy-bin"])
@pytest.mark.parametrize(
    "case",
    [
        (((1,), (1,)), 1, False),
        (((1, 8), (8, 1)), 2, False),
        (((1, 0), (8, 1)), 1, False),
        (((1, 4), (0, 1)), 1, False),
        (((2, 8), (8, 2)), 2, True),
    ],
)
def test_lora_shapes_must_match_config_rank(tmp_path, case, legacy):
    shapes, rank, expected = case
    run_dir = _make_run_dir(tmp_path)
    tensor_shapes = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": shapes[0],
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": shapes[1],
    }
    _write_lora_weights(run_dir, tensor_shapes, legacy=legacy)
    _write_adapter_config(run_dir, rank=rank)

    report = build_run_status(str(run_dir), allow_unsafe=legacy)
    assert report["export"]["ready"] is expected


@pytest.mark.parametrize("legacy", [False, True], ids=["safetensors", "legacy-bin"])
def test_lora_non_rank_dimensions_must_match_targeted_base_module(tmp_path, legacy):
    run_dir = _make_run_dir(tmp_path)
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "config.json").write_text(json.dumps(TINY_LLAMA_CONFIG))
    shapes = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (1, 1),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (1, 1),
    }
    _write_lora_weights(run_dir, shapes, legacy=legacy)
    _write_lora_config(
        run_dir,
        base_model_name_or_path=str(base_model),
        target_modules=["q_proj"],
    )

    assert build_run_status(str(run_dir), allow_unsafe=legacy)["export"]["ready"] is False


@pytest.mark.parametrize("legacy", [False, True], ids=["safetensors", "legacy-bin"])
def test_lora_weights_must_cover_every_base_module_selected_by_target(tmp_path, legacy):
    run_dir = _make_run_dir(tmp_path)
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    config = dict(TINY_LLAMA_CONFIG, num_hidden_layers=2)
    (base_model / "config.json").write_text(json.dumps(config))
    shapes = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (1, 8),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (8, 1),
    }
    _write_lora_weights(run_dir, shapes, legacy=legacy)
    _write_lora_config(
        run_dir,
        base_model_name_or_path=str(base_model),
        target_modules=["q_proj"],
    )

    assert build_run_status(str(run_dir), allow_unsafe=legacy)["export"]["ready"] is False


def test_genuine_peft_conv1d_lora_uses_transformers_input_output_orientation(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    base_config = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        n_positions=16,
        n_ctx=16,
        vocab_size=16,
        bos_token_id=0,
        eos_token_id=1,
    )
    (base_model / "config.json").write_text(base_config.to_json_string())
    lora_config = LoraConfig(
        r=2,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(base_config, trust_remote_code=False)
        peft_model = get_peft_model(base, lora_config)
    peft_model.peft_config["default"].base_model_name_or_path = str(base_model)
    shapes = {
        key: tuple(value.shape) for key, value in get_peft_model_state_dict(peft_model).items()
    }
    (run_dir / "adapter_model.safetensors").write_bytes(_safetensors_with_shapes(shapes))
    _write_lora_config(
        run_dir,
        base_model_name_or_path=str(base_model),
        r=2,
        target_modules=["c_attn"],
    )

    assert build_run_status(str(run_dir))["export"]["ready"] is True


@pytest.mark.parametrize("rank_pattern", [{"[": 1}, ["layer"]])
def test_invalid_rank_pattern_does_not_crash_run_status(tmp_path, rank_pattern):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    _write_lora_config(run_dir, rank_pattern=rank_pattern)

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_rank_pattern_suffix_overrides_default_lora_rank(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(
        _safetensors_with_shapes(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (2, 8),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (8, 2),
            }
        )
    )
    _write_lora_config(run_dir, rank_pattern={"q_proj": 2})

    assert build_run_status(str(run_dir))["export"]["ready"] is True


def test_lora_weights_must_match_configured_target_modules(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    _write_lora_config(run_dir, target_modules=["v_proj"])

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_lora_weights_must_cover_every_configured_target(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    _write_lora_config(run_dir, target_modules=["q_proj", "v_proj"])

    assert build_run_status(str(run_dir))["export"]["ready"] is False


@pytest.mark.parametrize(
    "target_modules",
    [r".*\.q_proj", r"^.*\.q_proj$", r".*\.layers\..*\.q_proj"],
)
def test_regex_target_modules_match_peft_module_names(tmp_path, target_modules):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    _write_lora_config(run_dir, target_modules=target_modules)

    assert build_run_status(str(run_dir))["export"]["ready"] is True


def test_invalid_regex_target_modules_fail_closed(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    _write_lora_config(run_dir, target_modules="[")

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_catastrophic_target_regex_is_rejected_within_subprocess_deadline():
    script = """
from agoge_forger._run_status_lora import _module_matches_targets, _target_spec

target = _target_spec('(a+)+$')
matched = target is not None and _module_matches_targets('a' * 64 + '!', target)
raise SystemExit(1 if matched else 0)
"""
    result = subprocess.run(  # nosec B603 - fixed interpreter and test-only script
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("alpha", ["bad", True, float("nan"), float("inf"), 10**1000])
def test_lora_alpha_must_be_a_finite_number(tmp_path, alpha):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    _write_lora_config(run_dir, lora_alpha=alpha)

    assert build_run_status(str(run_dir))["export"]["ready"] is False


@pytest.mark.parametrize(
    "alpha_pattern",
    [{"layer": "bad"}, {"layer": float("inf")}, {"[": 1}],
)
def test_lora_alpha_pattern_must_be_usable(tmp_path, alpha_pattern):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    _write_lora_config(run_dir, alpha_pattern=alpha_pattern)

    assert build_run_status(str(run_dir))["export"]["ready"] is False


@pytest.mark.parametrize("revision", ["   ", [], {"branch": "main"}, True])
def test_invalid_adapter_revision_is_not_export_ready(tmp_path, revision):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir, revision=revision)

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_corrupt_safetensors_does_not_fall_back_to_legacy_bin(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(b"corrupt")
    (run_dir / "adapter_model.bin").write_bytes(b"legacy")
    _write_adapter_config(run_dir)

    assert build_run_status(str(run_dir), allow_unsafe=True)["export"]["ready"] is False


@SKIP_IF_ROOT
def test_unreadable_legacy_weights_raise_instead_of_reporting_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_legacy_bin_adapter(run_dir)
    weights = run_dir / "adapter_model.bin"
    _deny_read_access_or_skip(weights)
    try:
        with pytest.raises(OSError):
            build_run_status(str(run_dir), allow_unsafe=True)
    finally:
        os.chmod(weights, 0o644)


# --------------------------------------------------------------------------
