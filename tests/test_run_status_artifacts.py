"""Merged-model and serialization readiness tests for run-status."""

import json
import os
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from test_run_status import (
    SKIP_IF_ROOT,
    TINY_LLAMA_CONFIG,
    TINY_LLAMA_SHAPES,
    TINY_TOKENIZER,
    _deny_read_access_or_skip,
    _make_run_dir,
    _minimal_safetensors,
    _safetensors_with_dtype,
    _safetensors_with_shapes,
    _safetensors_with_tensors,
    _test_base_model_path,
    _write_adapter_config,
    _write_checkpoint,
    _write_final_adapter,
    _write_legacy_bin_adapter,
    _write_merged_model,
)
from transformers import AutoModelForCausalLM, GPT2Config
from typer.testing import CliRunner

from agoge_forger.artifacts.safetensors_io import write_artifact_index
from agoge_forger.cli import app
from agoge_forger.run_status import build_run_status, find_merged_model_dir, is_merged_model_dir


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


# 6. Merged model discovery
# --------------------------------------------------------------------------


def test_merged_model_found_in_conventional_sibling_layout(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)

    report = build_run_status(str(run_dir))

    assert report["merged_model"] == {"present": True, "path": str(merged.resolve())}
    assert find_merged_model_dir(run_dir.resolve()) == merged.resolve()


def test_conventional_merged_path_is_absolute_from_relative_run_dir(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    monkeypatch.chdir(tmp_path)

    report = build_run_status(f"adapters/{run_dir.name}")

    path = report["merged_model"]["path"]
    assert report["merged_model"]["present"] is True
    assert Path(path).is_absolute()
    assert path == str(merged.resolve())


def test_merged_model_absent_when_never_exported(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)

    report = build_run_status(str(run_dir))

    assert report["merged_model"] == {"present": False, "path": None}
    assert find_merged_model_dir(run_dir.resolve()) is None


def test_merged_dir_without_config_json_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_text("merged-weights")

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


@pytest.mark.parametrize("weight_name", [None, "adapter_model.safetensors"])
def test_non_model_weights_are_not_a_merged_model(tmp_path, weight_name):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "tokenizer_config.json").write_text("{}")
    if weight_name is not None:
        (merged / weight_name).write_bytes(_minimal_safetensors())

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_merged_dir_with_only_nested_safetensors_is_not_a_merged_model(tmp_path):
    """A tree holding adapters further down is not an exported merged model.

    A merged model keeps its weights at the directory root; a run tree that only
    contains `checkpoint-N/adapter_model.safetensors` must not read as already
    merged, or an operator would skip an export that never happened.
    """
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    _write_checkpoint(merged, 10)
    (merged / "config.json").write_text('{"model_type": "llama"}')

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


@pytest.mark.parametrize("shard_count", [1, 2], ids=["shared-shard", "multiple-shards"])
def test_sharded_merged_model_is_recognised(tmp_path, shard_count):
    """Indexed exports allow distinct shards and tensors sharing a shard."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text(json.dumps(TINY_LLAMA_CONFIG))
    (merged / "tokenizer.json").write_text(json.dumps(TINY_TOKENIZER))
    (merged / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>"})
    )
    shard_names = [
        f"model-{ordinal:05d}-of-{shard_count:05d}.safetensors"
        for ordinal in range(1, shard_count + 1)
    ]
    shard_tensors = {name: {} for name in shard_names}
    weight_map = {}
    for index, (tensor_name, shape) in enumerate(TINY_LLAMA_SHAPES.items()):
        shard_name = shard_names[index % shard_count]
        shard_tensors[shard_name][tensor_name] = shape
        weight_map[tensor_name] = shard_name
    (merged / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard_name, tensor_shapes in shard_tensors.items():
        (merged / shard_name).write_bytes(_safetensors_with_shapes(tensor_shapes))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is True
    assert build_run_status(str(run_dir))["merged_model"] == {
        "present": True,
        "path": str(merged.resolve()),
    }


def test_deeply_nested_shard_index_fails_closed(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "model.safetensors").rename(merged / "model-00001-of-00001.safetensors")
    nested = '{"weight_map":' + "[" * 100_000 + "{}" + "]" * 100_000 + "}"
    (merged / "model.safetensors.index.json").write_text(nested)
    write_artifact_index(str(merged))

    result = runner.invoke(app, ["run-status", str(run_dir)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {"present": False, "path": None}


def test_shard_index_tensor_must_exist_in_designated_shard(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "tokenizer_config.json").write_text("{}")
    shard_name = "model-00001-of-00001.safetensors"
    (merged / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"expected": shard_name}})
    )
    (merged / shard_name).write_bytes(_safetensors_with_dtype("F32"))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_shard_index_must_cover_every_serialized_tensor(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").unlink()
    shard_name = "model-00001-of-00001.safetensors"
    (merged / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shard_name}})
    )
    (merged / shard_name).write_bytes(_safetensors_with_tensors("a", "b"))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_shard_index_must_cover_every_physical_shard(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").unlink()
    indexed = "model-00001-of-00001.safetensors"
    extra = "model-00002-of-00002.safetensors"
    (merged / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": indexed}}))
    (merged / indexed).write_bytes(_safetensors_with_tensors("a"))
    (merged / extra).write_bytes(_safetensors_with_tensors("b"))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_numbered_shard_series_must_be_complete(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").unlink()
    shard_name = "model-00001-of-00002.safetensors"
    (merged / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard_name}})
    )
    (merged / shard_name).write_bytes(_safetensors_with_tensors("weight"))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_absurd_shard_total_is_rejected_without_expanding_range(monkeypatch):
    from agoge_forger import _run_status_safetensors as module

    def fail_range(*args):
        raise AssertionError("must not materialize the attacker-controlled total")

    monkeypatch.setattr(module, "range", fail_range, raising=False)

    assert module._numbered_shards_complete({"model-00001-of-999999999.safetensors"}) is False


def test_truncated_merged_config_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{not json")
    (merged / "model.safetensors").write_text("merged-weights")

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}

    (merged / "model.safetensors").write_bytes(b"")
    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_incomplete_shard_set_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (merged / "model-00001-of-00002.safetensors").write_text("shard-1")

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_shard_index_rejects_adapter_filename(tmp_path):
    """weight_map must name root-local model shards, not leftover adapter files."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "adapter_model.safetensors").write_bytes(_minimal_safetensors())
    (merged / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "adapter_model.safetensors"}})
    )

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_shard_index_rejects_out_of_directory_filename(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "model.safetensors").write_bytes(_minimal_safetensors())
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    rel = os.path.relpath(elsewhere / "model.safetensors", merged)
    (merged / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": rel}}))

    assert "/" in rel or rel.startswith("..")
    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_truncated_merged_weights_are_not_present(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "model.safetensors").write_text("merged-weights")

    assert is_merged_model_dir(merged) is False


def test_merged_model_requires_completed_tokenizer_export(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "model.safetensors").write_bytes(_minimal_safetensors())

    assert is_merged_model_dir(merged) is False


def test_merged_model_requires_usable_tokenizer_inventory(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "tokenizer.json").unlink()
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_model_requires_final_artifact_index(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "artifact_index.json").unlink(missing_ok=True)

    assert is_merged_model_dir(merged) is False


def test_stale_artifact_index_is_not_a_completion_marker(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").write_bytes(_safetensors_with_tensors("replacement"))

    assert is_merged_model_dir(merged) is False


def test_deeply_nested_artifact_index_fails_closed(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    nested = '{"artifacts":' + "[" * 100_000 + "{}" + "]" * 100_000 + "}"
    (merged / "artifact_index.json").write_text(nested)

    result = runner.invoke(app, ["run-status", str(run_dir)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {"present": False, "path": None}


def test_merged_config_requires_model_type(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "config.json").write_text("{}")
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_config_requires_recognized_local_model_type(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "config.json").write_text('{"model_type": "not-a-real-model"}')
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_config_requires_causal_lm_model_type(tmp_path):
    from agoge_forger._run_status_validation import _local_causal_lm_config

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type": "vit"}')

    assert _local_causal_lm_config(merged) is None


def test_merged_weights_must_match_complete_local_architecture(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").write_bytes(
        _safetensors_with_shapes({"unrelated.weight": (8, 8)})
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_weight_shapes_must_match_local_architecture(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    wrong_shapes = dict(TINY_LLAMA_SHAPES)
    wrong_shapes["model.norm.weight"] = (7,)
    (merged / "model.safetensors").write_bytes(_safetensors_with_shapes(wrong_shapes))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_unsharded_model_rejects_simultaneous_shard_index(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_unsharded_model_rejects_stale_numbered_shard(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model-00001-of-00001.safetensors").write_bytes(
        _safetensors_with_shapes(TINY_LLAMA_SHAPES)
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_unsharded_model_rejects_extra_root_model_safetensors(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "consolidated.safetensors").write_bytes(
        _safetensors_with_shapes({"foreign.weight": (1,)})
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


@pytest.mark.parametrize(
    ("model_type", "multiplicity_key"),
    [("llama", "num_hidden_layers"), ("bart", "decoder_layers")],
)
def test_huge_architecture_is_rejected_before_meta_model_construction(
    tmp_path, monkeypatch, model_type, multiplicity_key
):
    from agoge_forger import _run_status_validation as validation

    merged = tmp_path / "merged"
    merged.mkdir()
    config_payload = (
        dict(TINY_LLAMA_CONFIG) if model_type == "llama" else {"model_type": model_type}
    )
    config_payload[multiplicity_key] = 1_000_000_000
    (merged / "config.json").write_text(json.dumps(config_payload))
    config = validation._local_causal_lm_config(merged)

    def fail_construction(*args, **kwargs):
        raise AssertionError("untrusted dimensions must be bounded before model construction")

    monkeypatch.setattr(validation.AutoModelForCausalLM, "from_config", fail_construction)

    assert validation._causal_lm_shapes(config) is None


def test_multiplicative_architecture_is_rejected_before_meta_model_construction(
    tmp_path, monkeypatch
):
    from agoge_forger import _run_status_validation as validation

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mixtral",
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 4_096,
                "num_local_experts": 4_096,
            }
        )
    )
    config = validation._local_causal_lm_config(merged)

    def fail_construction(*args, **kwargs):
        raise AssertionError("combined module count must be bounded before construction")

    monkeypatch.setattr(validation.AutoModelForCausalLM, "from_config", fail_construction)

    assert validation._causal_lm_shapes(config) is None


def test_symlinked_artifact_index_is_not_a_completion_marker(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    index = merged / "artifact_index.json"
    external = tmp_path / "external-index.json"
    index.replace(external)
    index.symlink_to(external)

    assert is_merged_model_dir(merged) is False


def test_symlinked_merged_weights_are_not_standalone(tmp_path):
    external = tmp_path / "external.safetensors"
    external.write_bytes(_minimal_safetensors())
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").unlink()
    (merged / "model.safetensors").symlink_to(external)
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_unsupported_merged_safetensors_dtype_is_not_present(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "tokenizer_config.json").write_text("{}")
    (merged / "model.safetensors").write_bytes(_safetensors_with_dtype("NOT_A_SAFETENSORS_DTYPE"))

    assert is_merged_model_dir(merged) is False


def test_explicit_merged_dir_is_honored(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    elsewhere = _write_merged_model(tmp_path / "elsewhere" / "custom_merge")

    report = build_run_status(str(run_dir), merged_dir=str(elsewhere))
    assert report["merged_model"] == {"present": True, "path": str(elsewhere.resolve())}

    result = runner.invoke(app, ["run-status", str(run_dir), "--merged-dir", str(elsewhere)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {
        "present": True,
        "path": str(elsewhere.resolve()),
    }


def test_explicit_missing_merged_dir_reports_absent_and_exits_zero(runner, tmp_path):
    """A not-yet-exported merged model is an answer, not an error: still exit 0."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    missing = tmp_path / "elsewhere" / "not_exported_yet"

    report = build_run_status(str(run_dir), merged_dir=str(missing))
    assert report["merged_model"] == {"present": False, "path": None}

    result = runner.invoke(app, ["run-status", str(run_dir), "--merged-dir", str(missing)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {"present": False, "path": None}


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

    assert build_run_status(str(run_dir))["export"]["ready"] is False


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
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(shapes[0]),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(shapes[1]),
    }
    if legacy:
        torch.save(tensors, run_dir / "adapter_model.bin")
    else:
        safetensor_shapes = {key: tuple(tensor.shape) for key, tensor in tensors.items()}
        (run_dir / "adapter_model.safetensors").write_bytes(
            _safetensors_with_shapes(safetensor_shapes)
        )
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
    if legacy:
        torch.save(
            {key: torch.zeros(shape) for key, shape in shapes.items()},
            run_dir / "adapter_model.bin",
        )
    else:
        (run_dir / "adapter_model.safetensors").write_bytes(_safetensors_with_shapes(shapes))
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
    if legacy:
        torch.save(
            {key: torch.zeros(shape) for key, shape in shapes.items()},
            run_dir / "adapter_model.bin",
        )
    else:
        (run_dir / "adapter_model.safetensors").write_bytes(_safetensors_with_shapes(shapes))
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
