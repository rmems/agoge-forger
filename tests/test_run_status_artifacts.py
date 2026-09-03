"""Merged-model and serialization readiness tests for run-status."""

import json
import os
from pathlib import Path

import pytest
import torch
from test_run_status import (
    SKIP_IF_ROOT,
    _deny_read_access_or_skip,
    _make_run_dir,
    _minimal_safetensors,
    _safetensors_with_dtype,
    _safetensors_with_shapes,
    _safetensors_with_tensors,
    _write_adapter_config,
    _write_checkpoint,
    _write_final_adapter,
    _write_legacy_bin_adapter,
    _write_merged_model,
)
from typer.testing import CliRunner

from agoge_forger.artifacts.safetensors_io import write_artifact_index
from agoge_forger.cli import app
from agoge_forger.run_status import build_run_status, find_merged_model_dir, is_merged_model_dir


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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


@pytest.mark.parametrize(
    ("weight_map", "shard_tensors"),
    [
        (
            {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            },
            {
                "model-00001-of-00002.safetensors": ("a",),
                "model-00002-of-00002.safetensors": ("b",),
            },
        ),
        (
            {
                "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                "model.norm.weight": "model-00001-of-00001.safetensors",
            },
            {
                "model-00001-of-00001.safetensors": (
                    "model.embed_tokens.weight",
                    "model.norm.weight",
                )
            },
        ),
    ],
    ids=["multiple-shards", "shared-shard"],
)
def test_sharded_merged_model_is_recognised(tmp_path, weight_map, shard_tensors):
    """Indexed exports allow distinct shards and tensors sharing a shard."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "tokenizer_config.json").write_text("{}")
    (merged / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard_name, tensor_names in shard_tensors.items():
        (merged / shard_name).write_bytes(_safetensors_with_tensors(*tensor_names))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is True
    assert build_run_status(str(run_dir))["merged_model"] == {
        "present": True,
        "path": str(merged.resolve()),
    }


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


def test_merged_model_requires_final_artifact_index(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "artifact_index.json").unlink(missing_ok=True)

    assert is_merged_model_dir(merged) is False


def test_stale_artifact_index_is_not_a_completion_marker(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").write_bytes(_safetensors_with_tensors("replacement"))

    assert is_merged_model_dir(merged) is False


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
    assert report["base_model"] == "Qwen/Qwen3.5-0.5B"

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


@pytest.mark.parametrize(
    ("shapes", "rank", "expected"),
    [
        (((1,), (1,)), 1, False),
        (((1, 4), (8, 1)), 2, False),
        (((1, 0), (8, 1)), 1, False),
        (((1, 4), (0, 1)), 1, False),
        (((2, 4), (8, 2)), 2, True),
    ],
)
def test_safetensors_lora_shapes_must_match_config_rank(tmp_path, shapes, rank, expected):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(
        _safetensors_with_shapes(
            {
                "base_model.model.layer.lora_A.weight": shapes[0],
                "base_model.model.layer.lora_B.weight": shapes[1],
            }
        )
    )
    _write_adapter_config(run_dir, rank=rank)

    assert build_run_status(str(run_dir))["export"]["ready"] is expected


@pytest.mark.parametrize(
    ("shapes", "rank", "expected"),
    [
        (((1,), (1,)), 1, False),
        (((1, 4), (8, 1)), 2, False),
        (((1, 0), (8, 1)), 1, False),
        (((1, 4), (0, 1)), 1, False),
        (((2, 4), (8, 2)), 2, True),
    ],
)
def test_legacy_lora_shapes_must_match_config_rank(tmp_path, shapes, rank, expected):
    run_dir = _make_run_dir(tmp_path)
    torch.save(
        {
            "base_model.model.layer.lora_A.weight": torch.zeros(shapes[0]),
            "base_model.model.layer.lora_B.weight": torch.zeros(shapes[1]),
        },
        run_dir / "adapter_model.bin",
    )
    _write_adapter_config(run_dir, rank=rank)

    assert build_run_status(str(run_dir), allow_unsafe=True)["export"]["ready"] is expected


@pytest.mark.parametrize("rank_pattern", [{"[": 1}, ["layer"]])
def test_invalid_rank_pattern_does_not_crash_run_status(tmp_path, rank_pattern):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-0.5B",
                "peft_type": "LORA",
                "r": 1,
                "rank_pattern": rank_pattern,
            }
        )
    )

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_rank_pattern_suffix_overrides_default_lora_rank(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(
        _safetensors_with_shapes(
            {
                "base_model.model.layer.lora_A.weight": (2, 4),
                "base_model.model.layer.lora_B.weight": (8, 2),
            }
        )
    )
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-0.5B",
                "peft_type": "LORA",
                "r": 1,
                "rank_pattern": {"layer": 2},
            }
        )
    )

    assert build_run_status(str(run_dir))["export"]["ready"] is True


def test_lora_weights_must_match_configured_target_modules(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-0.5B",
                "peft_type": "LORA",
                "r": 1,
                "target_modules": ["q_proj"],
            }
        )
    )

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_lora_weights_must_cover_every_configured_target(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-0.5B",
                "peft_type": "LORA",
                "r": 1,
                "target_modules": ["layer", "v_proj"],
            }
        )
    )

    assert build_run_status(str(run_dir))["export"]["ready"] is False


@pytest.mark.parametrize("alpha", ["bad", True, float("nan"), float("inf"), 10**1000])
def test_lora_alpha_must_be_a_finite_number(tmp_path, alpha):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-0.5B",
                "peft_type": "LORA",
                "r": 1,
                "lora_alpha": alpha,
            }
        )
    )

    assert build_run_status(str(run_dir))["export"]["ready"] is False


@pytest.mark.parametrize(
    "alpha_pattern",
    [{"layer": "bad"}, {"layer": float("inf")}, {"[": 1}],
)
def test_lora_alpha_pattern_must_be_usable(tmp_path, alpha_pattern):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-0.5B",
                "peft_type": "LORA",
                "r": 1,
                "alpha_pattern": alpha_pattern,
            }
        )
    )

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
