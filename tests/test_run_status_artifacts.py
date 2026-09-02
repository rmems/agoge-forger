"""Merged-model and serialization readiness tests for run-status."""

import json
import os
from pathlib import Path

import pytest
from test_run_status import (
    SKIP_IF_ROOT,
    _deny_read_access_or_skip,
    _make_run_dir,
    _minimal_safetensors,
    _safetensors_with_dtype,
    _write_adapter_config,
    _write_checkpoint,
    _write_final_adapter,
    _write_legacy_bin_adapter,
    _write_merged_model,
)
from typer.testing import CliRunner

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


def test_merged_dir_without_safetensors_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "tokenizer_config.json").write_text("{}")

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


def test_sharded_merged_model_is_recognised(tmp_path):
    """`save_pretrained` shards keep every weight file at the root."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "tokenizer_config.json").write_text("{}")
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
    (merged / "model-00001-of-00002.safetensors").write_bytes(_minimal_safetensors())
    (merged / "model-00002-of-00002.safetensors").write_bytes(_minimal_safetensors())

    assert is_merged_model_dir(merged) is True
    assert build_run_status(str(run_dir))["merged_model"] == {
        "present": True,
        "path": str(merged.resolve()),
    }


def test_shared_shard_filenames_are_recognised(tmp_path):
    """Transformers weight_map has one entry per tensor; many share a shard."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "tokenizer_config.json").write_text("{}")
    (merged / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                    "model.norm.weight": "model-00001-of-00001.safetensors",
                }
            }
        )
    )
    (merged / "model-00001-of-00001.safetensors").write_bytes(_minimal_safetensors())

    assert is_merged_model_dir(merged) is True
    assert build_run_status(str(run_dir))["merged_model"] == {
        "present": True,
        "path": str(merged.resolve()),
    }


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


def test_root_adapter_safetensors_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "adapter_model.safetensors").write_text("adapter-weights")

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
