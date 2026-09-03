"""Tests for `agoge run-status` and the `run_status` report builder.

The JSON document is an operator-facing contract intended for tools such as
`jq`, so the schema assertions here are deliberately exact: an added, renamed
or dropped key must break a test rather than silently change automation output.
"""

from __future__ import annotations

import json
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from agoge_forger.artifacts.safetensors_io import write_artifact_index
from agoge_forger.cli import app
from agoge_forger.run_status import (
    SCHEMA_VERSION,
    build_run_status,
)

TOP_LEVEL_KEYS = {
    "schema_version",
    "run_dir",
    "run_name",
    "allow_unsafe_serialization",
    "checkpoints",
    "final_adapter",
    "merged_model",
    "base_model",
    "base_revision",
    "resume",
    "export",
}
CHECKPOINTS_KEYS = {"valid_count", "steps", "latest_step", "latest_path"}
FINAL_ADAPTER_KEYS = {"present", "path"}
MERGED_MODEL_KEYS = {"present", "path"}
RESUME_KEYS = {"ready", "checkpoint_path"}
EXPORT_KEYS = {"ready", "source_path", "source_kind"}
SKIP_IF_ROOT = pytest.mark.skipif(
    getattr(os, "geteuid", lambda: -1)() == 0,
    reason="chmod-based permission denial is ineffective when running as root",
)

TINY_LLAMA_CONFIG = {
    "model_type": "llama",
    "vocab_size": 16,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "max_position_embeddings": 16,
    "tie_word_embeddings": False,
}
TINY_LLAMA_SHAPES = {
    "model.embed_tokens.weight": (16, 8),
    "model.layers.0.self_attn.q_proj.weight": (8, 8),
    "model.layers.0.self_attn.k_proj.weight": (8, 8),
    "model.layers.0.self_attn.v_proj.weight": (8, 8),
    "model.layers.0.self_attn.o_proj.weight": (8, 8),
    "model.layers.0.mlp.gate_proj.weight": (16, 8),
    "model.layers.0.mlp.up_proj.weight": (16, 8),
    "model.layers.0.mlp.down_proj.weight": (8, 16),
    "model.layers.0.input_layernorm.weight": (8,),
    "model.layers.0.post_attention_layernorm.weight": (8,),
    "model.norm.weight": (8,),
    "lm_head.weight": (16, 8),
}
TINY_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {"<unk>": 0, "hello": 1},
        "unk_token": "<unk>",
    },
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _minimal_safetensors() -> bytes:
    """Tiny valid PEFT LoRA safetensors container with one A/B pair."""
    return _safetensors_with_shapes(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (1, 8),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (8, 1),
        }
    )


def _safetensors_with_tensors(*names: str) -> bytes:
    return _safetensors_with_shapes({name: (1,) for name in names})


def _safetensors_with_shapes(
    shapes: dict[str, tuple[int, ...]],
    *,
    dtype: str = "F32",
    element_size: int = 4,
) -> bytes:
    offset = 0
    payload = {}
    for name, shape in shapes.items():
        size = element_size
        for dimension in shape:
            size *= dimension
        payload[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header = json.dumps(payload, separators=(",", ":")).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return len(header).to_bytes(8, "little") + header + b"\0" * offset


def _safetensors_with_dtype(dtype: str) -> bytes:
    payload = {"t": {"dtype": dtype, "shape": [1], "data_offsets": [0, 4]}}
    header = json.dumps(payload, separators=(",", ":")).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return len(header).to_bytes(8, "little") + header + b"\0" * 4


_DEFAULT_BASE_MODEL = object()


def _test_base_model_path(directory: Path) -> Path:
    base_model = directory / ".test-base-model"
    base_model.mkdir(exist_ok=True)
    (base_model / "config.json").write_text(json.dumps(TINY_LLAMA_CONFIG))
    (base_model / "model.safetensors").write_bytes(_safetensors_with_shapes(TINY_LLAMA_SHAPES))
    return base_model.resolve()


def _write_adapter_config(directory, base_model=_DEFAULT_BASE_MODEL, revision=None, rank=1):
    payload = {"peft_type": "LORA", "r": rank, "target_modules": ["q_proj"]}
    if base_model is _DEFAULT_BASE_MODEL:
        base_model = str(_test_base_model_path(directory))
    if base_model is not None:
        payload["base_model_name_or_path"] = base_model
    if revision is not None:
        payload["revision"] = revision
    (directory / "adapter_config.json").write_text(json.dumps(payload))


def _optimizer_state(step=1, shapes=((1, 8), (8, 1))):
    def adamw_group(params, weight_decay):
        return {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": weight_decay,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": True,
            "params": params,
        }

    return {
        "state": {
            parameter_id: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros(shape),
                "exp_avg_sq": torch.zeros(shape),
            }
            for parameter_id, shape in enumerate(shapes)
        },
        "param_groups": [
            adamw_group(list(range(len(shapes))), 0.01),
            adamw_group([], 0.0),
        ],
    }


def _write_torch_state(path, payload=None):
    if payload is None:
        payloads = {
            "optimizer.pt": _optimizer_state(),
            "scheduler.pt": {
                "last_epoch": 0,
                "_step_count": 1,
                "base_lrs": [0.001, 0.001],
                "_last_lr": [0.001, 0.001],
            },
            "rng_state.pth": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
                "cuda": torch.zeros(16, dtype=torch.uint8),
            },
            "adapter_model.bin": {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (
                    torch.zeros(1, 8)
                ),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (
                    torch.zeros(8, 1)
                ),
            },
        }
        payload = payloads[path.name]
    if not isinstance(payload, bytes):
        torch.save(payload, path)
        return
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/data/0", b"\0")
        archive.writestr("archive/version", b"3\n")
        archive.writestr("archive/.data/serialization_id", b"0")


def _write_checkpoint(root, step, base_model=_DEFAULT_BASE_MODEL, revision=None):
    checkpoint_dir = root / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "train_batch_size": 1})
    )
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(_minimal_safetensors())
    _write_torch_state(
        checkpoint_dir / "optimizer.pt",
        _optimizer_state(step),
    )
    _write_torch_state(
        checkpoint_dir / "scheduler.pt",
        {
            "last_epoch": step,
            "_step_count": step + 1,
            "base_lrs": [0.001, 0.001],
            "_last_lr": [0.001, 0.001],
        },
    )
    _write_torch_state(
        checkpoint_dir / "rng_state.pth",
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.zeros(16, dtype=torch.uint8),
        },
    )
    _write_adapter_config(checkpoint_dir, base_model=base_model, revision=revision)
    return checkpoint_dir


def _write_final_adapter(root, base_model=_DEFAULT_BASE_MODEL, revision=None):
    (root / "adapter_model.safetensors").write_bytes(_minimal_safetensors())
    _write_adapter_config(root, base_model=base_model, revision=revision)
    return root


def _write_legacy_bin_adapter(root, base_model=_DEFAULT_BASE_MODEL):
    _write_torch_state(root / "adapter_model.bin")
    _write_adapter_config(root, base_model=base_model)
    return root


def _write_merged_model(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(TINY_LLAMA_CONFIG))
    (path / "model.safetensors").write_bytes(_safetensors_with_shapes(TINY_LLAMA_SHAPES))
    (path / "tokenizer.json").write_text(json.dumps(TINY_TOKENIZER))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "eos_token": "<unk>",
                "model_max_length": 16,
            }
        )
    )
    write_artifact_index(str(path))
    return path


def _make_run_dir(tmp_path, name="demo_run"):
    """Build the conventional `<root>/adapters/<run_name>` run directory."""
    run_dir = tmp_path / "adapters" / name
    run_dir.mkdir(parents=True)
    return run_dir


def _deny_read_access_or_skip(path: Path) -> None:
    """Remove read access, or skip when the runtime bypasses mode bits."""
    os.chmod(path, 0)
    try:
        path.read_bytes()
    except PermissionError:
        return
    os.chmod(path, 0o644)
    pytest.skip("chmod-based permission denial is ineffective for this process")


# --------------------------------------------------------------------------
# 1. Schema stability
# --------------------------------------------------------------------------


def test_report_key_sets_are_exact(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 10)
    _write_final_adapter(run_dir)
    _write_merged_model(tmp_path / "merged" / run_dir.name)

    report = build_run_status(str(run_dir))

    assert set(report) == TOP_LEVEL_KEYS
    assert set(report["checkpoints"]) == CHECKPOINTS_KEYS
    assert set(report["final_adapter"]) == FINAL_ADAPTER_KEYS
    assert set(report["merged_model"]) == MERGED_MODEL_KEYS
    assert set(report["resume"]) == RESUME_KEYS
    assert set(report["export"]) == EXPORT_KEYS


def test_schema_version_is_one(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)

    assert SCHEMA_VERSION == 1
    assert build_run_status(str(run_dir))["schema_version"] == SCHEMA_VERSION == 1


def test_report_survives_json_round_trip(tmp_path):
    """No `Path` objects may leak into the report: it must be `json.dumps`-able."""
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 20)
    _write_final_adapter(run_dir)
    _write_merged_model(tmp_path / "merged" / run_dir.name)

    report = build_run_status(str(run_dir))

    assert json.loads(json.dumps(report)) == report


def test_report_identifies_run_dir_and_name(tmp_path):
    run_dir = _make_run_dir(tmp_path, name="my_run")

    report = build_run_status(str(run_dir))

    assert report["run_dir"] == str(run_dir.resolve())
    assert report["run_name"] == "my_run"


# --------------------------------------------------------------------------
# 2. Empty but inspectable run directory
# --------------------------------------------------------------------------


def test_empty_run_dir_reports_every_key_with_null_values(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    report = build_run_status(str(run_dir))

    assert set(report) == TOP_LEVEL_KEYS
    assert report["checkpoints"]["valid_count"] == 0
    assert report["checkpoints"]["steps"] == []
    assert report["checkpoints"]["latest_step"] is None
    assert report["checkpoints"]["latest_path"] is None
    assert report["final_adapter"] == {"present": False, "path": None}
    assert report["merged_model"] == {"present": False, "path": None}
    assert report["base_model"] is None
    assert report["base_revision"] is None
    assert report["resume"] == {"ready": False, "checkpoint_path": None}
    assert report["export"] == {"ready": False, "source_path": None, "source_kind": None}


def test_empty_run_dir_cli_exits_zero(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)

    result = runner.invoke(app, ["run-status", str(run_dir)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["checkpoints"]["valid_count"] == 0


# --------------------------------------------------------------------------
# 3. Checkpoints only
# --------------------------------------------------------------------------


def test_checkpoints_only_run_is_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    # Written newest-first to prove the report sorts by step, not by mtime.
    latest = _write_checkpoint(run_dir, 100)
    _write_checkpoint(run_dir, 50)

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["valid_count"] == 2
    assert report["checkpoints"]["steps"] == [50, 100]
    assert report["checkpoints"]["latest_step"] == 100
    assert report["checkpoints"]["latest_path"] == str(latest.resolve())
    assert report["resume"] == {"ready": True, "checkpoint_path": str(latest.resolve())}
    assert report["final_adapter"] == {"present": False, "path": None}
    assert report["export"]["ready"] is True
    assert report["export"]["source_kind"] == "checkpoint"
    assert report["export"]["source_path"] == str(latest.resolve())
    assert report["base_model"] == str(latest / ".test-base-model")


def test_latest_checkpoint_is_always_drawn_from_the_reported_steps(tmp_path):
    """`latest_step`/`latest_path` must describe a checkpoint the report lists.

    They are derived from the same single scan as `steps` and `valid_count`, so
    a checkpoint appearing or disappearing between two scans cannot yield a
    report whose latest checkpoint is missing from its own list.
    """
    run_dir = _make_run_dir(tmp_path)
    for step in (25, 50, 100):
        _write_checkpoint(run_dir, step)

    checkpoints = build_run_status(str(run_dir))["checkpoints"]

    assert checkpoints["latest_step"] in checkpoints["steps"]
    assert checkpoints["latest_step"] == checkpoints["steps"][-1]
    assert checkpoints["valid_count"] == len(checkpoints["steps"])
    assert checkpoints["latest_path"].endswith(f"checkpoint-{checkpoints['latest_step']}")


def test_export_source_uses_the_same_checkpoint_snapshot(tmp_path, monkeypatch):
    """A checkpoint arriving mid-report cannot become an unlisted export source."""
    run_dir = _make_run_dir(tmp_path)
    first = _write_checkpoint(run_dir, 50)
    real_resolve_export_source = __import__(
        "agoge_forger.run_status", fromlist=["resolve_export_source_from_snapshot"]
    ).resolve_export_source_from_snapshot

    def create_checkpoint_then_resolve(*args, **kwargs):
        _write_checkpoint(run_dir, 100)
        return real_resolve_export_source(*args, **kwargs)

    monkeypatch.setattr(
        "agoge_forger.run_status.resolve_export_source_from_snapshot",
        create_checkpoint_then_resolve,
    )

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["steps"] == [50]
    assert report["export"]["source_path"] == str(first.resolve())


# --------------------------------------------------------------------------
# 4. Final adapter at the run root
# --------------------------------------------------------------------------


def test_final_adapter_wins_over_checkpoints_as_export_source(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    latest = _write_checkpoint(run_dir, 100)
    _write_final_adapter(run_dir)

    report = build_run_status(str(run_dir))

    assert report["final_adapter"] == {"present": True, "path": str(run_dir.resolve())}
    assert report["export"]["ready"] is True
    assert report["export"]["source_kind"] == "final_adapter"
    assert report["export"]["source_path"] == str(run_dir.resolve())
    # Resume still points at the checkpoint, not the final adapter.
    assert report["resume"]["checkpoint_path"] == str(latest.resolve())


def test_final_adapter_without_checkpoints_is_export_ready_but_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)

    report = build_run_status(str(run_dir))

    assert report["export"]["ready"] is True
    assert report["export"]["source_kind"] == "final_adapter"
    assert report["resume"] == {"ready": False, "checkpoint_path": None}
    assert report["checkpoints"]["valid_count"] == 0
    assert report["checkpoints"]["steps"] == []


# --------------------------------------------------------------------------
# 5. Invalid checkpoints are skipped
# --------------------------------------------------------------------------


def test_invalid_checkpoints_are_not_counted(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 50)
    latest = _write_checkpoint(run_dir, 100)

    # Missing trainer_state.json.
    no_state = run_dir / "checkpoint-75"
    no_state.mkdir()
    (no_state / "adapter_model.safetensors").write_text("weights")
    _write_adapter_config(no_state)

    # Has trainer state but no adapter artifact at all.
    no_adapter = run_dir / "checkpoint-125"
    no_adapter.mkdir()
    (no_adapter / "trainer_state.json").write_text("{}")

    # Not a `checkpoint-N` directory.
    (run_dir / "checkpoint-final").mkdir()

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["valid_count"] == 2
    assert report["checkpoints"]["steps"] == [50, 100]
    assert report["checkpoints"]["latest_step"] == 100
    assert report["resume"]["checkpoint_path"] == str(latest.resolve())


@pytest.mark.parametrize("payload", ["{not json", "", "[]", '"text"', "3", "null"])
def test_malformed_trainer_state_is_not_resume_ready(tmp_path, payload):
    """A present-but-unparseable trainer_state.json is not resume-ready.

    list_valid_checkpoints only checks that the file exists, so train-qlora
    would still select this snapshot; Trainer.train then fails to deserialize it.
    """
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = run_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text(payload)
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(_minimal_safetensors())
    _write_adapter_config(checkpoint_dir)

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["valid_count"] == 1
    assert report["resume"]["ready"] is False
    assert report["resume"]["checkpoint_path"] == str(checkpoint_dir.resolve())
