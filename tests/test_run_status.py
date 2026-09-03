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


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _minimal_safetensors() -> bytes:
    """Tiny valid PEFT LoRA safetensors container with one A/B pair."""
    return _safetensors_with_shapes(
        {
            "base_model.model.layer.lora_A.weight": (1, 1),
            "base_model.model.layer.lora_B.weight": (1, 1),
        }
    )


def _safetensors_with_tensors(*names: str) -> bytes:
    return _safetensors_with_shapes({name: (1,) for name in names})


def _safetensors_with_shapes(shapes: dict[str, tuple[int, ...]]) -> bytes:
    offset = 0
    payload = {}
    for name, shape in shapes.items():
        size = 4
        for dimension in shape:
            size *= dimension
        payload[name] = {
            "dtype": "F32",
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


def _header_without_data_region() -> bytes:
    """Parseable header that declares a 4-byte tensor with no data bytes."""
    payload = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    header = json.dumps(payload, separators=(",", ":")).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return len(header).to_bytes(8, "little") + header


def _write_adapter_config(directory, base_model="Qwen/Qwen3.5-0.5B", revision=None, rank=1):
    payload = {"peft_type": "LORA", "r": rank, "target_modules": ["layer"]}
    if base_model is not None:
        payload["base_model_name_or_path"] = base_model
    if revision is not None:
        payload["revision"] = revision
    (directory / "adapter_config.json").write_text(json.dumps(payload))


def _write_torch_state(path, payload=None):
    if payload is None:
        payloads = {
            "optimizer.pt": {"state": {}, "param_groups": [{"params": [0]}]},
            "scheduler.pt": {"last_epoch": 0, "_step_count": 1},
            "rng_state.pth": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
                "cuda": torch.zeros(16, dtype=torch.uint8),
            },
            "adapter_model.bin": {
                "base_model.model.layer.lora_A.weight": torch.zeros(1, 1),
                "base_model.model.layer.lora_B.weight": torch.zeros(1, 1),
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


def _write_checkpoint(root, step, base_model="Qwen/Qwen3.5-0.5B", revision=None):
    checkpoint_dir = root / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "train_batch_size": 1})
    )
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(_minimal_safetensors())
    _write_torch_state(
        checkpoint_dir / "optimizer.pt",
        {"state": {0: {}}, "param_groups": [{"params": [0]}]},
    )
    _write_torch_state(
        checkpoint_dir / "scheduler.pt",
        {"last_epoch": step, "_step_count": step + 1},
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


def _write_final_adapter(root, base_model="Qwen/Qwen3.5-0.5B", revision=None):
    (root / "adapter_model.safetensors").write_bytes(_minimal_safetensors())
    _write_adapter_config(root, base_model=base_model, revision=revision)
    return root


def _write_legacy_bin_adapter(root, base_model="Qwen/Qwen3.5-0.5B"):
    _write_torch_state(root / "adapter_model.bin")
    _write_adapter_config(root, base_model=base_model)
    return root


def _write_merged_model(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type": "llama"}')
    (path / "model.safetensors").write_bytes(_minimal_safetensors())
    (path / "tokenizer_config.json").write_text("{}")
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
    latest = _write_checkpoint(run_dir, 100, base_model="Qwen/Qwen3.5-0.5B")
    _write_checkpoint(run_dir, 50, base_model="Qwen/Qwen3.5-0.5B")

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
    assert report["base_model"] == "Qwen/Qwen3.5-0.5B"


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


def test_empty_adapter_weights_are_not_export_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(b"")
    _write_adapter_config(run_dir)

    report = build_run_status(str(run_dir))

    assert report["final_adapter"]["present"] is True
    assert report["export"]["ready"] is False
    assert report["export"]["source_kind"] == "final_adapter"
    assert report["export"]["source_path"] == str(run_dir.resolve())


def test_truncated_adapter_weights_are_not_export_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    # Short junk, and an 8-byte length that claims more header than exists.
    (run_dir / "adapter_model.safetensors").write_bytes(b"trunc")
    _write_adapter_config(run_dir)
    assert build_run_status(str(run_dir))["export"]["ready"] is False

    (run_dir / "adapter_model.safetensors").write_bytes((64).to_bytes(8, "little") + b"{")
    report = build_run_status(str(run_dir))
    assert report["final_adapter"]["present"] is True
    assert report["export"]["ready"] is False


def test_zero_tensor_safetensors_is_not_export_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    header = b"{}" + b" " * 7
    (run_dir / "adapter_model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    _write_adapter_config(run_dir)

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_header_without_data_region_is_not_export_ready(tmp_path):
    """A padded JSON header that claims tensor bytes the file does not have."""
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(_header_without_data_region())
    _write_adapter_config(run_dir)

    report = build_run_status(str(run_dir))

    assert report["final_adapter"]["present"] is True
    assert report["export"]["ready"] is False


def test_unsupported_safetensors_dtype_is_not_export_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(
        _safetensors_with_dtype("NOT_A_SAFETENSORS_DTYPE")
    )
    _write_adapter_config(run_dir)

    assert build_run_status(str(run_dir))["export"]["ready"] is False


def test_empty_checkpoint_weights_are_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = run_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"")
    _write_adapter_config(checkpoint_dir)

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["valid_count"] == 1
    assert report["resume"]["ready"] is False
    assert report["resume"]["checkpoint_path"] == str(checkpoint_dir.resolve())


def test_header_without_data_region_is_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = run_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(_header_without_data_region())
    _write_adapter_config(checkpoint_dir)

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["valid_count"] == 1
    assert report["resume"]["ready"] is False


@pytest.mark.parametrize("missing_name", ["optimizer.pt", "scheduler.pt"])
def test_missing_training_state_is_not_resume_ready(tmp_path, missing_name):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    missing = checkpoint_dir / missing_name
    if missing.exists():
        missing.unlink()

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize("state_name", ["optimizer.pt", "scheduler.pt"])
def test_corrupt_training_state_is_not_resume_ready(tmp_path, state_name):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / state_name).write_bytes(b"not a torch zip")

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_non_torch_zip_state_is_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    with zipfile.ZipFile(checkpoint_dir / "optimizer.pt", "w") as archive:
        archive.writestr("unrelated", b"not torch serialization")

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize("state_name", ["optimizer.pt", "scheduler.pt", "rng_state.pth"])
def test_torch_zip_with_invalid_pickle_is_not_resume_ready(tmp_path, state_name):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    _write_torch_state(checkpoint_dir / state_name, payload=b"not a pickle")

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize("state_name", ["optimizer.pt", "scheduler.pt", "rng_state.pth"])
def test_torch_zip_with_empty_state_is_not_resume_ready(tmp_path, state_name):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    _write_torch_state(checkpoint_dir / state_name, payload=b"\x80\x02}q\x00.")

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_nested_optimizer_field_names_are_not_top_level_state(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    nested = (
        b"\x80\x02}q\x00X\x04\x00\x00\x00junkq\x01]q\x02("
        b"X\x05\x00\x00\x00stateq\x03X\x0c\x00\x00\x00param_groupsq\x04es."
    )
    _write_torch_state(checkpoint_dir / "optimizer.pt", payload=nested)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize(
    ("state_name", "payload"),
    [
        ("optimizer.pt", {"state": None, "param_groups": [{"params": []}]}),
        ("optimizer.pt", {"state": {}, "param_groups": [{"params": None}]}),
        ("optimizer.pt", {"state": {}, "param_groups": [{"params": [0]}]}),
        ("optimizer.pt", {"state": {0: {}}, "param_groups": [{"params": [0, 1]}]}),
        ("optimizer.pt", {"state": {0: {}, 1: {}}, "param_groups": [{"params": [0]}]}),
        ("optimizer.pt", {"state": {0: {}}, "param_groups": [{"params": [0, 0]}]}),
        ("scheduler.pt", {"last_epoch": None, "_step_count": 1}),
        ("scheduler.pt", {"last_epoch": 0, "_step_count": "1"}),
        ("scheduler.pt", {"last_epoch": 50.0, "_step_count": 51}),
        ("scheduler.pt", {"last_epoch": 50, "_step_count": 51.0}),
        ("scheduler.pt", {"last_epoch": 0, "_step_count": 1}),
        ("scheduler.pt", {"last_epoch": 50, "_step_count": 50}),
    ],
)
def test_malformed_training_state_values_are_not_resume_ready(tmp_path, state_name, payload):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    _write_torch_state(checkpoint_dir / state_name, payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize("field", ["python", "numpy", "cpu", "cuda"])
def test_malformed_rng_state_values_are_not_resume_ready(tmp_path, field):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.zeros(16, dtype=torch.uint8),
    }
    payload[field] = None
    _write_torch_state(checkpoint_dir / "rng_state.pth", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize(
    "cuda_state",
    [
        torch.zeros(1, dtype=torch.uint8),
        torch.zeros(16),
        torch.tensor([0] * 8 + [1] + [0] * 7, dtype=torch.uint8),
    ],
)
def test_malformed_cuda_rng_state_is_not_resume_ready(tmp_path, cuda_state):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    _write_torch_state(
        checkpoint_dir / "rng_state.pth",
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": cuda_state,
        },
    )

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_non_lora_checkpoint_config_is_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-0.5B",
                "peft_type": "IA3",
                "r": 1,
            }
        )
    )

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_trainer_state_step_must_match_checkpoint_name(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / "trainer_state.json").write_text('{"global_step": 49}')

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_trainer_state_requires_positive_batch_size(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / "trainer_state.json").write_text('{"global_step": 50}')

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_lone_ranked_rng_state_is_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / "rng_state.pth").rename(checkpoint_dir / "rng_state_0.pth")

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_missing_rng_state_is_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / "rng_state.pth").unlink(missing_ok=True)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


# --------------------------------------------------------------------------
