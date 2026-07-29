import pytest

from agoge_forger.export.merge_adapter import export_final_model
from agoge_forger.train.checkpoints import (
    find_latest_valid_checkpoint,
    infer_base_model_from_adapter,
    is_adapter_artifact,
    is_valid_checkpoint,
    list_valid_checkpoints,
    resolve_export_source,
    resolve_resume_checkpoint,
)


def _write_checkpoint(root, step, base_model="Qwen/Qwen3.5-0.5B"):
    checkpoint_dir = root / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    (checkpoint_dir / "adapter_model.safetensors").write_text("weights")
    (checkpoint_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "%s"}' % base_model
    )
    return checkpoint_dir


def _write_final_adapter(root, base_model="Qwen/Qwen3.5-0.5B"):
    (root / "adapter_model.safetensors").write_text("final-weights")
    (root / "adapter_config.json").write_text('{"base_model_name_or_path": "%s"}' % base_model)


def test_find_latest_valid_checkpoint_skips_incomplete_entries(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 50)
    broken_checkpoint = run_dir / "checkpoint-75"
    broken_checkpoint.mkdir()
    _write_checkpoint(run_dir, 100)

    latest = find_latest_valid_checkpoint(run_dir)

    assert latest == (run_dir / "checkpoint-100").resolve()


def _write_legacy_bin_checkpoint(root, step, base_model="Qwen/Qwen3.5-0.5B"):
    checkpoint_dir = root / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    (checkpoint_dir / "adapter_model.bin").write_text("legacy")
    (checkpoint_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "%s"}' % base_model
    )
    return checkpoint_dir


def test_resolve_resume_checkpoint_honors_allow_unsafe_opt_in(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    legacy = _write_legacy_bin_checkpoint(run_dir, 10)

    class Runtime:
        allow_unsafe_serialization = True

    class Training:
        resume_checkpoint_path = None
        resume_from_latest_checkpoint = True

    class Config:
        runtime = Runtime()
        training = Training()

    assert resolve_resume_checkpoint(str(run_dir), Config()) == str(legacy.resolve())


def test_resolve_resume_checkpoint_falls_back_when_no_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class Runtime:
        allow_unsafe_serialization = False

    class Training:
        resume_checkpoint_path = None
        resume_from_latest_checkpoint = True

    class Config:
        runtime = Runtime()
        training = Training()

    assert resolve_resume_checkpoint(str(run_dir), Config()) is None


def test_export_final_model_uses_latest_valid_checkpoint(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 10)
    latest = _write_checkpoint(run_dir, 20)
    out_dir = tmp_path / "merged"

    recorded = {}

    def fake_merge_adapter(base_model_id, adapter_path, out_dir_arg, **kwargs):
        recorded["base_model_id"] = base_model_id
        recorded["adapter_path"] = adapter_path
        recorded["out_dir"] = out_dir_arg
        recorded["kwargs"] = kwargs

    monkeypatch.setattr("agoge_forger.export.merge_adapter.merge_adapter", fake_merge_adapter)

    export_final_model(run_dir=str(run_dir), out_dir=str(out_dir))

    assert recorded["base_model_id"] == "Qwen/Qwen3.5-0.5B"
    assert recorded["adapter_path"] == str(latest.resolve())
    assert recorded["out_dir"] == str(out_dir.resolve())
    assert recorded["kwargs"]["save_safetensors"] is True


def test_resolve_export_source_prefers_final_adapter_over_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 50)
    _write_checkpoint(run_dir, 100)
    _write_final_adapter(run_dir)

    assert resolve_export_source(run_dir=str(run_dir)) == str(run_dir.resolve())


def test_adapter_artifact_rejects_bin_weights(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "test-model"}')
    (adapter_dir / "adapter_model.bin").write_text("unsafe")

    assert is_adapter_artifact(adapter_dir) is False


def test_adapter_artifact_rejects_mixed_safetensors_and_bin(tmp_path):
    """A directory containing BOTH safetensors and a pickle-based .bin
    weight file is not a valid adapter artifact under the safetensors-only
    policy, even though safetensors is present.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "test-model"}')
    (adapter_dir / "adapter_model.safetensors").write_text("weights")
    (adapter_dir / "adapter_model.bin").write_text("unsafe")

    assert is_adapter_artifact(adapter_dir) is False


def test_resolve_export_source_accepts_legacy_bin_with_opt_in(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "test-model"}')
    (adapter_dir / "adapter_model.bin").write_text("legacy")

    with pytest.raises(ValueError, match="not a valid adapter artifact"):
        resolve_export_source(adapter_path=str(adapter_dir))

    assert resolve_export_source(adapter_path=str(adapter_dir), allow_unsafe=True) == str(
        adapter_dir.resolve()
    )


def test_export_helpers_support_final_adapter_directory(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_text("weights")
    (adapter_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "Qwen/Qwen3.5-0.5B"}'
    )

    resolved = resolve_export_source(run_dir=str(adapter_dir))
    assert resolved == str(adapter_dir.resolve())
    assert infer_base_model_from_adapter(adapter_dir) == "Qwen/Qwen3.5-0.5B"


def test_latest_valid_checkpoint_skips_unsafe_bin_only_checkpoint(tmp_path):
    """If the newest checkpoint contains an unsafe .bin, the resolver must
    fall back to the latest safetensors-clean checkpoint instead of
    hard-failing the resume/export flow.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    safe = _write_checkpoint(run_dir, 10)
    unsafe = _write_checkpoint(run_dir, 20)
    (unsafe / "adapter_model.bin").write_text("unsafe")

    latest = find_latest_valid_checkpoint(run_dir)
    assert latest == safe.resolve()

    # list_valid_checkpoints must also exclude the unsafe entry.
    assert list_valid_checkpoints(run_dir) == [safe.resolve()]

    # is_valid_checkpoint must report False for the unsafe entry even
    # though it has the trainer state and a safetensors file.
    assert is_valid_checkpoint(unsafe) is False
    assert is_valid_checkpoint(safe) is True


def test_is_valid_checkpoint_accepts_path_or_str(tmp_path):
    """The new Path-based helper signature should accept both str and Path."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = _write_checkpoint(run_dir, 5)
    assert is_valid_checkpoint(checkpoint) is True
    assert is_valid_checkpoint(str(checkpoint)) is True
    assert is_valid_checkpoint(run_dir / "missing") is False
