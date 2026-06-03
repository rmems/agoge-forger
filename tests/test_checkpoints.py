from agoge_forger.export.merge_adapter import export_final_model
from agoge_forger.train.checkpoints import (
    find_latest_valid_checkpoint,
    infer_base_model_from_adapter,
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


def test_find_latest_valid_checkpoint_skips_incomplete_entries(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir, 50)
    broken_checkpoint = run_dir / "checkpoint-75"
    broken_checkpoint.mkdir()
    _write_checkpoint(run_dir, 100)

    latest = find_latest_valid_checkpoint(str(run_dir))

    assert latest == str(run_dir / "checkpoint-100")


def test_resolve_resume_checkpoint_falls_back_when_no_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class Training:
        resume_checkpoint_path = None
        resume_from_latest_checkpoint = True

    class Config:
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
    assert recorded["adapter_path"] == str(latest)
    assert recorded["out_dir"] == str(out_dir)
    assert recorded["kwargs"]["save_safetensors"] is True


def test_export_helpers_support_final_adapter_directory(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_text("weights")
    (adapter_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "Qwen/Qwen3.5-0.5B"}'
    )

    assert resolve_export_source(run_dir=str(adapter_dir)) == str(adapter_dir)
    assert infer_base_model_from_adapter(str(adapter_dir)) == "Qwen/Qwen3.5-0.5B"
