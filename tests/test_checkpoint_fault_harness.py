"""CPU fault-injection coverage for single-process checkpoint resume."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from agoge_forger._run_status_torch_archive import torch_mapping
from agoge_forger.cleanup_run import CleanupOptions, plan_cleanup
from agoge_forger.train.checkpoints import (
    ADAPTER_WEIGHT_FILES,
    CHECKPOINT_STAGING_PREFIX,
    QUARANTINE_REASON_FILENAME,
    find_latest_valid_checkpoint,
    is_adapter_artifact,
    is_valid_checkpoint,
    list_quarantined_checkpoints,
    list_valid_checkpoints,
    quarantine_incomplete_checkpoints,
    quarantine_tree,
    read_quarantine_reason,
    resolve_resume_checkpoint,
)
from agoge_forger.train.fault_harness import (
    FaultKind,
    FaultPoint,
    FaultSpec,
    HarnessConfig,
    last_failure_evidence,
    run_fault_harness,
)
from agoge_forger.train.resume_state import (
    RESUME_SIDECAR_FILENAME,
    inspect_resume_state,
    select_resume_checkpoint,
)


def _config(run_dir: Path, **kwargs) -> HarnessConfig:
    return HarnessConfig(run_dir=run_dir, **kwargs)


def _checkpoint_names(run_dir: Path) -> list[str]:
    return sorted(entry.name for entry in run_dir.iterdir() if entry.name.startswith("checkpoint-"))


def _staging_names(run_dir: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in run_dir.iterdir()
        if entry.name.startswith((".agoge-ckpt-staging-", ".agoge-export-staging-"))
    )


def _reasons(run_dir: Path) -> set[str]:
    reasons = set()
    for path in list_quarantined_checkpoints(run_dir):
        try:
            reasons.add(read_quarantine_reason(path)["reason"])
        except (OSError, TypeError, ValueError):
            continue
    return reasons


def _trainer_rng_fingerprint() -> tuple[object, ...]:
    """Compare restored trainer streams without drawing from ``random``/legacy NumPy."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().detach().cpu().tolist()
    return (
        python_state,
        numpy_state[0],
        bytes(numpy_state[1]),
        int(numpy_state[2]),
        tuple(numpy_state[3:]),
        torch_state,
    )


def test_fault_harness_completes_without_downloads_or_gpu(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = run_fault_harness(_config(run_dir))

    assert result.completed is True
    assert result.published_steps == (2, 4)
    assert result.export_published is True
    assert result.resume.equivalent is True
    assert result.resume.global_step == 4
    assert result.resume.sampler_position == 4
    assert set(result.resume.restored) == {
        "global_step",
        "optimizer",
        "scheduler",
        "sampler_position",
        "rng",
        "adapter_weights",
    }
    assert is_adapter_artifact(run_dir) is True
    assert "leftover_staging" not in _reasons(run_dir)
    assert _staging_names(run_dir) == []


@pytest.mark.parametrize(
    ("point", "kind"),
    [
        ("before_save", "oom"),
        ("before_save", "sigint"),
        ("before_save", "sigterm"),
    ],
)
def test_failure_before_save_resumes_last_complete_checkpoint(
    tmp_path: Path, point: FaultPoint, kind: FaultKind
) -> None:
    run_dir = tmp_path / "run"
    result = run_fault_harness(_config(run_dir, fault=FaultSpec(step=4, point=point, kind=kind)))

    assert result.completed is False
    assert result.published_steps == (2,)
    assert result.export_published is False
    assert _checkpoint_names(run_dir) == ["checkpoint-2"]
    assert _staging_names(run_dir) == []
    assert result.resume.checkpoint == (run_dir / "checkpoint-2").resolve()
    assert result.resume.equivalent is True
    assert result.resume.global_step == 2
    evidence = last_failure_evidence(run_dir)
    assert evidence is not None
    assert evidence["last_complete_step"] == 2
    assert evidence["resume_equivalent"] is True


def test_short_write_is_quarantined_and_does_not_become_the_resume_point(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = run_fault_harness(
        _config(run_dir, fault=FaultSpec(step=4, point="during_write", kind="short_write"))
    )

    assert result.completed is False
    assert result.published_steps == (2,)
    assert find_latest_valid_checkpoint(run_dir) == (run_dir / "checkpoint-2").resolve()
    assert not (run_dir / "checkpoint-4").exists()
    assert _staging_names(run_dir) == []
    assert "short_write" in _reasons(run_dir)
    assert result.resume.equivalent is True
    assert result.resume.global_step == 2


def test_rename_failure_rolls_back_atomic_publish(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = run_fault_harness(
        _config(run_dir, fault=FaultSpec(step=4, point="during_rename", kind="rename_failure"))
    )

    assert result.completed is False
    assert result.published_steps == (2,)
    assert not (run_dir / "checkpoint-4").exists()
    assert _staging_names(run_dir) == []
    assert result.resume.checkpoint == (run_dir / "checkpoint-2").resolve()
    assert {"interrupted_save", "leftover_staging"} & _reasons(run_dir)


@pytest.mark.parametrize(
    ("kind", "error_type"),
    [("oom", "InjectedOOMError"), ("sigint", "InjectedInterrupt")],
)
def test_failure_after_save_selects_the_new_complete_checkpoint(
    tmp_path: Path, kind: FaultKind, error_type: str
) -> None:
    run_dir = tmp_path / "run"
    result = run_fault_harness(
        _config(run_dir, fault=FaultSpec(step=4, point="after_save", kind=kind))
    )

    assert result.completed is False
    assert result.error_type == error_type
    assert result.published_steps == (2, 4)
    assert result.export_published is False
    assert result.resume.global_step == 4
    assert result.resume.equivalent is True
    assert is_adapter_artifact(run_dir) is False


def test_export_failure_preserves_checkpoints_and_skips_run_root_adapter(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = run_fault_harness(
        _config(
            run_dir,
            fault=FaultSpec(step=4, point="during_export", kind="export_failure"),
        )
    )

    assert result.completed is False
    assert result.error_type == "InjectedExportError"
    assert result.published_steps == (2, 4)
    assert result.export_published is False
    assert is_adapter_artifact(run_dir) is False
    assert not (run_dir / "adapter_model.safetensors").exists()
    assert result.resume.equivalent is True
    assert result.resume.global_step == 4


def test_interrupted_run_resumes_and_completes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    interrupted = run_fault_harness(
        _config(run_dir, fault=FaultSpec(step=2, point="after_save", kind="sigterm"))
    )
    resumed = run_fault_harness(_config(run_dir))

    assert interrupted.published_steps == (2,)
    assert interrupted.resume.equivalent is True
    assert resumed.completed is True
    assert resumed.published_steps == (2, 4)
    assert resumed.resume.global_step == 4
    assert resumed.resume.sampler_position == 4


def test_rng_and_sampler_restore_from_the_last_complete_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = run_fault_harness(
        _config(run_dir, fault=FaultSpec(step=2, point="after_save", kind="oom"))
    )
    checkpoint = result.resume.checkpoint
    assert checkpoint is not None
    inspect_resume_state(checkpoint, restore=True)
    first = _trainer_rng_fingerprint()
    inspect_resume_state(checkpoint, restore=True)
    second = _trainer_rng_fingerprint()
    assert first == second
    assert result.resume.sampler_position == 2


def test_resume_does_not_claim_equivalence_when_optimizer_is_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(_config(run_dir, fault=FaultSpec(step=2, point="after_save", kind="oom")))
    optimizer = run_dir / "checkpoint-2" / "optimizer.pt"
    optimizer.unlink()

    inspection = inspect_resume_state(run_dir / "checkpoint-2")

    assert is_valid_checkpoint(run_dir / "checkpoint-2") is True
    assert inspection.equivalent is False
    assert "optimizer" in inspection.missing
    assert inspection.reason is not None
    assert inspection.reason.startswith("missing ")
    assert "optimizer" in inspection.reason


def test_resolve_resume_checkpoint_quarantines_incomplete_trees(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(_config(run_dir, fault=FaultSpec(step=2, point="after_save", kind="oom")))
    broken = run_dir / "checkpoint-3"
    broken.mkdir()
    (broken / "adapter_model.safetensors").write_bytes(b"partial")

    class Runtime:
        allow_unsafe_serialization = False

    class Training:
        resume_checkpoint_path = None
        resume_from_latest_checkpoint = True

    class Config:
        runtime = Runtime()
        training = Training()

    selected = resolve_resume_checkpoint(str(run_dir), Config())

    assert selected == str((run_dir / "checkpoint-2").resolve())
    assert not broken.exists()
    assert "missing_trainer_state" in _reasons(run_dir)


def test_failure_cleanup_does_not_delete_recoverable_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(
        _config(run_dir, fault=FaultSpec(step=4, point="during_write", kind="short_write"))
    )

    with pytest.raises(ValueError, match="only recoverable artifact"):
        plan_cleanup(str(run_dir), CleanupOptions(force=False))

    assert list_valid_checkpoints(run_dir) == [(run_dir / "checkpoint-2").resolve()]
    evidence = last_failure_evidence(run_dir)
    assert evidence is not None
    assert evidence["last_complete_step"] == 2
    trainer_state = json.loads(
        (Path(evidence["last_complete_checkpoint"]) / "trainer_state.json").read_text()
    )
    assert trainer_state["global_step"] == 2


def test_select_resume_checkpoint_is_deterministic_with_mixed_trees(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(_config(run_dir, max_steps=2))
    later = run_dir / "checkpoint-10"
    later.mkdir()
    (later / "trainer_state.json").write_text("{}")

    inspection = select_resume_checkpoint(run_dir)

    assert inspection.checkpoint == (run_dir / "checkpoint-2").resolve()
    assert inspection.equivalent is True
    assert not later.exists()


def test_quarantine_tree_moves_directory_symlink_as_leaf(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    canary = target / "keep-me.txt"
    canary.write_text("external\n")
    linked = run_dir / "checkpoint-99"
    linked.symlink_to(target, target_is_directory=True)

    moved = quarantine_tree(linked, reason="leftover_staging", run_dir=run_dir)

    assert not linked.exists()
    assert not linked.is_symlink()
    assert canary.read_text() == "external\n"
    assert not (target / QUARANTINE_REASON_FILENAME).exists()
    assert (moved / "checkpoint-99").is_symlink()
    assert (moved / "checkpoint-99").resolve() == target.resolve()
    assert read_quarantine_reason(moved)["reason"] == "leftover_staging"


def test_quarantine_tree_refuses_mount_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    source = run_dir / "checkpoint-3"
    source.mkdir(parents=True)
    (source / "keep-me.txt").write_text("mounted\n")
    monkeypatch.setattr(
        "agoge_forger.train.checkpoints.os.path.ismount",
        lambda path: Path(path) == source,
    )

    with pytest.raises(ValueError, match="mount point"):
        quarantine_tree(source, reason="missing_trainer_state", run_dir=run_dir)

    assert (source / "keep-me.txt").read_text() == "mounted\n"
    assert not (source / QUARANTINE_REASON_FILENAME).exists()


def test_quarantine_incomplete_skips_mounted_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mounted = run_dir / "checkpoint-3"
    mounted.mkdir()
    canary = mounted / "keep-me.txt"
    canary.write_text("mounted\n")
    (mounted / "adapter_model.safetensors").write_bytes(b"partial")
    broken = run_dir / "checkpoint-4"
    broken.mkdir()
    (broken / "adapter_model.safetensors").write_bytes(b"partial")
    monkeypatch.setattr(
        "agoge_forger.path_safety.os.path.ismount",
        lambda path: Path(path) == mounted,
    )

    moved = quarantine_incomplete_checkpoints(run_dir)

    assert mounted.is_dir()
    assert canary.read_text() == "mounted\n"
    assert not broken.exists()
    assert "missing_trainer_state" in {read_quarantine_reason(path)["reason"] for path in moved}


def test_quarantine_incomplete_does_not_walk_staging_symlink_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = tmp_path / "outside-staging"
    target.mkdir()
    (target / "keep-me.txt").write_text("external\n")
    linked = run_dir / f"{CHECKPOINT_STAGING_PREFIX}link"
    linked.symlink_to(target, target_is_directory=True)

    moved = quarantine_incomplete_checkpoints(run_dir)

    assert linked.is_symlink() is False
    assert not linked.exists()
    assert (target / "keep-me.txt").read_text() == "external\n"
    assert not (target / QUARANTINE_REASON_FILENAME).exists()
    assert len(moved) == 1
    assert (moved[0] / linked.name).is_symlink()
    assert read_quarantine_reason(moved[0])["reason"] == "leftover_staging"


def test_quarantine_tree_replaces_existing_reason_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    source = run_dir / "checkpoint-3"
    source.mkdir(parents=True)
    (source / QUARANTINE_REASON_FILENAME).write_text('{"reason": "stale"}\n')

    moved = quarantine_tree(source, reason="interrupted_save", run_dir=run_dir)

    assert read_quarantine_reason(moved)["reason"] == "interrupted_save"
    assert not source.exists()


def test_select_resume_skips_truncated_newest_adapter_weights(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(_config(run_dir))
    newest = run_dir / "checkpoint-4"
    (newest / ADAPTER_WEIGHT_FILES[0]).write_bytes(b"\0\0\0\0")
    assert is_valid_checkpoint(newest) is True

    inspection = select_resume_checkpoint(run_dir)

    assert not newest.exists()
    assert "short_write" in _reasons(run_dir)
    assert inspection.checkpoint == (run_dir / "checkpoint-2").resolve()
    assert inspection.equivalent is True
    assert inspection.global_step == 2


def test_resolve_resume_checkpoint_skips_truncated_newest_weights(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(
        _config(run_dir, max_steps=2, fault=FaultSpec(step=2, point="after_save", kind="oom"))
    )
    newest = run_dir / "checkpoint-4"
    newest.mkdir()
    for name in ("trainer_state.json", "adapter_config.json", ADAPTER_WEIGHT_FILES[0]):
        (newest / name).write_bytes((run_dir / "checkpoint-2" / name).read_bytes())
    (newest / ADAPTER_WEIGHT_FILES[0]).write_bytes(b"\0\0\0\0")

    class Runtime:
        allow_unsafe_serialization = False

    class Training:
        resume_checkpoint_path = None
        resume_from_latest_checkpoint = True

    class Config:
        runtime = Runtime()
        training = Training()

    selected = resolve_resume_checkpoint(str(run_dir), Config())

    assert selected == str((run_dir / "checkpoint-2").resolve())
    assert not newest.exists()
    assert "short_write" in _reasons(run_dir)


def test_select_resume_skips_malformed_newest_trainer_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(_config(run_dir))
    newest = run_dir / "checkpoint-4"
    (newest / "trainer_state.json").write_text("{}\n")

    inspection = select_resume_checkpoint(run_dir)

    assert is_valid_checkpoint(newest) is True
    assert find_latest_valid_checkpoint(run_dir) == newest.resolve()
    assert inspection.checkpoint == (run_dir / "checkpoint-2").resolve()
    assert inspection.equivalent is True
    assert inspect_resume_state(newest).equivalent is False


def test_sampler_position_is_missing_without_sidecar(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(
        _config(run_dir, max_steps=2, fault=FaultSpec(step=2, point="after_save", kind="oom"))
    )
    (run_dir / "checkpoint-2" / RESUME_SIDECAR_FILENAME).unlink()

    inspection = inspect_resume_state(run_dir / "checkpoint-2")

    assert is_valid_checkpoint(run_dir / "checkpoint-2") is True
    assert inspection.equivalent is False
    assert "sampler_position" in inspection.missing


def test_resumed_optimizer_keeps_momentum_tensors(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(_config(run_dir, fault=FaultSpec(step=2, point="after_save", kind="oom")))
    first = torch_mapping(run_dir / "checkpoint-2" / "optimizer.pt", require_data_record=True)
    run_fault_harness(_config(run_dir))
    later = torch_mapping(run_dir / "checkpoint-4" / "optimizer.pt", require_data_record=True)
    assert first is not None and later is not None
    assert torch.equal(first["state"][0]["exp_avg"], later["state"][0]["exp_avg"])
    assert int(later["state"][0]["step"].item()) == 4
    first_sched = torch_mapping(run_dir / "checkpoint-2" / "scheduler.pt")
    later_sched = torch_mapping(run_dir / "checkpoint-4" / "scheduler.pt")
    assert first_sched is not None and later_sched is not None
    assert later_sched["base_lrs"] == first_sched["base_lrs"]
    assert later_sched["last_epoch"] == 4


def test_completed_rerun_does_not_quarantine_existing_export(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first = run_fault_harness(_config(run_dir))
    config_bytes = (run_dir / "adapter_config.json").read_bytes()
    weight_bytes = (run_dir / ADAPTER_WEIGHT_FILES[0]).read_bytes()

    second = run_fault_harness(_config(run_dir))

    assert first.export_published is True
    assert second.completed is True
    assert is_adapter_artifact(run_dir) is True
    assert (run_dir / "adapter_config.json").read_bytes() == config_bytes
    assert (run_dir / ADAPTER_WEIGHT_FILES[0]).read_bytes() == weight_bytes
    assert "export_failure" not in _reasons(run_dir)


def test_export_does_not_quarantine_preexisting_root_adapter_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fault_harness(
        _config(run_dir, max_steps=2, fault=FaultSpec(step=2, point="after_save", kind="oom"))
    )
    original = '{"peft_type": "LORA"}\n'
    (run_dir / "adapter_config.json").write_text(original)

    result = run_fault_harness(_config(run_dir, max_steps=2))

    assert result.export_published is False
    assert (run_dir / "adapter_config.json").read_text() == original
    assert not (run_dir / ADAPTER_WEIGHT_FILES[0]).exists()


def test_unfireable_fault_spec_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a save step"):
        _config(tmp_path / "run", fault=FaultSpec(step=3, point="before_save", kind="oom"))
    with pytest.raises(ValueError, match="during_export fault step"):
        _config(
            tmp_path / "run",
            fault=FaultSpec(step=2, point="during_export", kind="export_failure"),
        )
