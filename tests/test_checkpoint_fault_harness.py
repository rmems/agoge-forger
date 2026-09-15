"""CPU fault-injection coverage for single-process checkpoint resume."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from agoge_forger.cleanup_run import CleanupOptions, plan_cleanup
from agoge_forger.train.checkpoints import (
    find_latest_valid_checkpoint,
    is_adapter_artifact,
    is_valid_checkpoint,
    list_quarantined_checkpoints,
    list_valid_checkpoints,
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
from agoge_forger.train.resume_state import inspect_resume_state, select_resume_checkpoint


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
    }
    assert is_adapter_artifact(run_dir) is True


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
    first = (random.random(), float(np.random.random()), float(torch.rand(1)))
    inspect_resume_state(checkpoint, restore=True)
    second = (random.random(), float(np.random.random()), float(torch.rand(1)))
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
