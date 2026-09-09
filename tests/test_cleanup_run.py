"""Library-level behavior for `agoge cleanup-run`."""

import json
import os
import shutil
from pathlib import Path

import pytest

from agoge_forger.artifacts.safetensors_io import write_artifact_index
from agoge_forger.cleanup_run import (
    CLEANUP_SCHEMA_VERSION,
    CleanupOptions,
    execute_cleanup,
    format_cleanup_table,
    plan_cleanup,
)
from agoge_forger.run_status import build_run_status
from tests.test_run_status import (
    _make_run_dir,
    _write_checkpoint,
    _write_final_adapter,
    _write_merged_model,
)


def _run_with_checkpoints(tmp_path, steps=(50, 100, 150), final_adapter=True):
    run_dir = _make_run_dir(tmp_path)
    for step in steps:
        _write_checkpoint(run_dir, step)
    if final_adapter:
        _write_final_adapter(run_dir)
        # A run that actually finished carries a sealed index: trainer.py writes
        # one at the end of every run, and export-final-model reads provenance
        # off it unconditionally. Without it this is an interrupted run, which
        # the guard is supposed to refuse.
        write_artifact_index(str(run_dir), producer_provenance=_provenance())
    return run_dir


def _checkpoint_names(run_dir):
    return sorted(entry.name for entry in run_dir.iterdir() if entry.name.startswith("checkpoint-"))


# --- dry-run byte accounting -------------------------------------------------


def test_plan_reports_every_checkpoint_and_deletes_nothing(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    plan = plan_cleanup(str(run_dir))

    assert plan["schema_version"] == CLEANUP_SCHEMA_VERSION
    assert plan["dry_run"] is True
    assert [entry["step"] for entry in plan["removed"]] == [50, 100, 150]
    # The whole point of a dry run: the filesystem is untouched.
    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-150", "checkpoint-50"]


def test_planned_bytes_match_the_files_on_disk(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    expected = sum(p.stat().st_size for p in (run_dir / "checkpoint-50").rglob("*") if p.is_file())

    plan = plan_cleanup(str(run_dir))

    assert plan["bytes_reclaimed"] == expected
    assert plan["removed"][0]["bytes"] == expected


def test_bytes_reclaimed_counts_only_what_was_removed(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)
    everything = plan_cleanup(str(run_dir))["bytes_reclaimed"]

    report = execute_cleanup(plan_cleanup(str(run_dir), CleanupOptions(keep_latest=1)))

    assert report["bytes_reclaimed"] == sum(entry["bytes"] for entry in report["removed"])
    # One checkpoint was kept, so less was reclaimed than a full sweep would.
    assert 0 < report["bytes_reclaimed"] < everything


def test_execute_removes_checkpoints_and_keeps_the_adapter(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["dry_run"] is False
    assert _checkpoint_names(run_dir) == []
    assert (run_dir / "adapter_model.safetensors").is_file()
    assert (run_dir / "adapter_config.json").is_file()


# --- keep-latest -------------------------------------------------------------


@pytest.mark.parametrize(
    ("keep", "survivors"),
    [
        (0, []),
        (1, ["checkpoint-150"]),
        (2, ["checkpoint-100", "checkpoint-150"]),
        (99, ["checkpoint-100", "checkpoint-150", "checkpoint-50"]),
    ],
)
def test_keep_latest_retains_the_newest_valid_checkpoints(tmp_path, keep, survivors):
    run_dir = _run_with_checkpoints(tmp_path)

    execute_cleanup(plan_cleanup(str(run_dir), CleanupOptions(keep_latest=keep)))

    assert _checkpoint_names(run_dir) == survivors


def test_keep_latest_preserves_zero_padded_checkpoint_names(tmp_path):
    """`checkpoint-050` parses as step 50. Rebuilding `checkpoint-50` would
    miss the keep set and delete the snapshot `--keep-latest 1` asked to keep.
    """
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint(run_dir, 50).rename(run_dir / "checkpoint-050")
    _write_final_adapter(run_dir)
    write_artifact_index(str(run_dir), producer_provenance=_provenance())

    execute_cleanup(plan_cleanup(str(run_dir), CleanupOptions(keep_latest=1)))

    assert _checkpoint_names(run_dir) == ["checkpoint-050"]


def test_keep_latest_does_not_spend_slots_on_symlinked_checkpoints(tmp_path):
    """A symlink to a valid checkpoint still parses as a valid step. If it
    consumed the keep slot, `--keep-latest 1` would delete every real snapshot.
    """
    run_dir = _run_with_checkpoints(tmp_path, steps=(50, 100))
    real = run_dir / "checkpoint-100"
    outside = tmp_path / "elsewhere"
    real.rename(outside)
    (run_dir / "checkpoint-100").symlink_to(outside, target_is_directory=True)

    execute_cleanup(plan_cleanup(str(run_dir), CleanupOptions(keep_latest=1)))

    assert (run_dir / "checkpoint-50").is_dir()
    assert (run_dir / "checkpoint-100").is_symlink()
    assert (outside / "adapter_model.safetensors").is_file()


def test_invalid_checkpoint_is_reclaimed_even_inside_the_keep_window(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    # A crashed run leaves a half-written snapshot with the highest step. It must
    # not consume the keep slot, because it is not a usable resume point.
    partial = run_dir / "checkpoint-999"
    partial.mkdir()
    (partial / "trainer_state.json").write_text(json.dumps({"global_step": 999}))

    execute_cleanup(plan_cleanup(str(run_dir), CleanupOptions(keep_latest=1)))

    assert _checkpoint_names(run_dir) == ["checkpoint-50"]


def test_negative_keep_latest_is_rejected(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    with pytest.raises(ValueError, match="must not be negative"):
        plan_cleanup(str(run_dir), CleanupOptions(keep_latest=-1))


# --- refusal paths -----------------------------------------------------------


def test_refuses_when_checkpoints_are_the_only_artifact(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)

    with pytest.raises(ValueError, match="only recoverable artifact"):
        plan_cleanup(str(run_dir))

    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-150", "checkpoint-50"]


def test_force_overrides_the_guard_and_records_it(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)

    plan = plan_cleanup(str(run_dir), CleanupOptions(force=True))

    assert plan["guard"] == {"final_artifact": False, "forced": True}
    assert len(plan["removed"]) == 3


def test_unrelated_merged_model_does_not_satisfy_the_guard(tmp_path):
    """`merged/<run_name>` is a naming convention. A leftover model from an older
    experiment of the same name must not authorize deleting a new run's only
    checkpoints."""
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)
    _write_merged_model(tmp_path / "merged" / "demo_run")

    with pytest.raises(ValueError, match="only recoverable artifact"):
        plan_cleanup(str(run_dir))

    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-150", "checkpoint-50"]


def test_merged_model_sealed_from_this_run_satisfies_the_guard(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)
    merged = tmp_path / "merged" / "demo_run"
    _write_merged_model(merged)
    # Same sealed provenance on both sides is what binds the merge to this run.
    write_artifact_index(str(run_dir), producer_provenance=_provenance())
    write_artifact_index(str(merged), producer_provenance=_provenance())

    plan = plan_cleanup(str(run_dir))

    assert plan["guard"] == {"final_artifact": True, "forced": False}


def test_incomplete_final_adapter_does_not_satisfy_the_guard(tmp_path):
    """Filenames alone are not readiness: a run interrupted before its weights
    were written still looks finished to a presence check."""
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)
    (run_dir / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "x/y"}))
    (run_dir / "adapter_model.safetensors").write_bytes(b"truncated")

    with pytest.raises(ValueError, match="only recoverable artifact"):
        plan_cleanup(str(run_dir))


def test_symlinked_run_dir_is_refused(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)
    link = tmp_path / "adapters" / "link_run"
    link.symlink_to(run_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked path"):
        plan_cleanup(str(link))

    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-150", "checkpoint-50"]


def test_symlinked_parent_of_run_dir_is_refused(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)
    linked_parent = tmp_path / "linked_adapters"
    linked_parent.symlink_to(run_dir.parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked path"):
        plan_cleanup(str(linked_parent / run_dir.name))

    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-150", "checkpoint-50"]


def test_symlinked_checkpoint_is_skipped_not_followed(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "keepme.txt").write_text("must survive")
    (run_dir / "checkpoint-900").symlink_to(outside, target_is_directory=True)

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["skipped"] == [{"path": str(run_dir / "checkpoint-900"), "reason": "symlink"}]
    assert (outside / "keepme.txt").is_file()


def test_mounted_checkpoint_is_skipped(tmp_path, monkeypatch):
    run_dir = _run_with_checkpoints(tmp_path, steps=(50, 100))
    mounted = run_dir / "checkpoint-100"
    monkeypatch.setattr(
        "agoge_forger.cleanup_run.os.path.ismount",
        lambda path: Path(path) == mounted,
    )

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert {"path": str(mounted), "reason": "mount"} in report["skipped"]
    assert mounted.is_dir()
    assert not (run_dir / "checkpoint-50").exists()


def test_missing_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plan_cleanup(str(tmp_path / "nope"))


def test_parent_traversal_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not contain"):
        plan_cleanup(str(tmp_path / ".." / "escape"))


# --- artifact index consistency ---------------------------------------------


def test_cleanup_keeps_the_run_export_ready(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)
    assert build_run_status(str(run_dir))["export"]["ready"] is True

    execute_cleanup(plan_cleanup(str(run_dir)))

    assert build_run_status(str(run_dir))["export"]["ready"] is True


def test_artifact_index_is_rewritten_over_the_survivors(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    write_artifact_index(str(run_dir), producer_provenance=_provenance())
    listed = _indexed_files(run_dir)
    assert any(name.startswith("checkpoint-50/") for name in listed)

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["artifact_index_rewritten"] is True
    assert not any(name.startswith("checkpoint-") for name in _indexed_files(run_dir))


def test_run_without_an_index_is_cleaned_without_creating_one(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,), final_adapter=False)
    _write_final_adapter(run_dir)  # weights, but never sealed

    # No sealed provenance means export-final-model would refuse this run, so
    # the guard does too; --force is the documented way past it.
    report = execute_cleanup(plan_cleanup(str(run_dir), CleanupOptions(force=True)))

    assert report["artifact_index_rewritten"] is False
    assert not (run_dir / "artifact_index.json").exists()


def test_one_rmtree_failure_does_not_abort_the_rest(tmp_path, monkeypatch):
    """One busy checkpoint must not strand the others, and the index must
    still be rewritten over what actually survived.
    """
    run_dir = _run_with_checkpoints(tmp_path, steps=(50, 100))
    busy = run_dir / "checkpoint-50"
    real_rmtree = shutil.rmtree

    def flaky(path, *args, **kwargs):
        if Path(path) == busy:
            raise OSError("busy")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("agoge_forger.cleanup_run.shutil.rmtree", flaky)

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert busy.is_dir()
    assert not (run_dir / "checkpoint-100").exists()
    assert report["failed"] == [{"path": str(busy), "error": "busy"}]
    assert report["artifact_index_rewritten"] is True
    assert any(name.startswith("checkpoint-50/") for name in _indexed_files(run_dir))
    assert not any(name.startswith("checkpoint-100/") for name in _indexed_files(run_dir))


def test_failed_index_rewrite_is_reported_as_a_failure(tmp_path, monkeypatch):
    """A stale index after deletion is a failed cleanup, not a quiet success."""
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    write_artifact_index(str(run_dir), producer_provenance=_provenance())

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("agoge_forger.cleanup_run.write_artifact_index", explode)

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["artifact_index_rewritten"] is False
    assert report["failed"] == [
        {"path": str(run_dir / "artifact_index.json"), "error": "disk full"}
    ]


def test_unreadable_file_is_counted_as_zero_not_fatal(tmp_path, monkeypatch):
    """A file that vanishes mid-walk must not abort a cleanup plan."""
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    real_getsize = os.path.getsize

    def flaky(path, *args, **kwargs):
        if str(path).endswith("adapter_model.safetensors"):
            raise OSError("vanished")
        return real_getsize(path, *args, **kwargs)

    checkpoint = run_dir / "checkpoint-50"
    # Everything except the file that will be unreadable; that one must
    # contribute exactly zero rather than aborting the walk.
    expected = sum(
        real_getsize(p)
        for p in checkpoint.rglob("*")
        if p.is_file() and p.name != "adapter_model.safetensors"
    )
    assert expected > 0

    monkeypatch.setattr(os.path, "getsize", flaky)

    plan = plan_cleanup(str(run_dir))

    assert len(plan["removed"]) == 1
    assert plan["bytes_reclaimed"] == expected


def test_malformed_artifact_index_does_not_crash_cleanup(tmp_path):
    """A JSON array parses fine but raises TypeError, not ValueError, on read.

    The index is still rebuilt: one that exists must never be left listing files
    cleanup just deleted, whether or not its provenance could be read.
    """
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    (run_dir / "artifact_index.json").write_text("[]")

    report = execute_cleanup(plan_cleanup(str(run_dir), CleanupOptions(force=True)))

    assert report["artifact_index_rewritten"] is True
    assert report["failed"] == []
    assert _checkpoint_names(run_dir) == []
    assert not any(name.startswith("checkpoint-") for name in _indexed_files(run_dir))


def test_index_without_provenance_is_still_rebuilt(tmp_path):
    """Otherwise cleanup deletes the checkpoints and leaves the index naming
    them, which is exactly what breaks evaluation-contract validation."""
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    write_artifact_index(str(run_dir))  # no producer_provenance

    report = execute_cleanup(plan_cleanup(str(run_dir), CleanupOptions(force=True)))

    assert report["artifact_index_rewritten"] is True
    assert not any(name.startswith("checkpoint-") for name in _indexed_files(run_dir))


def test_missing_recorded_survivor_blocks_the_reseal(tmp_path):
    """A sealed non-checkpoint file deleted outside cleanup must not be
    dropped from the index while the old attestation is kept.
    """
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    (run_dir / "README.txt").write_text("sealed contents")
    write_artifact_index(str(run_dir), producer_provenance=_provenance())
    (run_dir / "README.txt").unlink()

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["artifact_index_rewritten"] is False
    assert len(report["failed"]) == 1
    assert "is missing but" in report["failed"][0]["error"]


def test_index_without_artifacts_does_not_carry_old_provenance(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    payload = {
        "output_dir": str(run_dir),
        "producer_provenance": _provenance(),
    }
    (run_dir / "artifact_index.json").write_text(json.dumps(payload))

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["artifact_index_rewritten"] is False
    assert report["failed"]
    assert "no verifiable artifacts list" in report["failed"][0]["error"]


def test_file_added_after_sealing_blocks_the_reseal(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    write_artifact_index(str(run_dir), producer_provenance=_provenance())
    (run_dir / "planted.txt").write_text("added after sealing")

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["artifact_index_rewritten"] is False
    assert "appeared after" in report["failed"][0]["error"]


def test_modified_surviving_file_blocks_the_reseal(tmp_path):
    """Resealing would stamp the original provenance onto altered content, so a
    later contract check could no longer detect the change."""
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    _write_final_adapter(run_dir)
    # A sealed file that is not the adapter weights, so the run stays
    # export-ready and the guard lets cleanup reach the reseal.
    (run_dir / "README.txt").write_text("sealed contents")
    write_artifact_index(str(run_dir), producer_provenance=_provenance())
    (run_dir / "README.txt").write_text("edited after sealing")

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["artifact_index_rewritten"] is False
    assert len(report["failed"]) == 1
    assert "changed since" in report["failed"][0]["error"]


def test_index_is_left_alone_when_nothing_was_removed(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    write_artifact_index(str(run_dir), producer_provenance=_provenance())
    before = (run_dir / "artifact_index.json").read_bytes()

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["removed"] == []
    assert report["artifact_index_rewritten"] is False
    assert (run_dir / "artifact_index.json").read_bytes() == before


# --- rendering ---------------------------------------------------------------


def test_table_renders_every_reported_field(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    table = format_cleanup_table(plan_cleanup(str(run_dir)))

    for label in ("run_name:", "dry_run:", "keep_latest:", "gib_reclaimed:", "final_artifact:"):
        assert label in table
    assert "dry_run:" in table and "yes" in table.split("dry_run:")[1].splitlines()[0]
    assert "final_artifact:" in table and "yes" in table.split("final_artifact:")[1].splitlines()[0]


def _provenance():
    return {
        "base_model_name_or_path": "example/base-model",
        "revision": "a" * 40,
        "training_split_manifest_sha256": "0" * 64,
        "training_split_name": "train",
        "training_split_sha256": "1" * 64,
    }


def _indexed_files(run_dir):
    payload = json.loads((run_dir / "artifact_index.json").read_text())
    return [entry["file"] for entry in payload["artifacts"]]


def test_unsealed_adapter_does_not_satisfy_the_guard(tmp_path):
    """Weights alone are not enough. `export-final-model` reads provenance off
    the adapter unconditionally, so a run interrupted before artifact_index.json
    cannot actually be exported -- deleting its checkpoints would strand it."""
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)
    _write_final_adapter(run_dir)

    with pytest.raises(ValueError, match="only recoverable artifact"):
        plan_cleanup(str(run_dir))

    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-150", "checkpoint-50"]
