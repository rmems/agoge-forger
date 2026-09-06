"""Library-level behavior for `agoge cleanup-run`."""

import json

import pytest

from agoge_forger.artifacts.safetensors_io import write_artifact_index
from agoge_forger.cleanup_run import (
    CLEANUP_SCHEMA_VERSION,
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

    report = execute_cleanup(plan_cleanup(str(run_dir), keep_latest=1))

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

    execute_cleanup(plan_cleanup(str(run_dir), keep_latest=keep))

    assert _checkpoint_names(run_dir) == survivors


def test_invalid_checkpoint_is_reclaimed_even_inside_the_keep_window(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))
    # A crashed run leaves a half-written snapshot with the highest step. It must
    # not consume the keep slot, because it is not a usable resume point.
    partial = run_dir / "checkpoint-999"
    partial.mkdir()
    (partial / "trainer_state.json").write_text(json.dumps({"global_step": 999}))

    execute_cleanup(plan_cleanup(str(run_dir), keep_latest=1))

    assert _checkpoint_names(run_dir) == ["checkpoint-50"]


def test_negative_keep_latest_is_rejected(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)

    with pytest.raises(ValueError, match="must not be negative"):
        plan_cleanup(str(run_dir), keep_latest=-1)


# --- refusal paths -----------------------------------------------------------


def test_refuses_when_checkpoints_are_the_only_artifact(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)

    with pytest.raises(ValueError, match="only recoverable artifact"):
        plan_cleanup(str(run_dir))

    assert _checkpoint_names(run_dir) == ["checkpoint-100", "checkpoint-150", "checkpoint-50"]


def test_force_overrides_the_guard_and_records_it(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)

    plan = plan_cleanup(str(run_dir), force=True)

    assert plan["guard"] == {"final_artifact": False, "forced": True}
    assert len(plan["removed"]) == 3


def test_a_merged_model_alone_satisfies_the_guard(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path, final_adapter=False)
    _write_merged_model(tmp_path / "merged" / "demo_run")

    plan = plan_cleanup(str(run_dir))

    assert plan["guard"] == {"final_artifact": True, "forced": False}


def test_symlinked_run_dir_is_refused(tmp_path):
    run_dir = _run_with_checkpoints(tmp_path)
    link = tmp_path / "adapters" / "link_run"
    link.symlink_to(run_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked run directory"):
        plan_cleanup(str(link))

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
    run_dir = _run_with_checkpoints(tmp_path, steps=(50,))

    report = execute_cleanup(plan_cleanup(str(run_dir)))

    assert report["artifact_index_rewritten"] is False
    assert not (run_dir / "artifact_index.json").exists()


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
