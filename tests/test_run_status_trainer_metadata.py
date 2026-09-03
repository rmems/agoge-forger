"""Trainer and scheduler metadata integrity tests."""

import os
from pathlib import Path

import pytest
import torch
from test_run_status import _make_run_dir, _write_checkpoint, _write_torch_state
from transformers import get_scheduler

from agoge_forger import _run_status_trainer_state as trainer_state_validation
from agoge_forger.run_status import build_run_status


def test_genuine_trainer_lambda_scheduler_requires_serialized_lambda_state(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    optimizer = torch.optim.AdamW(
        [{"params": [parameters[0]]}, {"params": [parameters[1]]}],
        lr=0.001,
        weight_decay=0.0,
    )
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=0,
        num_training_steps=50,
    )
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    payload = scheduler.state_dict()
    assert payload["lr_lambdas"] == [{}, {}]
    _write_torch_state(checkpoint_dir / "scheduler.pt", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is True

    payload.pop("lr_lambdas")
    _write_torch_state(checkpoint_dir / "scheduler.pt", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize(
    "lr_lambdas",
    [[], [{}], [{}, {}, {}], [{}, "not-callable-state"]],
    ids=["empty", "too-few", "too-many", "invalid-entry"],
)
def test_scheduler_lambda_state_must_match_optimizer_groups(tmp_path, lr_lambdas):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    _write_torch_state(
        checkpoint_dir / "scheduler.pt",
        {
            "last_epoch": 50,
            "_step_count": 51,
            "base_lrs": [0.001, 0.001],
            "_last_lr": [0.001, 0.001],
            "lr_lambdas": lr_lambdas,
        },
    )

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_trainer_state_read_is_descriptor_bounded_when_path_stat_lies(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoint-50"
    checkpoint_dir.mkdir()
    state_path = checkpoint_dir / "trainer_state.json"
    state_path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    original_stat = Path.stat
    original_read = os.read
    read_sizes = []

    def misleading_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path != state_path:
            return result
        fields = list(result)
        fields[6] = 1
        return os.stat_result(fields)

    def unexpected_read(path, *args, **kwargs):
        if path == state_path:
            raise AssertionError("trainer state used an unbounded pathname read")
        return original_read_text(path, *args, **kwargs)

    def bounded_read(descriptor, size):
        read_sizes.append(size)
        return original_read(descriptor, size)

    original_read_text = Path.read_text
    monkeypatch.setattr(Path, "stat", misleading_stat)
    monkeypatch.setattr(Path, "read_text", unexpected_read)
    monkeypatch.setattr(os, "read", bounded_read)

    assert trainer_state_validation._trainer_state_step(checkpoint_dir) is None
    assert sum(read_sizes) == 4 * 1024 * 1024 + 1


def test_trainer_state_memory_error_fails_closed(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoint-50"
    checkpoint_dir.mkdir()
    state_path = checkpoint_dir / "trainer_state.json"
    state_path.write_text('{"global_step":50,"train_batch_size":1}')

    def fail_read(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(os, "read", fail_read)

    assert trainer_state_validation._trainer_state_step(checkpoint_dir) is None
