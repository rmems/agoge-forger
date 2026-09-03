"""Trainer checkpoint-state readiness tests for run-status."""

from __future__ import annotations

import json
import random
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model
from test_run_status import (
    TINY_LLAMA_CONFIG,
    _make_run_dir,
    _minimal_safetensors,
    _optimizer_state,
    _write_checkpoint,
    _write_test_tokenizer,
    _write_torch_state,
)
from transformers import AutoModelForCausalLM, LlamaConfig, Trainer
from transformers.training_args import ParallelMode

from agoge_forger._run_status_torch_archive import torch_mapping
from agoge_forger.run_status import build_run_status


def test_empty_checkpoint_weights_are_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"")

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["valid_count"] == 1
    assert report["resume"]["ready"] is False
    assert report["resume"]["checkpoint_path"] == str(checkpoint_dir.resolve())


def test_header_without_data_region_is_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(_minimal_safetensors()[:-1])

    report = build_run_status(str(run_dir))

    assert report["checkpoints"]["valid_count"] == 1
    assert report["resume"]["ready"] is False


@pytest.mark.parametrize("missing_name", ["optimizer.pt", "scheduler.pt"])
def test_missing_training_state_is_not_resume_ready(tmp_path, missing_name):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    missing = checkpoint_dir / missing_name
    missing.unlink(missing_ok=True)

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


def test_empty_per_parameter_optimizer_state_is_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    _write_torch_state(
        checkpoint_dir / "optimizer.pt",
        {"state": {0: {}}, "param_groups": [{"params": [0]}]},
    )

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_optimizer_moment_shapes_must_match_trainable_adapter_tensors(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    unrelated = {
        "state": {
            parameter_id: {
                "step": torch.tensor(50.0),
                "exp_avg": torch.zeros(999),
                "exp_avg_sq": torch.zeros(999),
            }
            for parameter_id in (0, 1)
        },
        "param_groups": [{"params": [0, 1]}],
    }
    _write_torch_state(checkpoint_dir / "optimizer.pt", unrelated)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_optimizer_state_must_cover_every_trainable_adapter_tensor(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    _write_torch_state(checkpoint_dir / "optimizer.pt", _optimizer_state(50, ((1, 1),)))

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_optimizer_moment_shapes_follow_parameter_group_order(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    swapped = _optimizer_state(50, ((8, 1), (1, 8)))
    _write_torch_state(checkpoint_dir / "optimizer.pt", swapped)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_genuine_multitarget_peft_optimizer_order_is_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = run_dir / "checkpoint-1"
    checkpoint_dir.mkdir()
    base_dir = tmp_path / "base-model"
    base_model = AutoModelForCausalLM.from_config(LlamaConfig(**TINY_LLAMA_CONFIG))
    base_model.save_pretrained(base_dir)
    _write_test_tokenizer(base_dir)
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=2,
            target_modules=["q_proj", "gate_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.peft_config["default"].base_model_name_or_path = str(base_dir)
    model.save_pretrained(checkpoint_dir)

    named_trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert [tuple(parameter.shape) for _, parameter in named_trainable] == [
        (2, 8),
        (8, 2),
        (2, 8),
        (16, 2),
    ]
    trainer = object.__new__(Trainer)
    decay_names = set(trainer.get_decay_parameter_names(model))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for name, parameter in named_trainable if name in decay_names],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    parameter for name, parameter in named_trainable if name not in decay_names
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=0.001,
        weight_decay=0.01,
    )
    for _, parameter in named_trainable:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    _write_torch_state(
        checkpoint_dir / "scheduler.pt",
        {
            "last_epoch": 1,
            "_step_count": 2,
            "base_lrs": [0.001, 0.001],
            "_last_lr": [0.001, 0.001],
        },
    )
    _write_torch_state(checkpoint_dir / "rng_state.pth")
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": 1, "train_batch_size": 1})
    )

    assert build_run_status(str(run_dir))["resume"]["ready"] is True


def test_structural_lora_config_is_rejected_before_peft_construction(monkeypatch, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    config_path = checkpoint_dir / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["layer_replication"] = [[0, 1]]
    config_path.write_text(json.dumps(config))

    def fail_constructor(*args, **kwargs):
        raise AssertionError("untrusted structural config reached get_peft_model")

    monkeypatch.setattr(
        "agoge_forger._run_status_optimizer_order.get_peft_model",
        fail_constructor,
    )

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_optimizer_parameter_group_requires_adamw_hyperparameters(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    payload = _optimizer_state(50)
    payload["param_groups"] = [{"params": [0, 1]}]
    _write_torch_state(checkpoint_dir / "optimizer.pt", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_older_adamw_group_without_decoupled_marker_is_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    payload = _optimizer_state(50)
    for group in payload["param_groups"]:
        group.pop("decoupled_weight_decay")
    _write_torch_state(checkpoint_dir / "optimizer.pt", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is True


def test_amsgrad_requires_maximum_second_moment_for_each_group_parameter(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    payload = _optimizer_state(50)
    payload["param_groups"][0]["amsgrad"] = True
    _write_torch_state(checkpoint_dir / "optimizer.pt", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_optimizer_parameter_ids_must_remain_in_their_trainer_group(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    payload = _optimizer_state(50)
    payload["param_groups"][0]["params"] = [0]
    payload["param_groups"][0]["amsgrad"] = True
    payload["param_groups"][1]["params"] = [1]
    payload["state"][0]["max_exp_avg_sq"] = torch.zeros(1, 8)
    _write_torch_state(checkpoint_dir / "optimizer.pt", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize(
    "maximum",
    [torch.zeros(2), torch.zeros(1, 8, dtype=torch.float64), "not-a-tensor"],
    ids=["wrong-shape", "wrong-dtype", "wrong-type"],
)
def test_amsgrad_rejects_incompatible_maximum_second_moment(tmp_path, maximum):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    payload = _optimizer_state(50)
    payload["param_groups"][0]["amsgrad"] = True
    for state in payload["state"].values():
        state["max_exp_avg_sq"] = torch.zeros_like(state["exp_avg_sq"])
    payload["state"][0]["max_exp_avg_sq"] = maximum
    _write_torch_state(checkpoint_dir / "optimizer.pt", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr", float("nan")),
        ("lr", 10**1000),
        ("betas", (0.9, 1.0)),
        ("eps", "1e-8"),
        ("weight_decay", -0.01),
        ("amsgrad", "false"),
        ("decoupled_weight_decay", False),
    ],
)
def test_optimizer_parameter_group_rejects_invalid_values(tmp_path, field, value):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    payload = _optimizer_state(50)
    payload["param_groups"][0][field] = value
    _write_torch_state(checkpoint_dir / "optimizer.pt", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize(
    "rates",
    [
        {"base_lrs": [], "_last_lr": []},
        {"base_lrs": [0.001, 0.001], "_last_lr": [0.001]},
        {"base_lrs": [float("inf"), 0.001], "_last_lr": [0.001, 0.001]},
        {"base_lrs": [0.001, 0.001], "_last_lr": [float("nan"), 0.001]},
        {"base_lrs": [10**1000, 0.001], "_last_lr": [0.001, 0.001]},
    ],
    ids=["empty", "cardinality-mismatch", "nonfinite-base", "nonfinite-last", "oversized-base"],
)
def test_scheduler_rates_must_match_optimizer_groups(tmp_path, rates):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    payload = {"last_epoch": 50, "_step_count": 51, **rates}
    _write_torch_state(checkpoint_dir / "scheduler.pt", payload)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


@pytest.mark.parametrize(
    "entry",
    [
        {"exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)},
        {"step": torch.tensor(50.0), "exp_avg_sq": torch.zeros(1)},
        {"step": torch.tensor(50.0), "exp_avg": torch.zeros(1)},
        {
            "step": torch.tensor(50.0),
            "exp_avg": torch.zeros(1),
            "exp_avg_sq": torch.zeros(2),
        },
    ],
    ids=["missing-step", "missing-first-moment", "missing-second-moment", "shape-mismatch"],
)
def test_incomplete_per_parameter_optimizer_state_is_not_resume_ready(tmp_path, entry):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    _write_torch_state(
        checkpoint_dir / "optimizer.pt",
        {"state": {0: entry}, "param_groups": [{"params": [0]}]},
    )

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
    (checkpoint_dir / "trainer_state.json").write_text('{"global_step": 49, "train_batch_size": 1}')

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_trainer_state_requires_positive_batch_size(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / "trainer_state.json").write_text('{"global_step": 50, "train_batch_size": 0}')

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_trainer_state_rejects_unknown_installed_schema_fields(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": 50, "train_batch_size": 1, "junk": 1})
    )

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_symlinked_trainer_state_is_not_resume_ready(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    external = tmp_path / "external-trainer-state.json"
    external.write_text('{"global_step": 50, "train_batch_size": 1}')
    (checkpoint_dir / "trainer_state.json").unlink()
    (checkpoint_dir / "trainer_state.json").symlink_to(external)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_deeply_nested_trainer_state_fails_closed(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    nested = '{"global_step":50,"train_batch_size":1,"nested":' + "[" * 100_000
    nested += "0" + "]" * 100_000 + "}"
    (checkpoint_dir / "trainer_state.json").write_text(nested)

    assert build_run_status(str(run_dir))["resume"]["ready"] is False


def test_transformers_single_process_cuda_rng_payload_is_supported(tmp_path, monkeypatch):
    """Transformers 5.12 saves one CUDA Philox seed/offset tensor for world size 1."""
    run_dir = _make_run_dir(tmp_path)
    checkpoint_dir = _write_checkpoint(run_dir, 50)
    cuda_state = torch.tensor([0] * 16, dtype=torch.uint8)
    fake_trainer = SimpleNamespace(
        args=SimpleNamespace(world_size=1, parallel_mode=ParallelMode.NOT_DISTRIBUTED)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.random, "get_rng_state", lambda: cuda_state)

    Trainer._save_rng_state(fake_trainer, str(checkpoint_dir))
    payload = torch_mapping(checkpoint_dir / "rng_state.pth", allow_numpy=True)

    assert payload is not None
    assert payload["cuda"].shape == (16,)
    assert payload["cuda"].dtype == torch.uint8
    assert build_run_status(str(run_dir))["resume"]["ready"] is True


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
