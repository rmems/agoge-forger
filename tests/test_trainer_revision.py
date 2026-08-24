"""Revision from ExperimentConfig must reach load_base_model.

These tests mock the trainer's Hub/GPU/dataset side effects and raise out of
``load_base_model`` so nothing downloads weights or constructs SFTTrainer.
"""

from unittest.mock import MagicMock

import pytest

from agoge_forger.config import ExperimentConfig
from agoge_forger.train.trainer import _prepare_peft_model, run_training

PINNED_REVISION = "d3040b7c81a0a810fa13c6f392f3e304a0e121d5"


def _config(*, revision: str | None) -> ExperimentConfig:
    return ExperimentConfig(
        model_id="org/model",
        revision=revision,
        dataset_path="unused.jsonl",
        run_name="revision_pass_through",
    )


def _patch_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        "agoge_forger.train.trainer.check_cuda_available", lambda required=True: None
    )
    monkeypatch.setattr(
        "agoge_forger.train.trainer.get_gpu_report", lambda: {"device_name": "mock"}
    )
    monkeypatch.setattr("agoge_forger.train.trainer.estimate_training_risk", lambda *a, **k: None)
    monkeypatch.setattr("agoge_forger.train.trainer.warn_on_disk_pressure", lambda *a, **k: None)
    monkeypatch.setattr(
        "agoge_forger.train.trainer.validate_dataset_text_field_in_source", lambda *a, **k: None
    )


def test_run_training_passes_revision_to_load_base_model(monkeypatch):
    captured: dict[str, object] = {}

    def fake_load_base_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise RuntimeError("stop-before-trainer")

    _patch_preflight(monkeypatch)
    monkeypatch.setattr("agoge_forger.train.trainer.load_base_model", fake_load_base_model)

    with pytest.raises(RuntimeError, match="stop-before-trainer"):
        run_training(_config(revision=PINNED_REVISION))

    assert captured["kwargs"]["revision"] == PINNED_REVISION
    assert captured["args"][0] == "org/model"


def test_run_training_passes_none_revision_when_unset(monkeypatch):
    captured: dict[str, object] = {}

    def fake_load_base_model(*args, **kwargs):
        captured["kwargs"] = kwargs
        raise RuntimeError("stop-before-trainer")

    _patch_preflight(monkeypatch)
    monkeypatch.setattr("agoge_forger.train.trainer.load_base_model", fake_load_base_model)

    with pytest.raises(RuntimeError, match="stop-before-trainer"):
        run_training(_config(revision=None))

    assert captured["kwargs"]["revision"] is None


def test_prepare_peft_model_persists_revision_on_lora_config(monkeypatch):
    captured: dict[str, object] = {}

    def fake_get_peft_model(model, peft_config, **kwargs):
        captured["revision"] = peft_config.revision
        model.print_trainable_parameters = lambda: None
        return model

    monkeypatch.setattr(
        "agoge_forger.train.trainer.validate_lora_targets_exist",
        lambda model, lora: ["q_proj"],
    )
    monkeypatch.setattr("agoge_forger.train.trainer.get_peft_model", fake_get_peft_model)

    cfg = _config(revision=PINNED_REVISION)
    cfg.training.gradient_checkpointing = False
    cfg.quantization.load_in_4bit = False
    _prepare_peft_model(cfg, MagicMock())

    assert captured["revision"] == PINNED_REVISION


def test_prepare_peft_model_leaves_revision_none_when_unset(monkeypatch):
    captured: dict[str, object] = {}

    def fake_get_peft_model(model, peft_config, **kwargs):
        captured["revision"] = peft_config.revision
        model.print_trainable_parameters = lambda: None
        return model

    monkeypatch.setattr(
        "agoge_forger.train.trainer.validate_lora_targets_exist",
        lambda model, lora: ["q_proj"],
    )
    monkeypatch.setattr("agoge_forger.train.trainer.get_peft_model", fake_get_peft_model)

    cfg = _config(revision=None)
    cfg.training.gradient_checkpointing = False
    cfg.quantization.load_in_4bit = False
    _prepare_peft_model(cfg, MagicMock())

    assert captured["revision"] is None
