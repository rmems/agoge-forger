"""Revision from ExperimentConfig must reach load_base_model.

These tests mock the trainer's Hub/GPU/dataset side effects and raise out of
``load_base_model`` so nothing downloads weights or constructs SFTTrainer.
"""

import pytest

from agoge_forger.config import ExperimentConfig
from agoge_forger.train.trainer import run_training

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
    monkeypatch.setattr("agoge_forger.train.trainer.get_gpu_report", lambda: {"device_name": "mock"})
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
