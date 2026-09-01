"""Revision from ExperimentConfig must reach load_base_model.

These tests mock the trainer's Hub/GPU/dataset side effects and raise out of
``load_base_model`` so nothing downloads weights or constructs SFTTrainer.
"""

from unittest.mock import MagicMock

import pytest

from agoge_forger.config import ExperimentConfig
from agoge_forger.split_loaders import FrozenSplitBinding
from agoge_forger.train.trainer import (
    _bind_frozen_training_input,
    _frozen_producer_provenance,
    _prepare_peft_model,
    run_training,
)

PINNED_REVISION = "d3040b7c81a0a810fa13c6f392f3e304a0e121d5"


def _config(*, revision: str | None) -> ExperimentConfig:
    return ExperimentConfig(
        model_id="org/model",
        revision=revision,
        dataset_path="unused.jsonl",
        run_name="revision_pass_through",
    )


def _frozen_config(manifest_path, **updates) -> ExperimentConfig:
    values = {
        "model_id": "org/model",
        "revision": PINNED_REVISION,
        "split_manifest_path": str(manifest_path),
        "split_name": "train",
    }
    values.update(updates)
    return ExperimentConfig(**values)


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


@pytest.mark.parametrize("revision", [PINNED_REVISION, None])
def test_run_training_passes_revision_to_load_base_model(monkeypatch, revision):
    captured: dict[str, object] = {}

    def fake_load_base_model(*args, **kwargs):
        captured["kwargs"] = kwargs
        raise RuntimeError("stop-before-trainer")

    _patch_preflight(monkeypatch)
    monkeypatch.setattr("agoge_forger.train.trainer.load_base_model", fake_load_base_model)

    with pytest.raises(RuntimeError, match="stop-before-trainer"):
        run_training(_config(revision=revision))

    assert captured["kwargs"]["revision"] == revision


@pytest.mark.parametrize("revision", [PINNED_REVISION, None])
def test_prepare_peft_model_persists_revision_on_lora_config(monkeypatch, revision):
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

    cfg = _config(revision=revision)
    cfg.training.gradient_checkpointing = False
    cfg.quantization.load_in_4bit = False
    _prepare_peft_model(cfg, MagicMock())

    assert captured["revision"] == revision


def test_frozen_training_provenance_uses_exact_bound_digests(tmp_path, monkeypatch):
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text("{}\n")
    binding = FrozenSplitBinding(
        manifest_path=manifest_path,
        manifest=MagicMock(),
        manifest_sha256="b" * 64,
        split="train",
        split_sha256="c" * 64,
    )
    config = _frozen_config(manifest_path)
    monkeypatch.setattr("agoge_forger.train.trainer.bind_frozen_split", lambda *args: binding)

    actual_binding = _bind_frozen_training_input(config)
    assert actual_binding is not None
    provenance = _frozen_producer_provenance(config, actual_binding)

    assert provenance.model_dump(mode="json") == {
        "base_model_name_or_path": "org/model",
        "revision": PINNED_REVISION,
        "training_split_manifest_sha256": "b" * 64,
        "training_split_name": "train",
        "training_split_sha256": "c" * 64,
    }


def test_frozen_training_rejects_unpinned_revision_before_binding(tmp_path, monkeypatch):
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text("{}\n")
    config = _frozen_config(manifest_path, revision="main")
    bind = MagicMock()
    monkeypatch.setattr("agoge_forger.train.trainer.bind_frozen_split", bind)

    with pytest.raises(ValueError, match="immutable lowercase commit revision"):
        _bind_frozen_training_input(config)

    bind.assert_not_called()


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        ({"dataset_text_field": "body"}, "dataset_text_field: text"),
        ({"model_id": r"C:\\models\\mutable"}, "Hub model repository"),
        ({"model_id": r"\\server\\share\\mutable"}, "Hub model repository"),
    ],
)
def test_frozen_training_rejects_nonportable_inputs_before_binding(
    tmp_path, monkeypatch, updates, expected_error
):
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text("{}\n")
    bind = MagicMock()
    monkeypatch.setattr("agoge_forger.train.trainer.bind_frozen_split", bind)

    with pytest.raises(ValueError, match=expected_error):
        _bind_frozen_training_input(_frozen_config(manifest_path, **updates))

    bind.assert_not_called()


@pytest.mark.parametrize("local_kind", ["directory", "broken-symlink"])
def test_frozen_training_rejects_existing_local_model_before_binding(
    tmp_path, monkeypatch, local_kind
):
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text("{}\n")
    model_path = tmp_path / "mutable-model"
    if local_kind == "directory":
        model_path.mkdir()
    else:
        model_path.symlink_to(tmp_path / "missing-model")
    bind = MagicMock()
    monkeypatch.setattr("agoge_forger.train.trainer.bind_frozen_split", bind)

    with pytest.raises(ValueError, match="Hub model repository"):
        _bind_frozen_training_input(_frozen_config(manifest_path, model_id=str(model_path)))

    bind.assert_not_called()
