import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from agoge_forger.artifacts.producer_provenance import (
    producer_provenance_from_adapter,
    producer_provenance_from_config,
    require_producer_provenance,
)
from agoge_forger.artifacts.safetensors_io import assert_no_unsafe_weight_bins, write_artifact_index
from agoge_forger.config import ExperimentConfig
from agoge_forger.eval import ArtifactProducerProvenance, verified_adapter_source
from agoge_forger.eval.contract import ArtifactValidationContext, require_artifact_index
from agoge_forger.split_contract import sha256_file
from agoge_forger.train.lora import train_lora
from agoge_forger.train.qlora import train_qlora
from agoge_forger.train.trainer import _finalize_training_run, _TrainingFinalization
from tests.evaluation_contract_cases import frozen_manifest, model_provenance
from tests.peft_adapter_fixtures import write_complete_adapter_model


def test_artifact_index_hashes_file(tmp_path):
    out_dir = tmp_path / "test_out"
    out_dir.mkdir()

    test_file = out_dir / "test.txt"
    test_file.write_text("hello world")

    index_path = write_artifact_index(str(out_dir))
    assert os.path.exists(index_path)

    with open(index_path) as f:
        data = json.load(f)
        assert data["output_dir"] == str(out_dir)
        assert len(data["artifacts"]) == 1
        assert data["artifacts"][0]["file"] == "test.txt"
        assert data["artifacts"][0]["size_bytes"] == 11
        assert data["artifacts"][0]["sha256"] != "unknown"
        assert "producer_provenance" not in data


def test_write_artifact_index_replaces_existing_index_atomically(tmp_path, monkeypatch):
    from agoge_forger import _atomic_file as atomic_file

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "first.bin").write_bytes(b"one")
    index_path = Path(write_artifact_index(str(out_dir)))
    previous = index_path.read_bytes()
    json.loads(previous)
    (out_dir / "second.bin").write_bytes(b"two")

    seen_before_replace: list[bytes] = []
    real_replace = atomic_file.os.replace

    def capturing_replace(src, dst, *args, **kwargs):
        if Path(dst) == index_path:
            snapshot = Path(dst).read_bytes()
            json.loads(snapshot)
            seen_before_replace.append(snapshot)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(atomic_file.os, "replace", capturing_replace)
    write_artifact_index(str(out_dir))

    assert seen_before_replace == [previous]
    data = json.loads(index_path.read_text())
    assert {item["file"] for item in data["artifacts"]} == {"first.bin", "second.bin"}
    assert "producer_provenance" not in data


def test_write_artifact_index_provenance_is_required_by_eval_validation(
    tmp_path, cached_test_base_config
):
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    write_complete_adapter_model(output_dir)

    provenance = ArtifactProducerProvenance.model_validate(model_provenance())
    context = ArtifactValidationContext(
        kind="peft_adapter",
        model_repository=provenance.base_model_name_or_path,
        model_revision=provenance.revision,
        split_manifest_sha256=provenance.training_split_manifest_sha256,
        train_split_sha256=provenance.training_split_sha256,
    )

    index_path = Path(write_artifact_index(str(output_dir), producer_provenance=provenance))
    require_artifact_index(index_path, sha256_file(index_path), context)

    index_path.unlink()
    index_path = Path(write_artifact_index(str(output_dir)))
    with pytest.raises(
        ValueError, match="peft_adapter artifact index requires producer_provenance"
    ):
        require_artifact_index(index_path, sha256_file(index_path), context)

    mismatched = replace(context, model_revision="9" * 40)
    index_path.unlink()
    index_path = Path(write_artifact_index(str(output_dir), producer_provenance=provenance))
    with pytest.raises(ValueError, match="does not match the contracted"):
        require_artifact_index(index_path, sha256_file(index_path), mismatched)


def test_no_bin_outputs_when_safe_serialization_required(tmp_path):
    out_dir = tmp_path / "safe_dir"
    out_dir.mkdir()

    bin_file = out_dir / "pytorch_model.bin"
    bin_file.touch()

    with pytest.raises(RuntimeError, match="Unsafe weight binaries found"):
        assert_no_unsafe_weight_bins(str(out_dir))


def test_assert_no_unsafe_weight_bins_ignores_trainer_state_pt(tmp_path):
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"trainer-state")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    assert_no_unsafe_weight_bins(str(tmp_path))


def test_assert_no_unsafe_weight_bins_detects_bin_in_checkpoint_subdir(tmp_path):
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.bin").write_bytes(b"unsafe")
    with pytest.raises(RuntimeError, match="Unsafe weight binaries found"):
        assert_no_unsafe_weight_bins(str(tmp_path))


def _frozen_train_config(tmp_path) -> ExperimentConfig:
    manifest_path, manifest = frozen_manifest(tmp_path)
    train_path = manifest_path.parent / manifest.splits["train"].path
    return ExperimentConfig(
        model_id="example/base-model",
        revision="abcdef0123456789abcdef0123456789abcdef01",
        dataset_path=str(train_path),
        output_dir=str(tmp_path / "adapters"),
        run_name="prov",
    )


def _peft_context(provenance: ArtifactProducerProvenance) -> ArtifactValidationContext:
    return ArtifactValidationContext(
        kind="peft_adapter",
        model_repository=provenance.base_model_name_or_path,
        model_revision=provenance.revision,
        split_manifest_sha256=provenance.training_split_manifest_sha256,
        train_split_sha256=provenance.training_split_sha256,
    )


def test_train_path_writes_index_that_passes_require_artifact_index(
    tmp_path, cached_test_base_config, monkeypatch
):
    config = _frozen_train_config(tmp_path)
    provenance = producer_provenance_from_config(config)
    output_dir = tmp_path / "adapters" / "prov"
    output_dir.mkdir(parents=True)
    write_complete_adapter_model(output_dir)

    trainer = MagicMock()
    trainer.model.save_pretrained.return_value = None
    trainer.processing_class.save_pretrained.return_value = None
    trainer.model.dtype = "float32"
    trainer.model.num_parameters.return_value = 1
    trainer.train_dataset = []
    monkeypatch.setattr("agoge_forger.train.trainer.torch.cuda.max_memory_allocated", lambda: 0)
    monkeypatch.setattr("agoge_forger.train.trainer.write_run_manifest", lambda *a, **k: None)

    _finalize_training_run(
        config,
        _TrainingFinalization(
            trainer=trainer,
            out_dir=str(output_dir),
            gpu_report={"device_name": "mock"},
            producer_provenance=provenance,
        ),
    )

    index_path = output_dir / "artifact_index.json"
    require_artifact_index(index_path, sha256_file(index_path), _peft_context(provenance))
    assert producer_provenance_from_adapter(output_dir) == provenance


def test_train_qlora_and_lora_pass_constructed_provenance(tmp_path, monkeypatch):
    config = _frozen_train_config(tmp_path)
    expected = producer_provenance_from_config(config)
    captured: dict[str, object] = {}

    def fake_run_training(cfg, producer_provenance=None):
        captured["cfg"] = cfg
        captured["provenance"] = producer_provenance

    monkeypatch.setattr("agoge_forger.train.qlora.run_training", fake_run_training)
    train_qlora(config)
    assert captured["provenance"] == expected

    monkeypatch.setattr("agoge_forger.train.lora.run_training", fake_run_training)
    train_lora(config)
    assert captured["provenance"] == expected
    assert config.quantization.load_in_4bit is False


def test_train_finalize_fails_closed_before_save_without_provenance(tmp_path):
    config = ExperimentConfig(
        model_id="example/base-model",
        dataset_path=str(tmp_path / "unused.jsonl"),
        run_name="missing_prov",
    )
    saved = {"model": False}

    class FakeModel:
        def save_pretrained(self, *args, **kwargs):
            saved["model"] = True

    trainer = MagicMock()
    trainer.model = FakeModel()
    with pytest.raises(ValueError, match="without ArtifactProducerProvenance"):
        _finalize_training_run(
            config,
            _TrainingFinalization(
                trainer=trainer,
                out_dir=str(tmp_path / "out"),
                gpu_report={"device_name": "mock"},
            ),
        )
    assert saved["model"] is False


def test_producer_provenance_from_config_fails_closed_without_freeze_metadata(tmp_path):
    dataset = tmp_path / "plain.jsonl"
    dataset.write_text('{"text": "hello"}\n')
    config = ExperimentConfig(
        model_id="example/base-model",
        revision="abcdef0123456789abcdef0123456789abcdef01",
        dataset_path=str(dataset),
    )
    with pytest.raises(ValueError, match="cannot construct producer_provenance"):
        producer_provenance_from_config(config)


def test_producer_provenance_from_config_fails_closed_without_revision_digest(tmp_path):
    config = _frozen_train_config(tmp_path)
    config = config.model_copy(update={"revision": None})
    with pytest.raises(ValueError, match="cannot construct producer_provenance"):
        producer_provenance_from_config(config)


def test_require_producer_provenance_rejects_mismatched_sealed_identity(tmp_path):
    config = _frozen_train_config(tmp_path)
    sealed = producer_provenance_from_config(config)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    write_complete_adapter_model(adapter)
    write_artifact_index(str(adapter), producer_provenance=sealed)

    mismatched = sealed.model_copy(update={"revision": "9" * 40})
    with pytest.raises(ValueError, match="does not match the adapter artifact identity"):
        require_producer_provenance(mismatched, adapter)

    assert require_producer_provenance(sealed, adapter) == sealed
    assert require_producer_provenance(None, adapter) == sealed


def test_verified_adapter_source_requires_caller_supplied_identity(
    tmp_path, cached_test_base_config
):
    config = _frozen_train_config(tmp_path)
    sealed = producer_provenance_from_config(config)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    write_complete_adapter_model(adapter)
    write_artifact_index(str(adapter), producer_provenance=sealed)

    with (
        pytest.raises(ValueError, match="does not match the contracted"),
        verified_adapter_source(
            adapter,
            sealed.base_model_name_or_path,
            sealed.revision,
            split_manifest_sha256="a" * 64,
            train_split_sha256=sealed.training_split_sha256,
        ),
    ):
        pass

    with verified_adapter_source(
        adapter,
        sealed.base_model_name_or_path,
        sealed.revision,
        split_manifest_sha256=sealed.training_split_manifest_sha256,
        train_split_sha256=sealed.training_split_sha256,
    ) as source:
        assert source.provenance == sealed


def test_artifact_validation_accepts_unknown_adapter_config_keys(tmp_path, cached_test_base_config):
    config = _frozen_train_config(tmp_path)
    provenance = producer_provenance_from_config(config)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    write_complete_adapter_model(adapter)
    payload = json.loads((adapter / "adapter_config.json").read_text())
    payload["future_peft_flag"] = True
    (adapter / "adapter_config.json").write_text(json.dumps(payload))
    index_path = Path(write_artifact_index(str(adapter), producer_provenance=provenance))
    require_artifact_index(index_path, sha256_file(index_path), _peft_context(provenance))


def test_artifact_validation_accepts_saved_embedding_layers(tmp_path, cached_test_base_config):
    config = _frozen_train_config(tmp_path)
    provenance = producer_provenance_from_config(config)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    base.name_or_path = provenance.base_model_name_or_path
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            revision=provenance.revision,
        ),
    )
    peft_model.save_pretrained(adapter, save_embedding_layers=True)
    index_path = Path(write_artifact_index(str(adapter), producer_provenance=provenance))
    require_artifact_index(index_path, sha256_file(index_path), _peft_context(provenance))
