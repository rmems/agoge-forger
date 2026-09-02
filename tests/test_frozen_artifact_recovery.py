import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agoge_forger.config import ExperimentConfig
from agoge_forger.train import recover
from agoge_forger.train.recover import recover_frozen_artifact_index
from tests.evaluation_contract_cases import write_safetensors

REVISION = "a" * 40


def _config(tmp_path, **updates):
    values = {
        "model_id": "example/base",
        "revision": REVISION,
        "split_manifest_path": str(tmp_path / "split_manifest.json"),
        "split_name": "train",
        "output_dir": str(tmp_path / "adapters"),
        "run_name": "frozen",
    }
    values.update(updates)
    return ExperimentConfig(**values)


def _binding():
    return SimpleNamespace(
        manifest_sha256="b" * 64,
        split_sha256="c" * 64,
    )


def _completed_run(config):
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True)
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": config.model_id,
                "revision": config.revision,
            }
        )
    )
    write_safetensors(run_dir / "adapter_model.safetensors")
    return run_dir


def _patch_binding(monkeypatch):
    monkeypatch.setattr(recover, "_bind_frozen_source", lambda config: _binding())


def test_recovers_absent_index_with_exact_provenance(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run_dir = _completed_run(config)
    _patch_binding(monkeypatch)

    index_path = recover_frozen_artifact_index(config)

    payload = json.loads(index_path.read_text())
    assert payload["producer_provenance"] == {
        "base_model_name_or_path": "example/base",
        "revision": REVISION,
        "training_split_manifest_sha256": "b" * 64,
        "training_split_name": "train",
        "training_split_sha256": "c" * 64,
    }
    assert {entry["file"] for entry in payload["artifacts"]} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    assert index_path == run_dir / "artifact_index.json"


def test_quarantines_truncated_index_before_recovery(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run_dir = _completed_run(config)
    damaged = run_dir / "artifact_index.json"
    damaged.write_bytes(b'{"truncated":')
    _patch_binding(monkeypatch)
    monkeypatch.setattr(recover.secrets, "token_hex", lambda size: "evidence")

    recover_frozen_artifact_index(config)

    quarantine = run_dir.parent / ".frozen.artifact_index.invalid.evidence"
    assert quarantine.read_bytes() == b'{"truncated":'
    assert json.loads(damaged.read_text())["producer_provenance"]["revision"] == REVISION


def test_refuses_schema_valid_existing_index(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run_dir = _completed_run(config)
    _patch_binding(monkeypatch)
    recover_frozen_artifact_index(config)
    original = (run_dir / "artifact_index.json").read_bytes()

    with pytest.raises(FileExistsError, match="schema-valid"):
        recover_frozen_artifact_index(config)

    assert (run_dir / "artifact_index.json").read_bytes() == original


@pytest.mark.parametrize(
    "case",
    [
        ("base_model_name_or_path", "other/base", "base model"),
        ("revision", "d" * 40, "revision"),
    ],
)
def test_refuses_adapter_identity_mismatch(tmp_path, monkeypatch, case):
    field, value, message = case
    config = _config(tmp_path)
    run_dir = _completed_run(config)
    adapter_config = json.loads((run_dir / "adapter_config.json").read_text())
    adapter_config[field] = value
    (run_dir / "adapter_config.json").write_text(json.dumps(adapter_config))
    _patch_binding(monkeypatch)

    with pytest.raises(ValueError, match=message):
        recover_frozen_artifact_index(config)

    assert not (run_dir / "artifact_index.json").exists()


def test_refuses_malformed_adapter_safetensors(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run_dir = _completed_run(config)
    (run_dir / "adapter_model.safetensors").write_bytes(b"not safetensors")
    _patch_binding(monkeypatch)

    with pytest.raises(ValueError, match="valid adapter safetensors"):
        recover_frozen_artifact_index(config)

    assert not (run_dir / "artifact_index.json").exists()


def test_concurrent_index_publisher_wins_after_quarantine(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run_dir = _completed_run(config)
    damaged = run_dir / "artifact_index.json"
    damaged.write_bytes(b"truncated")
    _patch_binding(monkeypatch)
    monkeypatch.setattr(recover.secrets, "token_hex", lambda size: "evidence")
    winner = b'{"winner":true}'

    def publish_winner(*args, **kwargs):
        damaged.write_bytes(winner)
        raise FileExistsError("concurrent publisher won")

    monkeypatch.setattr(recover, "write_artifact_index_noreplace", publish_winner)

    with pytest.raises(FileExistsError, match="concurrent publisher won"):
        recover_frozen_artifact_index(config)

    assert damaged.read_bytes() == winner
    quarantine = run_dir.parent / ".frozen.artifact_index.invalid.evidence"
    assert quarantine.read_bytes() == b"truncated"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"runtime": {"allow_unsafe_serialization": True}}, "unsafe serialization disabled"),
        (
            {"training": {"resume_from_latest_checkpoint": True}},
            "frozen training resume is disabled",
        ),
    ],
)
def test_recovery_rejects_unsafe_or_resume_config_before_binding(
    tmp_path, monkeypatch, updates, message
):
    config = _config(tmp_path, **updates)
    observed = []
    monkeypatch.setattr(recover, "_bind_frozen_source", lambda config: observed.append(config))

    with pytest.raises(ValueError, match=message):
        recover_frozen_artifact_index(config)

    assert observed == []
