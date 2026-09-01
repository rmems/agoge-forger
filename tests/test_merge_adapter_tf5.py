"""Transformers 5: merged-model save_pretrained kwargs must bind cleanly.

Spun out of #63 / PR #67 — see GH#68 / RM-229. PeftModel still accepts
``safe_serialization``; plain PreTrainedModel after merge_and_unload does not.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedModel

from agoge_forger.artifacts.safetensors_io import write_artifact_index
from agoge_forger.eval import _artifact_validation
from agoge_forger.eval._artifact_schema import ArtifactProducerProvenance
from agoge_forger.export.merge_adapter import (
    _merge_source,
    _MergeSource,
    merge_adapter,
    merged_model_save_kwargs,
)
from tests.evaluation_contract_cases import write_safetensors


def test_merged_model_save_kwargs_bind_to_pretrained_save_pretrained():
    kwargs = merged_model_save_kwargs(max_shard_size="4GB")
    assert "safe_serialization" not in kwargs

    sig = inspect.signature(PreTrainedModel.save_pretrained)
    # **kwargs on save_pretrained would accept unknown names; require membership.
    unexpected = set(kwargs) - set(sig.parameters)
    assert not unexpected, f"Unsupported save_pretrained kwargs: {unexpected}"
    # self + save_directory + our kwargs must bind without TypeError
    sig.bind(None, "merged-out", **kwargs)

    # Document the TF5 regression surface: this name is no longer a real param.
    assert "safe_serialization" not in sig.parameters


def test_merged_model_save_kwargs_default_shard():
    assert merged_model_save_kwargs() == {"max_shard_size": "4GB"}


def test_merge_adapter_rejects_save_safetensors_false():
    """False must not silently no-op: TF5 cannot emit legacy .bin merged weights."""
    with pytest.raises(ValueError, match="save_safetensors=False is not supported"):
        merge_adapter(
            "dummy/base",
            "/nonexistent/adapter",
            "/nonexistent/out",
            save_safetensors=False,
        )


def test_merge_uses_verified_snapshot_and_propagates_provenance(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    provenance = {
        "base_model_name_or_path": "example/base",
        "revision": "a" * 40,
        "training_split_manifest_sha256": "b" * 64,
        "training_split_name": "train",
        "training_split_sha256": "c" * 64,
    }
    (adapter / "adapter_config.json").write_text("{}\n")
    write_safetensors(adapter / "adapter_model.safetensors")
    write_artifact_index(str(adapter), producer_provenance=provenance)
    monkeypatch.setattr(
        _artifact_validation,
        "_require_peft_adapter_structure",
        lambda indexed, context: {},
    )

    with _merge_source(str(adapter), "example/base", None, False) as source:
        snapshot = source.root
        assert snapshot != adapter
        assert (snapshot / "adapter_model.safetensors").read_bytes() == (
            adapter / "adapter_model.safetensors"
        ).read_bytes()
        assert source.provenance is not None
        assert source.provenance.model_dump(mode="json") == provenance

    assert not snapshot.exists()


@pytest.mark.parametrize(
    ("include_index", "expected_error"),
    [
        (False, "artifact_index.json is required"),
        (True, "requires producer_provenance"),
    ],
)
def test_merge_requires_verified_index_before_model_loading(
    tmp_path, monkeypatch, include_index, expected_error
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n")
    write_safetensors(adapter / "adapter_model.safetensors")
    if include_index:
        write_artifact_index(str(adapter))
    observed = []
    monkeypatch.setattr(
        "agoge_forger.export.merge_adapter.load_base_model",
        lambda *args, **kwargs: observed.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match=expected_error):
        merge_adapter("example/base", str(adapter), str(tmp_path / "merged"))

    assert observed == []
    assert not (tmp_path / "merged").exists()


def test_merge_loads_pinned_base_before_offline_tensor_schema(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n")
    write_safetensors(adapter / "adapter_model.safetensors")
    provenance = ArtifactProducerProvenance(
        base_model_name_or_path="example/base",
        revision="a" * 40,
        training_split_manifest_sha256="b" * 64,
        training_split_name="train",
        training_split_sha256="c" * 64,
    )
    verified = SimpleNamespace(provenance=provenance)
    events = []

    @contextmanager
    def source(*args):
        yield _MergeSource(adapter, provenance, verified)

    monkeypatch.setattr("agoge_forger.export.merge_adapter._merge_source", source)
    monkeypatch.setattr(
        "agoge_forger.export.merge_adapter.load_base_model",
        lambda *args, **kwargs: (events.append("base-loaded") or object(), object()),
    )
    monkeypatch.setattr(
        "agoge_forger.export.merge_adapter.require_adapter_source_tensor_schema",
        lambda *args: events.append("tensor-schema"),
    )
    from_pretrained = MagicMock(side_effect=RuntimeError("stop-after-schema"))
    monkeypatch.setattr(
        "agoge_forger.export.merge_adapter.PeftModel.from_pretrained", from_pretrained
    )

    with pytest.raises(RuntimeError, match="stop-after-schema"):
        merge_adapter("example/base", str(adapter), str(tmp_path / "merged"))

    assert events == ["base-loaded", "tensor-schema"]


def test_explicit_unsafe_merge_bypasses_index_provenance(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "artifact_index.json").write_text("not valid JSON")

    with _merge_source(str(adapter), "example/base", None, True) as source:
        assert source.root == adapter
        assert source.provenance is None
        assert source.verified is None
