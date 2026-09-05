"""Transformers 5: merged-model save_pretrained kwargs must bind cleanly.

Spun out of #63 / PR #67 — see GH#68 / RM-229. PeftModel still accepts
``safe_serialization``; plain PreTrainedModel after merge_and_unload does not.
"""

from __future__ import annotations

import inspect

import pytest
from transformers import PreTrainedModel

from agoge_forger.export.merge_adapter import merge_adapter, merged_model_save_kwargs


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


def test_merge_adapter_fails_closed_before_load_without_provenance(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "dummy/base", "revision": "d" * 40}'
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    loaded: list[object] = []

    def fake_load_base_model(*args, **kwargs):
        loaded.append(True)
        raise RuntimeError("should-not-load")

    monkeypatch.setattr("agoge_forger.export.merge_adapter.load_base_model", fake_load_base_model)

    with pytest.raises(ValueError, match="cannot construct producer_provenance"):
        merge_adapter(
            "dummy/base",
            str(adapter),
            str(tmp_path / "merged"),
            allow_unsafe=True,
        )

    assert loaded == []
    assert not (tmp_path / "merged").exists()
