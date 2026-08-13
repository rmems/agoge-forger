"""Transformers 5: merged-model save_pretrained kwargs must bind cleanly.

Spun out of #63 / PR #67 — see GH#68 / RM-229. PeftModel still accepts
``safe_serialization``; plain PreTrainedModel after merge_and_unload does not.
"""

from __future__ import annotations

import inspect

from transformers import PreTrainedModel

from agoge_forger.export.merge_adapter import merged_model_save_kwargs


def test_merged_model_save_kwargs_bind_to_pretrained_save_pretrained():
    kwargs = merged_model_save_kwargs(max_shard_size="4GB")
    assert "safe_serialization" not in kwargs

    sig = inspect.signature(PreTrainedModel.save_pretrained)
    # self + save_directory + our kwargs must bind without TypeError
    sig.bind(None, "/tmp/merged-out", **kwargs)

    # Document the TF5 regression surface: this name is no longer a real param.
    assert "safe_serialization" not in sig.parameters


def test_merged_model_save_kwargs_default_shard():
    assert merged_model_save_kwargs() == {"max_shard_size": "4GB"}
