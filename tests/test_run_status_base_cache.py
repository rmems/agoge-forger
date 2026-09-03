"""Exact cached Hub snapshot validation for adapter base models."""

import json
from types import SimpleNamespace

import pytest
from test_run_status import (
    TINY_LLAMA_CONFIG,
    TINY_LLAMA_SHAPES,
    TINY_TOKENIZER,
    _safetensors_with_shapes,
)
from transformers.utils import hub as transformers_hub

import agoge_forger._run_status_base_model as base_model_validation
import agoge_forger._run_status_hub_cache as hub_cache
import agoge_forger._run_status_validation as validation

REVISION = "d3040b7c81a0a810fa13c6f392f3e304a0e121d5"
OTHER_REVISION = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
MODEL_ID = "cached-org/remote-base"


def _cached_snapshot(cache_dir, *, weights: bool):
    repository = cache_dir / "models--cached-org--remote-base"
    snapshot = repository / "snapshots" / REVISION
    blobs = repository / "blobs"
    refs = repository / "refs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    refs.mkdir()
    (refs / "main").write_text(REVISION)
    files = {
        "config.json": json.dumps(TINY_LLAMA_CONFIG).encode(),
        "tokenizer.json": json.dumps(TINY_TOKENIZER).encode(),
        "tokenizer_config.json": json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "eos_token": "<unk>",
            }
        ).encode(),
    }
    if weights:
        files["model.safetensors"] = _safetensors_with_shapes(TINY_LLAMA_SHAPES)
    for ordinal, (name, payload) in enumerate(files.items()):
        blob = blobs / f"blob-{ordinal}"
        blob.write_bytes(payload)
        (snapshot / name).symlink_to(blob)
    return snapshot


@pytest.mark.parametrize("weights", [False, True], ids=["missing-weights", "complete"])
def test_pinned_cached_base_requires_exact_weights_without_hub_access(
    monkeypatch, tmp_path, weights
):
    _cached_snapshot(tmp_path, weights=weights)
    monkeypatch.setattr(base_model_validation, "HF_HUB_CACHE", tmp_path, raising=False)
    monkeypatch.setattr(transformers_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    snapshot_download = hub_cache.snapshot_download
    snapshot_calls = []

    def resolve_cached_snapshot(repo_id, **kwargs):
        snapshot_calls.append((repo_id, kwargs))
        return snapshot_download(repo_id, **kwargs)

    monkeypatch.setattr(hub_cache, "snapshot_download", resolve_cached_snapshot)

    def deny_hub_access(*args, **kwargs):
        raise AssertionError("cached base validation attempted Hub access")

    monkeypatch.setattr("huggingface_hub.HfApi.repo_info", deny_hub_access)

    result = validation._adapter_base_config(
        SimpleNamespace(base_model_name_or_path=MODEL_ID, revision=REVISION)
    )

    assert (result is not None) is weights
    assert snapshot_calls == [
        (
            MODEL_ID,
            {
                "revision": REVISION,
                "cache_dir": tmp_path,
                "local_files_only": True,
            },
        )
    ]


@pytest.mark.parametrize("revision", [None, "main", "release-v1"])
def test_remote_base_rejects_mutable_revision_before_loading(monkeypatch, revision):
    def fail_load(*args, **kwargs):
        raise AssertionError("mutable remote revision reached artifact loading")

    monkeypatch.setattr(validation.AutoConfig, "from_pretrained", fail_load)
    monkeypatch.setattr(hub_cache, "snapshot_download", fail_load)

    result = validation._adapter_base_config(
        SimpleNamespace(base_model_name_or_path=MODEL_ID, revision=revision)
    )

    assert result is None


@pytest.mark.parametrize("revision", [None, "main", "release-v1"])
def test_cache_resolver_rejects_mutable_revision_without_loading(monkeypatch, tmp_path, revision):
    def fail_load(*args, **kwargs):
        raise AssertionError("mutable revision reached snapshot resolution")

    monkeypatch.setattr(hub_cache, "snapshot_download", fail_load)

    assert hub_cache.cached_snapshot(MODEL_ID, revision, tmp_path) is None


def test_pinned_base_rejects_cache_for_different_commit(monkeypatch, tmp_path):
    snapshot = _cached_snapshot(tmp_path, weights=True)
    other_snapshot = snapshot.parent / OTHER_REVISION
    snapshot.rename(other_snapshot)
    (snapshot.parent.parent / "refs" / "main").write_text(OTHER_REVISION)
    monkeypatch.setattr(base_model_validation, "HF_HUB_CACHE", tmp_path, raising=False)
    monkeypatch.setattr(transformers_hub.constants, "HF_HUB_CACHE", str(tmp_path))

    def deny_hub_access(*args, **kwargs):
        raise AssertionError("pinned base validation attempted Hub access")

    monkeypatch.setattr("huggingface_hub.HfApi.repo_info", deny_hub_access)

    result = validation._adapter_base_config(
        SimpleNamespace(base_model_name_or_path=MODEL_ID, revision=REVISION)
    )

    assert result is None
