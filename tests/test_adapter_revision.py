"""Pinned Hub revision must survive on the adapter and reach eval/export loaders."""

import json

import pytest

from agoge_forger.eval.smoke_eval import run_smoke_eval
from agoge_forger.export.merge_adapter import export_final_model, merge_adapter
from agoge_forger.train.checkpoints import infer_base_revision_from_adapter

PINNED_REVISION = "d3040b7c81a0a810fa13c6f392f3e304a0e121d5"
BASE_MODEL = "HuggingFaceM4/tiny-random-LlamaForCausalLM"


def _write_adapter(path, *, revision=PINNED_REVISION, base_model=BASE_MODEL):
    path.mkdir(parents=True)
    payload = {"base_model_name_or_path": base_model}
    if revision is not False:
        payload["revision"] = revision
    (path / "adapter_config.json").write_text(json.dumps(payload))
    (path / "adapter_model.safetensors").write_text("weights")
    return path


def test_infer_base_revision_from_adapter(tmp_path):
    pinned = _write_adapter(tmp_path / "pinned")
    assert infer_base_revision_from_adapter(pinned) == PINNED_REVISION

    omitted = _write_adapter(tmp_path / "omitted", revision=False)
    assert infer_base_revision_from_adapter(omitted) is None

    empty = _write_adapter(tmp_path / "empty", revision="")
    assert infer_base_revision_from_adapter(empty) is None


def test_smoke_eval_forwards_adapter_revision_to_load_base_model(monkeypatch, tmp_path):
    adapter = _write_adapter(tmp_path / "adapter")
    captured: dict[str, object] = {}

    def fake_load_base_model(*args, **kwargs):
        captured["revision"] = kwargs.get("revision")
        raise RuntimeError("stop-before-eval")

    monkeypatch.setattr("agoge_forger.eval.smoke_eval.load_base_model", fake_load_base_model)

    with pytest.raises(RuntimeError, match="stop-before-eval"):
        run_smoke_eval(BASE_MODEL, str(adapter))

    assert captured["revision"] == PINNED_REVISION


def test_smoke_eval_forwards_none_when_adapter_has_no_revision(monkeypatch, tmp_path):
    adapter = _write_adapter(tmp_path / "adapter", revision=False)
    captured: dict[str, object] = {}

    def fake_load_base_model(*args, **kwargs):
        captured["revision"] = kwargs.get("revision")
        raise RuntimeError("stop-before-eval")

    monkeypatch.setattr("agoge_forger.eval.smoke_eval.load_base_model", fake_load_base_model)

    with pytest.raises(RuntimeError, match="stop-before-eval"):
        run_smoke_eval(BASE_MODEL, str(adapter))

    assert captured["revision"] is None


def test_merge_adapter_forwards_adapter_revision_to_load_base_model(monkeypatch, tmp_path):
    adapter = _write_adapter(tmp_path / "adapter")
    captured: dict[str, object] = {}

    def fake_load_base_model(*args, **kwargs):
        captured["revision"] = kwargs.get("revision")
        raise RuntimeError("stop-before-merge")

    monkeypatch.setattr("agoge_forger.export.merge_adapter.load_base_model", fake_load_base_model)

    with pytest.raises(RuntimeError, match="stop-before-merge"):
        merge_adapter(BASE_MODEL, str(adapter), str(tmp_path / "merged"))

    assert captured["revision"] == PINNED_REVISION


def test_export_final_model_forwards_adapter_revision_to_load_base_model(monkeypatch, tmp_path):
    adapter = _write_adapter(tmp_path / "adapter")
    captured: dict[str, object] = {}

    def fake_load_base_model(*args, **kwargs):
        captured["revision"] = kwargs.get("revision")
        raise RuntimeError("stop-before-export")

    monkeypatch.setattr("agoge_forger.export.merge_adapter.load_base_model", fake_load_base_model)

    with pytest.raises(RuntimeError, match="stop-before-export"):
        export_final_model(out_dir=str(tmp_path / "merged"), adapter_path=str(adapter))

    assert captured["revision"] == PINNED_REVISION
