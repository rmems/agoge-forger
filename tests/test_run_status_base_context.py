"""Adapter base-model context readiness tests."""

import json

from test_run_status import TINY_LLAMA_CONFIG, TINY_LLAMA_SHAPES, _safetensors_with_shapes
from test_run_status_adapters import _make_run_dir, _write_final_adapter

from agoge_forger.run_status import build_run_status


def test_local_adapter_base_requires_usable_tokenizer(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "config.json").write_text(json.dumps(TINY_LLAMA_CONFIG))
    (base_model / "model.safetensors").write_bytes(_safetensors_with_shapes(TINY_LLAMA_SHAPES))
    _write_final_adapter(run_dir, base_model=str(base_model))

    assert build_run_status(str(run_dir))["export"]["ready"] is False
