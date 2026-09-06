"""Merged-model and serialization readiness tests for run-status."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from agoge_forger._run_status_artifact_index import artifact_index_usable
from agoge_forger.artifacts.safetensors_io import write_artifact_index
from agoge_forger.cli import app
from agoge_forger.run_status import build_run_status, find_merged_model_dir, is_merged_model_dir
from tests.test_run_status import (
    TINY_LLAMA_CONFIG,
    TINY_LLAMA_SHAPES,
    TINY_TOKENIZER,
    _make_run_dir,
    _minimal_safetensors,
    _safetensors_with_dtype,
    _safetensors_with_shapes,
    _safetensors_with_tensors,
    _write_checkpoint,
    _write_final_adapter,
    _write_merged_model,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# 6. Merged model discovery
# --------------------------------------------------------------------------


def test_merged_model_found_in_conventional_sibling_layout(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)

    report = build_run_status(str(run_dir))

    assert report["merged_model"] == {"present": True, "path": str(merged.resolve())}
    assert find_merged_model_dir(run_dir.resolve()) == merged.resolve()


def test_conventional_merged_path_is_absolute_from_relative_run_dir(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    monkeypatch.chdir(tmp_path)

    report = build_run_status(f"adapters/{run_dir.name}")

    path = report["merged_model"]["path"]
    assert report["merged_model"]["present"] is True
    assert Path(path).is_absolute()
    assert path == str(merged.resolve())


def test_merged_model_absent_when_never_exported(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)

    report = build_run_status(str(run_dir))

    assert report["merged_model"] == {"present": False, "path": None}
    assert find_merged_model_dir(run_dir.resolve()) is None


def test_merged_dir_without_config_json_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "config.json").unlink()
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


@pytest.mark.parametrize("weight_name", [None, "adapter_model.safetensors"])
def test_non_model_weights_are_not_a_merged_model(tmp_path, weight_name):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "model.safetensors").unlink()
    if weight_name is not None:
        (merged / weight_name).write_bytes(_minimal_safetensors())
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_merged_dir_with_only_nested_safetensors_is_not_a_merged_model(tmp_path):
    """A tree holding adapters further down is not an exported merged model.

    A merged model keeps its weights at the directory root; a run tree that only
    contains `checkpoint-N/adapter_model.safetensors` must not read as already
    merged, or an operator would skip an export that never happened.
    """
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "model.safetensors").unlink()
    _write_checkpoint(merged, 10)
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


@pytest.mark.parametrize("shard_count", [1, 2], ids=["shared-shard", "multiple-shards"])
def test_sharded_merged_model_is_recognised(tmp_path, shard_count):
    """Indexed exports allow distinct shards and tensors sharing a shard."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = tmp_path / "merged" / run_dir.name
    merged.mkdir(parents=True)
    (merged / "config.json").write_text(json.dumps(TINY_LLAMA_CONFIG))
    (merged / "tokenizer.json").write_text(json.dumps(TINY_TOKENIZER))
    (merged / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "eos_token": "<unk>",
            }
        )
    )
    shard_names = [
        f"model-{ordinal:05d}-of-{shard_count:05d}.safetensors"
        for ordinal in range(1, shard_count + 1)
    ]
    shard_tensors = {name: {} for name in shard_names}
    weight_map = {}
    for index, (tensor_name, shape) in enumerate(TINY_LLAMA_SHAPES.items()):
        shard_name = shard_names[index % shard_count]
        shard_tensors[shard_name][tensor_name] = shape
        weight_map[tensor_name] = shard_name
    (merged / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard_name, tensor_shapes in shard_tensors.items():
        (merged / shard_name).write_bytes(_safetensors_with_shapes(tensor_shapes))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is True
    assert build_run_status(str(run_dir))["merged_model"] == {
        "present": True,
        "path": str(merged.resolve()),
    }


def test_deeply_nested_shard_index_fails_closed(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "model.safetensors").rename(merged / "model-00001-of-00001.safetensors")
    nested = '{"weight_map":' + "[" * 100_000 + "{}" + "]" * 100_000 + "}"
    (merged / "model.safetensors.index.json").write_text(nested)
    write_artifact_index(str(merged))

    result = runner.invoke(app, ["run-status", str(run_dir)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {"present": False, "path": None}


def test_shard_index_must_not_be_a_symlink(tmp_path):
    from agoge_forger._run_status_safetensors import _load_shard_weight_map

    merged = tmp_path / "merged"
    merged.mkdir()
    external = tmp_path / "external-index.json"
    external.write_text(json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}))
    (merged / "model.safetensors.index.json").symlink_to(external)

    assert _load_shard_weight_map(merged) is None


def test_shard_index_has_a_bounded_metadata_size(tmp_path):
    from agoge_forger._run_status_safetensors import _load_shard_weight_map

    merged = tmp_path / "merged"
    merged.mkdir()
    payload = {
        "weight_map": {"weight": "model-00001-of-00001.safetensors"},
        "padding": "x" * (4 * 1024 * 1024),
    }
    (merged / "model.safetensors.index.json").write_text(json.dumps(payload))

    assert _load_shard_weight_map(merged) is None


def test_shard_index_tensor_must_exist_in_designated_shard(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "tokenizer_config.json").write_text("{}")
    shard_name = "model-00001-of-00001.safetensors"
    (merged / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"expected": shard_name}})
    )
    (merged / shard_name).write_bytes(_safetensors_with_dtype("F32"))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_shard_index_must_cover_every_serialized_tensor(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").unlink()
    shard_name = "model-00001-of-00001.safetensors"
    (merged / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shard_name}})
    )
    (merged / shard_name).write_bytes(_safetensors_with_tensors("a", "b"))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_shard_index_must_cover_every_physical_shard(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").unlink()
    indexed = "model-00001-of-00001.safetensors"
    extra = "model-00002-of-00002.safetensors"
    (merged / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": indexed}}))
    (merged / indexed).write_bytes(_safetensors_with_tensors("a"))
    (merged / extra).write_bytes(_safetensors_with_tensors("b"))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_numbered_shard_series_must_be_complete(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").unlink()
    shard_name = "model-00001-of-00002.safetensors"
    (merged / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard_name}})
    )
    (merged / shard_name).write_bytes(_safetensors_with_tensors("weight"))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_absurd_shard_total_is_rejected_without_expanding_range(monkeypatch):
    from agoge_forger import _run_status_safetensors as module

    def fail_range(*args):
        raise AssertionError("must not materialize the attacker-controlled total")

    monkeypatch.setattr(module, "range", fail_range, raising=False)

    assert module._numbered_shards_complete({"model-00001-of-999999999.safetensors"}) is False


def test_truncated_merged_config_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "config.json").write_text("{not json")
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_incomplete_shard_set_is_not_a_merged_model(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "model.safetensors").unlink()
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    weight_map = {
        tensor_name: shard_names[index % 2] for index, tensor_name in enumerate(TINY_LLAMA_SHAPES)
    }
    (merged / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    first_shapes = {
        name: shape
        for name, shape in TINY_LLAMA_SHAPES.items()
        if weight_map[name] == shard_names[0]
    }
    (merged / "model-00001-of-00002.safetensors").write_bytes(
        _safetensors_with_shapes(first_shapes)
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_shard_index_rejects_adapter_filename(tmp_path):
    """weight_map must name root-local model shards, not leftover adapter files."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "model.safetensors").unlink()
    (merged / "adapter_model.safetensors").write_bytes(_safetensors_with_shapes(TINY_LLAMA_SHAPES))
    (merged / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {name: "adapter_model.safetensors" for name in TINY_LLAMA_SHAPES}}
        )
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_shard_index_rejects_out_of_directory_filename(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "model.safetensors").write_bytes(_safetensors_with_shapes(TINY_LLAMA_SHAPES))
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "model.safetensors").unlink()
    rel = os.path.relpath(elsewhere / "model.safetensors", merged)
    (merged / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(TINY_LLAMA_SHAPES, rel)})
    )
    write_artifact_index(str(merged))

    assert "/" in rel or rel.startswith("..")
    assert is_merged_model_dir(merged) is False
    assert build_run_status(str(run_dir))["merged_model"] == {"present": False, "path": None}


def test_truncated_merged_weights_are_not_present(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    (merged / "model.safetensors").write_text("merged-weights")
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_model_requires_completed_tokenizer_export(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "tokenizer.json").unlink()
    (merged / "tokenizer_config.json").unlink()
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_model_requires_usable_tokenizer_inventory(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "tokenizer.json").unlink()
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_model_tokenizer_requires_pad_or_eos_token(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>"})
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_model_requires_final_artifact_index(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "artifact_index.json").unlink(missing_ok=True)

    assert is_merged_model_dir(merged) is False


def test_stale_artifact_index_is_not_a_completion_marker(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").write_bytes(_safetensors_with_tensors("replacement"))

    assert is_merged_model_dir(merged) is False


def test_deeply_nested_artifact_index_fails_closed(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    merged = _write_merged_model(tmp_path / "merged" / run_dir.name)
    nested = '{"artifacts":' + "[" * 100_000 + "{}" + "]" * 100_000 + "}"
    (merged / "artifact_index.json").write_text(nested)

    result = runner.invoke(app, ["run-status", str(run_dir)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {"present": False, "path": None}


def test_oversized_artifact_index_is_rejected_before_read(tmp_path, monkeypatch):
    merged = _write_merged_model(tmp_path / "merged")
    index = merged / "artifact_index.json"
    index.write_bytes(b" " * (4 * 1024 * 1024 + 1))

    def unexpected_read(*args, **kwargs):
        raise AssertionError("oversized artifact index was read")

    monkeypatch.setattr(Path, "read_text", unexpected_read)

    assert artifact_index_usable(merged) is False


def test_merged_config_requires_model_type(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "config.json").write_text("{}")
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_config_requires_recognized_local_model_type(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "config.json").write_text('{"model_type": "not-a-real-model"}')
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_config_requires_causal_lm_model_type(tmp_path):
    from agoge_forger._run_status_validation import _local_causal_lm_config

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type": "vit"}')

    assert _local_causal_lm_config(merged) is None


def test_merged_weights_must_match_complete_local_architecture(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").write_bytes(
        _safetensors_with_shapes({"unrelated.weight": (8, 8)})
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_merged_weight_shapes_must_match_local_architecture(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    wrong_shapes = dict(TINY_LLAMA_SHAPES)
    wrong_shapes["model.norm.weight"] = (7,)
    (merged / "model.safetensors").write_bytes(_safetensors_with_shapes(wrong_shapes))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_unsharded_model_rejects_simultaneous_shard_index(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_unsharded_model_rejects_stale_numbered_shard(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model-00001-of-00001.safetensors").write_bytes(
        _safetensors_with_shapes(TINY_LLAMA_SHAPES)
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_unsharded_model_rejects_extra_root_model_safetensors(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "consolidated.safetensors").write_bytes(
        _safetensors_with_shapes({"foreign.weight": (1,)})
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


@pytest.mark.parametrize(
    ("model_type", "multiplicity_key"),
    [("llama", "num_hidden_layers"), ("bart", "decoder_layers")],
)
def test_huge_architecture_is_rejected_before_meta_model_construction(
    tmp_path, monkeypatch, model_type, multiplicity_key
):
    from agoge_forger import _run_status_validation as validation

    merged = tmp_path / "merged"
    merged.mkdir()
    config_payload = (
        dict(TINY_LLAMA_CONFIG) if model_type == "llama" else {"model_type": model_type}
    )
    config_payload[multiplicity_key] = 1_000_000_000
    (merged / "config.json").write_text(json.dumps(config_payload))
    config = validation._local_causal_lm_config(merged)

    def fail_construction(*args, **kwargs):
        raise AssertionError("untrusted dimensions must be bounded before model construction")

    monkeypatch.setattr(validation.AutoModelForCausalLM, "from_config", fail_construction)

    assert validation._causal_lm_shapes(config) is None


def test_multiplicative_architecture_is_rejected_before_meta_model_construction(
    tmp_path, monkeypatch
):
    from agoge_forger import _run_status_validation as validation

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mixtral",
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 4_096,
                "num_local_experts": 4_096,
            }
        )
    )
    config = validation._local_causal_lm_config(merged)

    def fail_construction(*args, **kwargs):
        raise AssertionError("combined module count must be bounded before construction")

    monkeypatch.setattr(validation.AutoModelForCausalLM, "from_config", fail_construction)

    assert validation._causal_lm_shapes(config) is None


def test_symlinked_artifact_index_is_not_a_completion_marker(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    index = merged / "artifact_index.json"
    external = tmp_path / "external-index.json"
    index.replace(external)
    index.symlink_to(external)

    assert is_merged_model_dir(merged) is False


def test_artifact_inventory_stops_after_more_files_than_the_index(tmp_path, monkeypatch):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "unindexed-one").write_text("one")
    (merged / "unindexed-two").write_text("two")
    paths = [path for path in merged.iterdir() if path.name != "artifact_index.json"]
    first_extra = paths.pop(paths.index(merged / "unindexed-one"))
    second_extra = paths.pop(paths.index(merged / "unindexed-two"))
    overread = False

    def inventory(_self, _pattern):
        nonlocal overread
        yield from paths
        yield first_extra
        overread = True
        yield second_extra

    monkeypatch.setattr(Path, "rglob", inventory)

    assert artifact_index_usable(merged) is False
    assert overread is False


def test_nested_artifact_index_symlink_is_rejected_before_name_filter(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    nested = merged / "nested"
    nested.mkdir()
    (nested / "artifact_index.json").symlink_to(tmp_path / "outside")

    assert artifact_index_usable(merged) is False


def test_nested_regular_artifact_index_participates_in_authenticated_inventory(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    nested = merged / "nested"
    nested.mkdir()
    (nested / "artifact_index.json").write_text("nested marker")
    write_artifact_index(str(merged))
    index = json.loads((merged / "artifact_index.json").read_text())

    assert "nested/artifact_index.json" in {entry["file"] for entry in index["artifacts"]}
    assert artifact_index_usable(merged) is True


def test_broken_symlink_in_artifact_tree_is_not_ignored(tmp_path):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "broken-tokenizer-link").symlink_to(tmp_path / "missing-tokenizer")

    assert is_merged_model_dir(merged) is False


def test_validation_json_loader_rejects_symlinked_metadata(tmp_path):
    from agoge_forger._run_status_validation import _load_json_object

    external = tmp_path / "external.json"
    external.write_text("{}")
    linked = tmp_path / "adapter_config.json"
    linked.symlink_to(external)

    assert _load_json_object(linked) is None


def test_offline_pretrained_loader_preserves_security_flags(tmp_path):
    from agoge_forger import _run_status_validation as validation

    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return "loaded"

    assert validation._offline_pretrained(Factory, tmp_path) == "loaded"
    assert calls == [(tmp_path, {"local_files_only": True, "trust_remote_code": False})]


def test_adapter_base_config_forwards_pinned_revision_and_security_flags(monkeypatch):
    from agoge_forger import _run_status_pretrained as pretrained
    from agoge_forger import _run_status_validation as validation

    calls = []
    loaded = validation.CONFIG_MAPPING["llama"]()

    class Tokenizer:
        pad_token = None
        eos_token = object()

        def __len__(self):
            return 2

    def load_config(source, **kwargs):
        calls.append(("config", source, kwargs))
        return loaded

    def load_tokenizer(source, **kwargs):
        calls.append(("tokenizer", source, kwargs))
        return Tokenizer()

    def validate_weights(source, config, revision):
        calls.append(("weights", source, revision, config))
        return True

    monkeypatch.setattr(validation.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(pretrained.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(validation, "local_base_weights_usable", validate_weights)
    monkeypatch.setattr(pretrained, "_tokenizer_inventory_usable", lambda *args, **kwargs: True)
    adapter = SimpleNamespace(
        base_model_name_or_path="org/base-model",
        revision="deadbeefcafedeadbeefcafedeadbeefcafedead",
    )

    assert validation._adapter_base_config(adapter) is loaded
    assert calls == [
        (
            "config",
            "org/base-model",
            {
                "revision": "deadbeefcafedeadbeefcafedeadbeefcafedead",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        ),
        (
            "weights",
            "org/base-model",
            "deadbeefcafedeadbeefcafedeadbeefcafedead",
            loaded,
        ),
        (
            "tokenizer",
            "org/base-model",
            {
                "revision": "deadbeefcafedeadbeefcafedeadbeefcafedead",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        ),
    ]


def test_validation_json_loader_fails_closed_on_deep_json(tmp_path):
    from agoge_forger._run_status_validation import _load_json_object

    metadata = tmp_path / "adapter_config.json"
    metadata.write_text('{"nested":' + "[" * 100_000 + "{}" + "]" * 100_000 + "}")

    assert _load_json_object(metadata) is None


def test_symlinked_merged_weights_are_not_standalone(tmp_path):
    external = tmp_path / "external.safetensors"
    external.write_bytes(_minimal_safetensors())
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").unlink()
    (merged / "model.safetensors").symlink_to(external)
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_unsupported_merged_safetensors_dtype_is_not_present(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type": "llama"}')
    (merged / "tokenizer_config.json").write_text("{}")
    (merged / "model.safetensors").write_bytes(_safetensors_with_dtype("NOT_A_SAFETENSORS_DTYPE"))

    assert is_merged_model_dir(merged) is False


@pytest.mark.parametrize(
    ("dtype", "element_size"),
    [("BOOL", 1), ("I32", 4)],
)
def test_non_floating_merged_weights_are_not_present(tmp_path, dtype, element_size):
    merged = _write_merged_model(tmp_path / "merged")
    (merged / "model.safetensors").write_bytes(
        _safetensors_with_shapes(
            TINY_LLAMA_SHAPES,
            dtype=dtype,
            element_size=element_size,
        )
    )
    write_artifact_index(str(merged))

    assert is_merged_model_dir(merged) is False


def test_explicit_merged_dir_is_honored(runner, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    elsewhere = _write_merged_model(tmp_path / "elsewhere" / "custom_merge")

    report = build_run_status(str(run_dir), merged_dir=str(elsewhere))
    assert report["merged_model"] == {"present": True, "path": str(elsewhere.resolve())}

    result = runner.invoke(app, ["run-status", str(run_dir), "--merged-dir", str(elsewhere)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {
        "present": True,
        "path": str(elsewhere.resolve()),
    }


def test_explicit_missing_merged_dir_reports_absent_and_exits_zero(runner, tmp_path):
    """A not-yet-exported merged model is an answer, not an error: still exit 0."""
    run_dir = _make_run_dir(tmp_path)
    _write_final_adapter(run_dir)
    missing = tmp_path / "elsewhere" / "not_exported_yet"

    report = build_run_status(str(run_dir), merged_dir=str(missing))
    assert report["merged_model"] == {"present": False, "path": None}

    result = runner.invoke(app, ["run-status", str(run_dir), "--merged-dir", str(missing)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["merged_model"] == {"present": False, "path": None}
