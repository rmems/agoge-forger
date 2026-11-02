"""Merged-model export must seal only its own files, and only onto the pinned base.

The merged ``artifact_index.json`` copies the adapter's sealed producer_provenance
(``base_model_name_or_path`` at ``revision``). Two ways used to break that seal:

* exporting into a non-empty directory sealed stale files left by an earlier
  export (an old unsharded ``model.safetensors`` that wins at load time);
* ``--base-model`` let the caller merge into a different base (or the Hub HEAD of
  the same repo) while the index still claimed the adapter's pinned base.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from typer.testing import CliRunner

from agoge_forger.artifacts.safetensors_io import write_artifact_index
from agoge_forger.cli import app
from agoge_forger.eval import ArtifactProducerProvenance
from agoge_forger.export.merge_adapter import export_final_model, merge_adapter
from tests.evaluation_contract_cases import MODEL_REPOSITORY, MODEL_REVISION, model_provenance

LOAD_TARGET = "agoge_forger.export.merge_adapter.load_base_model"
OTHER_REVISION = "0" * 40


class _StopBeforeMerge(RuntimeError):
    pass


def _provenance() -> ArtifactProducerProvenance:
    return ArtifactProducerProvenance.model_validate(model_provenance())


def _write_sealed_adapter(path: Path, *, revision: str | None = MODEL_REVISION) -> Path:
    path.mkdir(parents=True)
    config: dict[str, object] = {"base_model_name_or_path": MODEL_REPOSITORY}
    if revision is not None:
        config["revision"] = revision
    (path / "adapter_config.json").write_text(json.dumps(config))
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "artifact_index.json").write_text(
        json.dumps(
            {"output_dir": str(path), "artifacts": [], "producer_provenance": model_provenance()}
        )
    )
    return path


def _snapshot_dir(root: Path, repository: str = MODEL_REPOSITORY, revision: str = MODEL_REVISION):
    snapshot = root / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    snapshot.mkdir(parents=True)
    return snapshot


@pytest.fixture
def captured_load(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_load_base_model(model_id, **kwargs):
        calls.append({"model_id": model_id, "revision": kwargs.get("revision")})
        raise _StopBeforeMerge("stop-before-merge")

    monkeypatch.setattr(LOAD_TARGET, fake_load_base_model)
    return calls


# --- base identity (cli-eval-export-c1) --------------------------------------


@pytest.mark.parametrize("base", ["other/model", "example/base-model-instruct"])
def test_merge_refuses_a_hub_base_other_than_the_sealed_one(tmp_path, captured_load, base):
    adapter = _write_sealed_adapter(tmp_path / "adapter")

    with pytest.raises(ValueError, match="does not match the adapter's sealed"):
        merge_adapter(base, str(adapter), str(tmp_path / "merged"))

    assert captured_load == []
    assert not (tmp_path / "merged").exists()


def test_export_override_refuses_a_different_hub_base(tmp_path, captured_load):
    adapter = _write_sealed_adapter(tmp_path / "adapter")

    with pytest.raises(ValueError, match="does not match the adapter's sealed"):
        export_final_model(
            out_dir=str(tmp_path / "merged"),
            adapter_path=str(adapter),
            base_model_id="replacement/model",
        )

    assert captured_load == []


def test_export_override_naming_the_sealed_repo_loads_the_pinned_revision(tmp_path, captured_load):
    """Same repo, but the override used to drop the revision and load Hub HEAD."""
    adapter = _write_sealed_adapter(tmp_path / "adapter")

    with pytest.raises(_StopBeforeMerge):
        export_final_model(
            out_dir=str(tmp_path / "merged"),
            adapter_path=str(adapter),
            base_model_id=MODEL_REPOSITORY,
        )

    assert captured_load == [{"model_id": MODEL_REPOSITORY, "revision": MODEL_REVISION}]


def test_merge_loads_the_sealed_revision_when_the_adapter_records_none(tmp_path, captured_load):
    adapter = _write_sealed_adapter(tmp_path / "adapter", revision=None)

    with pytest.raises(_StopBeforeMerge):
        merge_adapter(MODEL_REPOSITORY, str(adapter), str(tmp_path / "merged"))

    assert captured_load == [{"model_id": MODEL_REPOSITORY, "revision": MODEL_REVISION}]


def test_merge_refuses_a_revision_other_than_the_sealed_one(tmp_path, captured_load):
    adapter = _write_sealed_adapter(tmp_path / "adapter", revision=OTHER_REVISION)

    with pytest.raises(ValueError, match="does not match the sealed producer_provenance"):
        merge_adapter(MODEL_REPOSITORY, str(adapter), str(tmp_path / "merged"))
    with pytest.raises(ValueError, match="does not match the sealed producer_provenance"):
        merge_adapter(
            MODEL_REPOSITORY,
            str(adapter),
            str(tmp_path / "merged"),
            revision=OTHER_REVISION,
            infer_revision=False,
        )

    assert captured_load == []


def test_merge_refuses_a_local_base_not_tied_to_the_sealed_identity(tmp_path, captured_load):
    adapter = _write_sealed_adapter(tmp_path / "adapter")
    plain = tmp_path / "models" / "base-model"
    plain.mkdir(parents=True)
    wrong_sha = _snapshot_dir(tmp_path / "hub", revision=OTHER_REVISION)
    wrong_repo = _snapshot_dir(tmp_path / "hub", repository="example/other-model")

    for local in (plain, wrong_sha, wrong_repo):
        with pytest.raises(ValueError, match="cannot be tied to the sealed"):
            merge_adapter(str(local), str(adapter), str(tmp_path / "merged"))

    assert captured_load == []


def test_merge_refuses_a_local_dir_named_like_the_sealed_repo(tmp_path, captured_load, monkeypatch):
    """Transformers loads a local directory whenever one matches the string."""
    adapter = _write_sealed_adapter(tmp_path / "adapter")
    (tmp_path / "cwd" / MODEL_REPOSITORY).mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "cwd")

    with pytest.raises(ValueError, match="cannot be tied to the sealed"):
        merge_adapter(MODEL_REPOSITORY, str(adapter), str(tmp_path / "merged"))

    assert captured_load == []


def test_merge_accepts_the_sealed_hf_cache_snapshot(tmp_path, captured_load):
    adapter = _write_sealed_adapter(tmp_path / "adapter")
    snapshot = _snapshot_dir(tmp_path / "hub")
    link = tmp_path / "granite-base"
    link.symlink_to(snapshot, target_is_directory=True)

    with pytest.raises(_StopBeforeMerge):
        merge_adapter(str(link), str(adapter), str(tmp_path / "merged"))

    assert captured_load == [{"model_id": str(snapshot.resolve()), "revision": MODEL_REVISION}]


def test_merge_accepts_the_local_base_directory_the_adapter_was_trained_from(
    tmp_path, captured_load
):
    """Measured runs train from a local copy, so the sealed base is that directory."""
    local = tmp_path / "models" / "granite-base"
    local.mkdir(parents=True)
    other = tmp_path / "models" / "other-base"
    other.mkdir(parents=True)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    sealed = model_provenance(repository=str(local))
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(local), "revision": MODEL_REVISION})
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "artifact_index.json").write_text(
        json.dumps({"output_dir": str(adapter), "artifacts": [], "producer_provenance": sealed})
    )

    with pytest.raises(ValueError, match="cannot be tied to the sealed"):
        merge_adapter(str(other), str(adapter), str(tmp_path / "merged"))
    assert captured_load == []

    with pytest.raises(_StopBeforeMerge):
        merge_adapter(str(local), str(adapter), str(tmp_path / "merged"))
    assert captured_load == [{"model_id": str(local.resolve()), "revision": MODEL_REVISION}]


# --- stale output (cli-eval-export-e1) ---------------------------------------


def test_merge_refuses_a_non_empty_output_directory_before_loading(tmp_path, captured_load):
    adapter = _write_sealed_adapter(tmp_path / "adapter")
    out = tmp_path / "merged"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"stale")

    with pytest.raises(ValueError, match="non-empty output directory"):
        merge_adapter(MODEL_REPOSITORY, str(adapter), str(out))

    assert captured_load == []
    assert sorted(p.name for p in out.iterdir()) == ["model.safetensors"]


def test_merge_refuses_an_output_path_that_is_a_file(tmp_path, captured_load):
    adapter = _write_sealed_adapter(tmp_path / "adapter")
    out = tmp_path / "merged"
    out.write_text("not a directory")

    with pytest.raises(ValueError, match="not a directory"):
        merge_adapter(MODEL_REPOSITORY, str(adapter), str(out))

    assert captured_load == []


@pytest.mark.parametrize("pre_create", [False, True])
def test_merge_accepts_an_absent_or_empty_output_directory(tmp_path, captured_load, pre_create):
    adapter = _write_sealed_adapter(tmp_path / "adapter")
    out = tmp_path / "merged"
    if pre_create:
        out.mkdir()

    with pytest.raises(_StopBeforeMerge):
        merge_adapter(MODEL_REPOSITORY, str(adapter), str(out))

    assert len(captured_load) == 1
    assert out.exists() is pre_create


# --- CLI boundary -------------------------------------------------------------


def _assert_clean_exit(result, code):
    assert result.exit_code == code, result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_cli_merge_adapter_refuses_a_different_base(tmp_path, captured_load, caplog):
    adapter = _write_sealed_adapter(tmp_path / "adapter")
    other = tmp_path / "base_b"
    other.mkdir()

    with caplog.at_level("ERROR", logger="agoge"):
        result = CliRunner().invoke(
            app,
            [
                "merge-adapter",
                "--base-model",
                str(other),
                "--adapter-path",
                str(adapter),
                "--out-dir",
                str(tmp_path / "merged"),
            ],
        )

    _assert_clean_exit(result, 1)
    assert captured_load == []
    assert any("cannot be tied to the sealed" in m for m in caplog.messages)


def test_cli_merge_adapter_reports_a_non_object_artifact_index(tmp_path, captured_load):
    adapter = _write_sealed_adapter(tmp_path / "adapter")
    (adapter / "artifact_index.json").write_text("[]")

    result = CliRunner().invoke(
        app,
        [
            "merge-adapter",
            "--base-model",
            MODEL_REPOSITORY,
            "--adapter-path",
            str(adapter),
            "--out-dir",
            str(tmp_path / "merged"),
        ],
    )

    _assert_clean_exit(result, 1)
    assert captured_load == []


def test_cli_export_final_model_refuses_a_different_base(tmp_path, captured_load, caplog):
    adapter = _write_sealed_adapter(tmp_path / "adapter")

    with caplog.at_level("ERROR", logger="agoge"):
        result = CliRunner().invoke(
            app,
            [
                "export-final-model",
                "--adapter-path",
                str(adapter),
                "--base-model",
                "other/model",
                "--out-dir",
                str(tmp_path / "merged"),
            ],
        )

    _assert_clean_exit(result, 1)
    assert captured_load == []
    assert any("does not match the adapter's sealed" in m for m in caplog.messages)


@pytest.mark.parametrize("command", ["merge-adapter", "export-final-model"])
def test_cli_refuses_a_non_empty_out_dir(tmp_path, captured_load, caplog, command):
    adapter = _write_sealed_adapter(tmp_path / "adapter")
    out = tmp_path / "merged"
    out.mkdir()
    (out / "artifact_index.json").write_text("{}")
    args = [command, "--adapter-path", str(adapter), "--out-dir", str(out)]
    if command == "merge-adapter":
        args += ["--base-model", MODEL_REPOSITORY]

    with caplog.at_level("ERROR", logger="agoge"):
        result = CliRunner().invoke(app, args)

    _assert_clean_exit(result, 1)
    assert captured_load == []
    assert any("non-empty output directory" in m for m in caplog.messages)


# --- end to end with real (tiny) weights -------------------------------------


def _save_tiny_base(snapshot: Path) -> None:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    LlamaForCausalLM(config).save_pretrained(snapshot)
    vocab = {f"t{i}": i for i in range(32)}
    special_tokens = {"unk_token": "<unk>", "eos_token": "t1"}
    vocab[special_tokens["unk_token"]] = 0
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=special_tokens["unk_token"]))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # type: ignore[assignment]
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, **special_tokens).save_pretrained(snapshot)


def _save_tiny_adapter(base: Path, out: Path) -> None:
    torch.manual_seed(1)
    model = get_peft_model(
        LlamaForCausalLM.from_pretrained(base),  # nosec B615 - local tmp snapshot, no Hub
        LoraConfig(r=2, target_modules=["q_proj", "v_proj"], init_lora_weights=False),
    )
    model.save_pretrained(out)
    write_artifact_index(str(out), producer_provenance=_provenance())


def test_reexport_into_a_used_directory_is_refused_and_keeps_its_seal(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    snapshot = _snapshot_dir(tmp_path / "hub")
    _save_tiny_base(snapshot)
    adapter = tmp_path / "adapter"
    _save_tiny_adapter(snapshot, adapter)
    out = tmp_path / "merged"

    merge_adapter(str(snapshot), str(adapter), str(out), max_shard_size="1GB")
    first_index = (out / "artifact_index.json").read_bytes()
    first_files = sorted(p.name for p in out.iterdir())
    assert "model.safetensors" in first_files
    sealed = json.loads(first_index)["producer_provenance"]
    assert sealed["base_model_name_or_path"] == MODEL_REPOSITORY
    assert sealed["revision"] == MODEL_REVISION

    with pytest.raises(ValueError, match="non-empty output directory"):
        merge_adapter(str(snapshot), str(adapter), str(out), max_shard_size="5KB")

    assert sorted(p.name for p in out.iterdir()) == first_files
    assert (out / "artifact_index.json").read_bytes() == first_index
