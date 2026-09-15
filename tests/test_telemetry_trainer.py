"""Bounded torch.profiler window on a tiny CPU SFTTrainer (no CUDA, no Hub)."""

from __future__ import annotations

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

from agoge_forger.config import ExperimentConfig, ProfileWindowConfig
from agoge_forger.telemetry import attach_telemetry, open_training_session
from agoge_forger.train.trainer import _build_sft_trainer, _build_training_args
from datasets import Dataset, disable_progress_bars, enable_progress_bars

VOCAB = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "agoge": 4}


@pytest.fixture(autouse=True)
def _quiet_dataset_progress_bars(monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    disable_progress_bars()
    yield
    enable_progress_bars()


@pytest.fixture
def tokenizer():
    special_tokens = {"unk_token": "[UNK]", "pad_token": "[PAD]", "eos_token": "[UNK]"}
    backend = Tokenizer(WordLevel(VOCAB, unk_token=special_tokens["unk_token"]))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


@pytest.fixture
def model():
    return AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=len(VOCAB),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )


def test_profile_window_covers_one_optimizer_step(tmp_path, monkeypatch, model, tokenizer):
    monkeypatch.chdir(tmp_path)
    dataset_path = tmp_path / "tiny.jsonl"
    dataset_path.write_text("{}\n")
    config = ExperimentConfig(
        model_id="local-tiny-llama",
        dataset_path=str(dataset_path),
        dataset_text_field="text",
        run_name="cpu_window",
    )
    config.training.batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.bf16 = False
    config.training.gradient_checkpointing = False
    config.training.save_steps = 1000
    config.telemetry.profile_window = ProfileWindowConfig(
        enabled=True,
        phase="train",
        start_step=2,
        end_step=2,
        backend="torch",
    )
    args = _build_training_args(config, str(tmp_path / "out"))
    args.max_steps = 2
    args.report_to = []
    args.save_strategy = "no"
    args.use_cpu = True
    rows = Dataset.from_list([{"text": "hello world"}, {"text": "world agoge"}])
    trainer = _build_sft_trainer(model, rows, tokenizer, args)
    session = open_training_session(config)
    attach_telemetry(trainer, session)
    trainer.train()
    session.close()

    markers = [json.loads(line) for line in session.markers_path.read_text().splitlines()]
    events = [row["event"] for row in markers]
    assert "train_start" in events
    assert "profile_window_start" in events
    assert "profile_window_end" in events
    assert "train_end" in events
    window_steps = [row["global_step"] for row in markers if row.get("profile_window_ref")]
    assert 2 in window_steps
    summary = json.loads(session.summary_path.read_text())
    assert summary["record_kind"] == "agoge_profile_window"
    assert summary["profile_window_id"] == "cpu_window:train:2-2"
    assert summary["phase"] == "train"
    assert summary["backend"] == "torch"
    assert summary["overhead"]["unprofiled_step_mean_s"] is not None
    assert summary["kernel_duration"]["status"] in {"ok", "unavailable"}
    if summary["kernel_duration"]["status"] != "ok":
        assert summary["kernel_duration"]["value"] is None
    request = json.loads(session.request_path.read_text())
    assert request["enabled"] is True
    assert request["backends"]["nsight_systems"]["status"] == "unsupported"
