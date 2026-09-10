"""Pin trainer construction to the *installed* TRL / Transformers API.

The train path broke silently once before: TRL 1.x moved the SFT knobs off
``SFTTrainer.__init__`` onto ``SFTConfig`` (``max_seq_length`` -> ``max_length``,
``tokenizer`` -> ``processing_class``) and Transformers 5.x deleted
``TrainingArguments.save_safetensors`` outright. Every ``agoge train-qlora`` run
died at trainer construction while the unit suite stayed green, because no test
ever instantiated ``SFTTrainer``.

These tests close that gap by exercising ``_build_training_args`` and
``_build_sft_trainer`` against the **real** ``trl`` package — no stubs, no
monkeypatching of the library — so the next upstream kwarg rename fails the PR
quality gate instead of shipping another broken train path.

Staying CI-friendly: the model is built from a hand-written ``LlamaConfig``
(1 layer, hidden size 8) and the tokenizer from an in-memory word-level vocab,
so there is no GPU requirement and no Hugging Face Hub download.
"""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast
from trl import SFTConfig

from agoge_forger.config import ExperimentConfig
from agoge_forger.train.trainer import _build_sft_trainer, _build_training_args
from datasets import Dataset, disable_progress_bars, enable_progress_bars

# Deliberately not "text": proves the configured field name is what reaches TRL.
TEXT_FIELD = "body"
VOCAB = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "agoge": 4}


@pytest.fixture(autouse=True)
def _quiet_dataset_progress_bars(monkeypatch):
    """TRL maps over the dataset at construction; keep pytest output readable."""
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    disable_progress_bars()
    yield
    enable_progress_bars()


@pytest.fixture
def config():
    """A config whose every field is distinguishable from the TRL defaults."""
    cfg = ExperimentConfig(
        model_id="local-tiny-llama",
        dataset_path="unused.jsonl",
        dataset_text_field=TEXT_FIELD,
    )
    cfg.training.batch_size = 3
    cfg.training.gradient_accumulation_steps = 7
    cfg.training.learning_rate = 1.5e-4
    cfg.training.num_train_epochs = 2
    cfg.training.seed = 1234
    cfg.training.save_steps = 25
    cfg.training.save_total_limit = 4
    cfg.training.max_seq_length = 321
    cfg.training.gradient_checkpointing = True
    # CI runners are CPU-only; disable bf16 even when configs/minicpm5_canary.yaml enables it.
    cfg.training.bf16 = False
    return cfg


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


@pytest.fixture
def dataset():
    return Dataset.from_list([{TEXT_FIELD: "hello world"}, {TEXT_FIELD: "world agoge"}])


def test_build_training_args_returns_sft_config(config, tmp_path):
    """Every kwarg we pass must exist on the installed SFTConfig.

    A removed or renamed field raises TypeError here — this is the assertion
    that would have caught `save_safetensors` disappearing in Transformers 5.
    """
    args = _build_training_args(config, str(tmp_path))

    assert isinstance(args, SFTConfig)


def test_sft_specific_fields_come_from_experiment_config(config, tmp_path):
    """The YAML keys stay stable; only the TRL-side names change."""
    args = _build_training_args(config, str(tmp_path))

    assert args.max_length == config.training.max_seq_length == 321
    assert args.dataset_text_field == config.dataset_text_field == TEXT_FIELD


def test_shared_training_fields_survive_the_sft_config_switch(config, tmp_path):
    args = _build_training_args(config, str(tmp_path))

    assert str(args.output_dir) == str(tmp_path)
    assert args.per_device_train_batch_size == 3
    assert args.gradient_accumulation_steps == 7
    assert args.learning_rate == pytest.approx(1.5e-4)
    assert args.num_train_epochs == 2
    assert args.seed == 1234
    assert args.save_strategy == "steps"
    assert args.save_steps == 25
    assert args.save_total_limit == 4
    assert args.gradient_checkpointing is True
    assert args.bf16 is False


def test_build_sft_trainer_constructs_against_installed_trl(
    config, model, dataset, tokenizer, tmp_path
):
    """Construct a real SFTTrainer so removed kwargs raise TypeError in CI."""
    args = _build_training_args(config, str(tmp_path))

    trainer = _build_sft_trainer(model, dataset, tokenizer, args)

    assert trainer.args is args
    # TRL tokenized the dataset, which it can only do via the configured text
    # field — proof that `dataset_text_field` actually reached the library.
    assert "input_ids" in trainer.train_dataset.column_names


def test_trainer_exposes_tokenizer_for_the_adapter_save_path(
    config, model, dataset, tokenizer, tmp_path
):
    """`_finalize_training_run` saves the tokenizer next to the adapter.

    Transformers 5 removed `Trainer.tokenizer`, so that code now reads
    `processing_class`; this guards the attribute it depends on.
    """
    args = _build_training_args(config, str(tmp_path))

    trainer = _build_sft_trainer(model, dataset, tokenizer, args)

    assert trainer.processing_class is not None
    assert callable(trainer.processing_class.save_pretrained)


def completion_args(config, tmp_path):
    from dataclasses import replace

    config.dataset_text_field = "text"
    config.training.completion_only_loss = True
    config.training.gradient_checkpointing = False
    config.training.gradient_accumulation_steps = 1
    args = replace(_build_training_args(config, str(tmp_path)), use_cpu=True)
    args.max_steps = 1
    args.report_to = []
    args.save_strategy = "no"
    return args


def test_completion_batch_and_cpu_step(config, model, tokenizer, tmp_path):
    """Catch prompt leakage, completion truncation, duplicate EOS, and masked padding."""
    import math

    args = completion_args(config, tmp_path)
    rows = Dataset.from_list(
        [
            {
                "text": "hello world agoge",
                "completion_start_char": 12,
                "canonical_id": "a",
                "lineage_id": "l",
                "group_id": "g",
            },
            {
                "text": "hello world[UNK]",
                "completion_start_char": 6,
                "canonical_id": "b",
                "lineage_id": "m",
                "group_id": "h",
            },
        ]
    )
    trainer = _build_sft_trainer(model, rows, tokenizer, args)
    assert trainer.args.completion_only_loss is True
    assert trainer.train_dataset[0]["canonical_id"] == "a"
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["input_ids"].device.type == "cpu"
    assert next(trainer.model.parameters()).device.type == "cpu"
    pairs = sorted(zip(batch["input_ids"].tolist(), batch["labels"].tolist()))
    assert pairs == [([2, 3, 0, 1], [-100, 3, 0, -100]), ([2, 3, 4, 0], [-100, -100, 4, 0])]
    assert math.isfinite(trainer.train().training_loss)


@pytest.mark.parametrize(
    "case",
    [
        ("hello world", True, 20, "completion_start_char"),
        ("hello world", 1.5, 20, "completion_start_char"),
        ("hello world", -1, 20, "completion_start_char"),
        ("hello world", 11, 20, "completion_start_char"),
        ("hello world", None, 20, "completion_start_char"),
        ("hello world", 2, 20, "boundary"),
        ("hello world", 6, 2, "max_seq_length"),
        ("hello world", 6, None, "max_seq_length"),
        # Its sole content token is at index 0: after shifting, only EOS would
        # receive loss. Predicting EOS is not supervision of completion content.
        ("hello", 0, 20, "causal"),
        ("hello   ", 5, 20, "completion"),
    ],
)
def test_completion_refuses_unusable_rows(config, model, tokenizer, tmp_path, case):
    text, offset, limit, reason = case
    args = completion_args(config, tmp_path)
    args.max_length = limit
    rows = Dataset.from_list([{"text": text, "completion_start_char": offset}])
    with pytest.raises(ValueError, match=reason):
        _build_sft_trainer(model, rows, tokenizer, args)


@pytest.mark.parametrize("case", [("é world é world agoge", 16), ("hello ### world ### agoge", 20)])
def test_unicode_boundary_uses_declared_offset(config, model, tokenizer, tmp_path, case):
    text, offset = case
    args = completion_args(config, tmp_path)
    rows = Dataset.from_list([{"text": text, "completion_start_char": offset}])
    trainer = _build_sft_trainer(model, rows, tokenizer, args)
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, 4, 0]]


def test_completion_requires_offsets(config, model, tokenizer, tmp_path):
    args = completion_args(config, tmp_path)
    rows = Dataset.from_list([{"text": "hello world", "completion_start_char": 6}])
    with pytest.raises(ValueError, match="offset"):
        _build_sft_trainer(model, rows, None, args)


@pytest.mark.parametrize("enabled", [False, True])
def test_flat_yaml_completion_mode(tmp_path, enabled):
    from agoge_forger.config import load_config

    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"hello world","completion_start_char":6}\n')
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_id: local\ndataset_path: rows.jsonl\nbf16: false\n"
        + ("completion_only_loss: true\n" if enabled else "")
    )
    cfg = load_config(str(config_path))
    assert _build_training_args(cfg, str(tmp_path)).completion_only_loss is enabled


def test_completion_field_must_be_text(config, tmp_path):
    config.training.completion_only_loss = True
    with pytest.raises(ValueError, match="dataset_text_field"):
        _build_training_args(config, str(tmp_path))


def test_completion_bos_and_evidence(config, model, tokenizer, tmp_path):
    import json

    from tokenizers.processors import TemplateProcessing

    tokenizer.add_special_tokens({"bos_token": tokenizer.pad_token})
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="[PAD] $A [UNK]", special_tokens=[("[PAD]", 1), ("[UNK]", 0)]
    )
    args = completion_args(config, tmp_path)
    rows = Dataset.from_list([{"text": "hello world", "completion_start_char": 6}])
    trainer = _build_sft_trainer(model, rows, tokenizer, args)
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["input_ids"].tolist() == [[1, 2, 3, 0]]
    assert batch["labels"].tolist() == [[-100, -100, 3, 0]]
    evidence = json.loads((tmp_path / "completion_preprocessing.json").read_text())
    assert evidence["shifted_supervised_tokens"] == 2
    assert evidence["max_tokens"] == 4
    assert evidence["truncation"] is False


@pytest.mark.parametrize("column", ["labels", "assistant_masks", "seq_lengths"])
def test_completion_refuses_preexisting_loss_controls(config, model, tokenizer, tmp_path, column):
    args = completion_args(config, tmp_path)
    rows = Dataset.from_list(
        [{"text": "hello world", "completion_start_char": 6, column: [-100, -100, -100]}]
    )
    with pytest.raises(ValueError, match="reserved"):
        _build_sft_trainer(model, rows, tokenizer, args)


def test_shared_pad_eos_keeps_terminal_target(config, model, tokenizer, tmp_path):
    tokenizer.pad_token = tokenizer.eos_token
    args = completion_args(config, tmp_path)
    rows = Dataset.from_list(
        [
            {"text": "hello world[UNK]", "completion_start_char": 6},
            {"text": "hello world agoge", "completion_start_char": 6},
        ]
    )
    trainer = _build_sft_trainer(model, rows, tokenizer, args)
    batch = next(iter(trainer.get_train_dataloader()))
    assert sorted(batch["labels"].tolist()) == [[-100, 3, 0, -100], [-100, 3, 4, 0]]


@pytest.mark.parametrize(
    "case",
    [
        ("hello world[UNK]", 3, [2, 3, 0], [-100, 3, 0]),
        ("hello world[UNK][UNK]", 4, [2, 3, 0, 0], [-100, 3, 0, 0]),
    ],
)
def test_source_eos_suppresses_only_added_eos_at_length_limit(
    config, model, tokenizer, tmp_path, case
):
    """Keep exact source EOS tokens, without a redundant postprocessor EOS."""
    from tokenizers.processors import TemplateProcessing

    text, limit, ids, labels = case
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="$A [UNK]", special_tokens=[("[UNK]", 0)]
    )
    args = completion_args(config, tmp_path)
    args.max_length = limit
    rows = Dataset.from_list([{"text": text, "completion_start_char": 6}])
    trainer = _build_sft_trainer(model, rows, tokenizer, args)
    batch = next(iter(trainer.get_train_dataloader()))
    assert trainer.train_dataset[0]["text"] == text
    assert batch["input_ids"].tolist() == [ids]
    assert batch["labels"].tolist() == [labels]
