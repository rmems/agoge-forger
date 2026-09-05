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
def _quiet_dataset_progress_bars():
    """TRL maps over the dataset at construction; keep pytest output readable."""
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
    backend = Tokenizer(WordLevel(VOCAB, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[UNK]",
    )


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
