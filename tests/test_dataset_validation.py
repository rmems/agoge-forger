import json
import logging

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.datasets import normalize_row


def test_normalize_text():
    row = {"text": "hello"}
    assert normalize_row(row) == {"text": "hello"}


def test_normalize_messages():
    row = {"messages": [{"role": "user", "content": "hi"}]}
    res = normalize_row(row)
    assert "User: hi" in res["text"] or "hi" in res["text"]


def test_normalize_instruction():
    row = {"instruction": "do this", "output": "done"}
    res = normalize_row(row)
    assert "Instruction: do this" in res["text"]
    assert "Output: done" in res["text"]


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_dataset_stats_counts_local_jsonl_without_loading_model(
    tmp_path, monkeypatch, caplog, trust_remote_code
):
    special_tokens = {"unk_token": "[UNK]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token=special_tokens["unk_token"])
    )
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens(special_tokens)
    if trust_remote_code:
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }} {% endfor %}"
        )
    model_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(model_path)
    path = tmp_path / "data.jsonl"
    rows = [
        {"text": "hi"},
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello world"},
            ]
        },
        {"instruction": "repair", "output": "done"},
        {"instruction": "repair", "input": "code", "output": "done"},
    ]
    path.write_text("\n" + "\n\n".join(json.dumps(row) for row in rows) + "\n")

    load_tokenizer = AutoTokenizer.from_pretrained

    def checked_load_tokenizer(*args, **kwargs):
        assert kwargs["trust_remote_code"] is trust_remote_code
        return load_tokenizer(*args, **kwargs)

    def forbidden_load_model(*args, **kwargs):
        pytest.fail("dataset-stats must not load causal LM weights")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", checked_load_tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden_load_model)
    args = ["dataset-stats", "--path", str(path), "--model-id", str(model_path)]
    if trust_remote_code:
        args.append("--trust-remote-code")
    with caplog.at_level(logging.INFO, logger="agoge"):
        result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.exception
    # BOS/EOS add two tokens to each row; chat templates omit the two role labels.
    assert "Dataset Rows: 4" in caplog.messages
    assert "Min Tokens: 3" in caplog.messages
    assert "Max Tokens: 8" in caplog.messages
    assert f"Mean Tokens: {'5.50' if trust_remote_code else '6.00'}" in caplog.messages
