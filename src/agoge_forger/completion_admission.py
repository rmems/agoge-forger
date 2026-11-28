"""Offline, pre-freeze admission using the trainer's exact completion encoding."""

import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ._atomic_directory import rename_noreplace
from ._strict_json import decode_json_object
from .compact_completion import CompletionFormat, compact_json_line, render_compact_row
from .models.tokenizer_setup import prepare_training_tokenizer
from .path_safety import resolve_absent_output_directory, resolve_existing_path
from .split_schema import sha256_file
from .train.completion import completion_token_length

_RESERVED = {"labels", "assistant_masks", "seq_lengths", "input_ids", "completion_mask"}


def _load_tokenizer(directory: Path):
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    if not (directory / "tokenizer.json").is_file():
        raise ValueError("tokenizer directory must contain a saved fast tokenizer.json")
    # Match the model loader's explicit return type for the dynamic tokenizer factory.
    tokenizer = cast(
        Callable[..., PreTrainedTokenizerBase], cast(object, AutoTokenizer.from_pretrained)
    )(directory, local_files_only=True, trust_remote_code=False, use_fast=True)
    return prepare_training_tokenizer(tokenizer)


def _tokenizer_hashes(directory: Path):
    return {path.name: sha256_file(path) for path in sorted(directory.glob("*.json"))}


def _json_line(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def _prepare_row(raw, line_number, tokenizer, completion_format):
    row = decode_json_object(raw, f"line {line_number}")
    try:
        if _RESERVED.intersection(row):
            raise ValueError("source contains reserved loss controls or prepared tokens")
        if completion_format == CompletionFormat.SEARCH_REPLACE:
            row = render_compact_row(row)
        return row, completion_token_length(row, tokenizer)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: {exc}") from exc


def _admitted_line(raw, row, completion_format):
    if completion_format == CompletionFormat.SEARCH_REPLACE:
        return compact_json_line(raw, row)
    return raw if raw.endswith(b"\n") else raw + b"\n"


def _admit_rows(source: Path, stage: Path, tokenizer, max_length: int, completion_format):
    histogram: Counter[int] = Counter()
    accepted = 0
    digest = hashlib.sha256()
    with (
        source.open("rb") as inputs,
        (stage / "admitted.jsonl").open("wb") as output,
        (stage / "decisions.jsonl").open("w", encoding="utf-8") as decisions,
    ):
        for line_number, raw in enumerate(inputs, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row, length = _prepare_row(raw, line_number, tokenizer, completion_format)
            histogram[length] += 1
            fits = length <= max_length
            accepted += int(fits)
            if fits:
                output.write(_admitted_line(raw, row, completion_format))
            decisions.write(
                _json_line(
                    {
                        "source_line": line_number,
                        "tokens": length,
                        "decision": "admitted" if fits else "over_budget",
                    }
                )
            )
    if not histogram:
        raise ValueError("source has no completion rows")
    return {
        "source_sha256": digest.hexdigest(),
        "source_rows": sum(histogram.values()),
        "admitted_rows": accepted,
        "over_budget_rows": sum(histogram.values()) - accepted,
        "token_length_histogram": {str(key): histogram[key] for key in sorted(histogram)},
    }


def admit_completions(
    source: str,
    tokenizer_dir: str,
    output_dir: str,
    max_length: int,
    *,
    completion_format: CompletionFormat = CompletionFormat.FROZEN_TEXT,
):
    """Publish a new local admission directory; invalid rows abort the entire operation."""
    if type(max_length) is not int or max_length < 1:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("max_length must be a positive integer")
    completion_format = CompletionFormat(completion_format)
    source_path = resolve_existing_path(source, must_be_file=True)
    tokenizer_path = resolve_existing_path(tokenizer_dir, must_be_dir=True)
    destination = resolve_absent_output_directory(output_dir)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    source_hash = sha256_file(source_path)
    hashes = _tokenizer_hashes(tokenizer_path)
    tokenizer = _load_tokenizer(tokenizer_path)
    with tempfile.TemporaryDirectory(prefix=".admission-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        report = _admit_rows(source_path, stage, tokenizer, max_length, completion_format)
        report.update(
            schema_version="completion-admission-v1",
            max_seq_length=max_length,
            truncation=False,
            completion_format=completion_format.value,
            tokenizer_files_sha256=hashes,
            admitted_sha256=sha256_file(stage / "admitted.jsonl"),
            decisions_sha256=sha256_file(stage / "decisions.jsonl"),
        )
        (stage / "admission_report.json").write_text(_json_line(report), encoding="utf-8")
        if hashes != _tokenizer_hashes(tokenizer_path):
            raise ValueError("tokenizer files changed during admission")
        _require_unchanged_source(source_path, source_hash, report["source_sha256"])
        rename_noreplace(stage, destination)
    return report


def _require_unchanged_source(source: Path, expected_hash: str, observed_hash: str):
    if observed_hash != expected_hash or sha256_file(source) != expected_hash:
        raise ValueError("source changed during admission")
