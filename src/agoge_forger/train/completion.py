"""Loss masks for frozen text with explicit Unicode character boundaries."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Token:
    id: int
    left: int
    right: int
    added: bool


def _token_metadata(token, offset, special):
    left, right = offset
    return _Token(token, left, right, bool(special) and (left, right) == (0, 0))


def _completion_boundary(row):
    text = row.get("text")
    start = row.get("completion_start_char")
    # Exact int rejects bool and subclasses at the frozen-data boundary.
    if not isinstance(text, str) or type(start) is not int:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("completion_start_char must be an integer offset into nonempty text")
    if not 0 <= start < len(text):
        raise ValueError("completion_start_char must be an integer offset into nonempty text")
    if not text[start:].strip():
        raise ValueError("completion must be nonempty")
    return text, start


def _validate_tokenizer(tokenizer):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("completion masking requires reliable fast-tokenizer offsets")
    if tokenizer.eos_token_id is None:
        raise ValueError("completion masking requires an EOS token")


def _tokenize_with_offsets(text, tokenizer):
    _validate_tokenizer(tokenizer)
    encoded = tokenizer(
        text,
        truncation=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    return _decode_token_metadata(encoded)


def _decode_token_metadata(encoded):
    ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    specials = encoded.get("special_tokens_mask")
    if offsets is None or specials is None:
        raise ValueError("tokenizer lacks reliable offsets")
    if len(offsets) != len(ids):
        raise ValueError("tokenizer lacks reliable offsets")
    return [
        _token_metadata(token, offset, special)
        for token, offset, special in zip(ids, offsets, specials, strict=True)
    ]


def _source_eos(token, text, tokenizer):
    return (
        token.id == tokenizer.eos_token_id and text[token.left : token.right] == tokenizer.eos_token
    )


def _remove_redundant_eos(tokens, text, tokenizer):
    """Remove only an added EOS immediately following a literal source EOS."""
    if len(tokens) < 2:
        return tokens
    last = tokens[-1]
    if not last.added or last.id != tokenizer.eos_token_id:
        return tokens
    if _source_eos(tokens[-2], text, tokenizer):
        return tokens[:-1]
    return tokens


def _added_token_supervision(token, index, count, tokenizer):
    if index == 0 and token.id == tokenizer.bos_token_id:
        return 0, False
    if index == count - 1 and token.id == tokenizer.eos_token_id:
        return 1, True
    raise ValueError("unsupported special token without reliable offsets")


def _validate_source_offset(token, text_length, previous_start):
    if not 0 <= token.left < token.right <= text_length:
        raise ValueError("tokenizer lacks reliable character offsets")
    if token.left < previous_start:
        raise ValueError("tokenizer lacks reliable character offsets")


def _is_prompt_special(token, tokenizer, is_eos):
    return not is_eos and token.id in {tokenizer.bos_token_id, tokenizer.pad_token_id}


def _source_token_supervision(token, text, start, tokenizer):
    if token.left < start < token.right:
        raise ValueError("token crosses completion boundary")
    active = int(token.left >= start)
    is_eos = _source_eos(token, text, tokenizer)
    if active and _is_prompt_special(token, tokenizer, is_eos):
        raise ValueError("completion contains BOS or padding special token")
    return active, is_eos


def _token_supervision(token, position, source, previous_start):
    index, count = position
    text, start, tokenizer = source
    if token.added:
        active, eos = _added_token_supervision(token, index, count, tokenizer)
        return active, eos, previous_start
    _validate_source_offset(token, len(text), previous_start)
    active, eos = _source_token_supervision(token, text, start, tokenizer)
    return active, eos, token.left


def _validate_causal_context(index, active):
    if index == 0 and active:
        raise ValueError("completion content at index 0 has no causal context")


def _effective_content_target(token, index, active, eos):
    if token.added or eos:
        return False
    _validate_causal_context(index, active)
    return bool(active)


def _completion_mask(tokens, text, start, tokenizer):
    mask = []
    effective_content = False
    previous_start = 0
    terminal_eos = False
    for i, token in enumerate(tokens):
        active, terminal_eos, previous_start = _token_supervision(
            token, (i, len(tokens)), (text, start, tokenizer), previous_start
        )
        effective_content |= _effective_content_target(token, i, active, terminal_eos)
        mask.append(active)
    if not effective_content:
        raise ValueError("completion has no content targets after causal shifting")
    return mask, terminal_eos


def _validate_budget(token_count, max_length):
    if max_length is None:
        raise ValueError("completion_only_loss requires a finite max_seq_length budget")
    if token_count > max_length:
        raise ValueError(f"row has {token_count} tokens, exceeding max_seq_length={max_length}")


def completion_tokens(row, tokenizer, max_length):
    """Tokenize exact text without truncation; refuse ambiguous supervision."""
    text, start = _completion_boundary(row)
    tokens = _tokenize_with_offsets(text, tokenizer)
    tokens = _remove_redundant_eos(tokens, text, tokenizer)
    mask, terminal_eos = _completion_mask(tokens, text, start, tokenizer)
    ids = [token.id for token in tokens]
    if not terminal_eos:
        ids.append(tokenizer.eos_token_id)
        mask.append(1)
    _validate_budget(len(ids), max_length)
    labels = [token if active else -100 for token, active in zip(ids, mask, strict=True)]
    return {"input_ids": ids, "completion_mask": mask, "labels": labels}


def _preprocessing_evidence(prepared, max_length):
    digest = hashlib.sha256()
    supervised = 0
    longest = 0
    for row in prepared:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
        supervised += sum(row["completion_mask"][1:])
        longest = max(longest, len(row["input_ids"]))
    return {
        "completion_only_loss": True,
        "boundary_unit": "unicode_code_point",
        "truncation": False,
        "max_seq_length": max_length,
        "rows": len(prepared),
        "max_tokens": longest,
        "shifted_supervised_tokens": supervised,
        "prepared_sha256": digest.hexdigest(),
    }


def _write_preprocessing_evidence(output_dir, evidence):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "completion_preprocessing.json").write_text(json.dumps(evidence, indent=2) + "\n")


def prepare_completion_dataset(dataset, tokenizer, training_args):
    """Preserve metadata, validate every row, and persist preprocessing evidence."""
    if training_args.dataset_text_field != "text":
        raise ValueError("completion_only_loss requires dataset_text_field=text")
    reserved = {"labels", "assistant_masks", "seq_lengths"} & set(dataset.column_names)
    if reserved:
        raise ValueError(f"completion rows contain reserved loss controls: {sorted(reserved)}")
    prepared = dataset.map(
        lambda row: completion_tokens(row, tokenizer, training_args.max_length),
        load_from_cache_file=False,
    )
    evidence = _preprocessing_evidence(prepared, training_args.max_length)
    _write_preprocessing_evidence(training_args.output_dir, evidence)
    return prepared
