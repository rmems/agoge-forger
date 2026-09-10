"""Loss masks for frozen text with explicit Unicode character boundaries."""

import hashlib
import json
from pathlib import Path


def completion_tokens(row, tokenizer, max_length):
    """Tokenize exact text without truncation; refuse ambiguous supervision."""
    text = row.get("text")
    start = row.get("completion_start_char")
    if not isinstance(text, str) or type(start) is not int or not 0 <= start < len(text):
        raise ValueError("completion_start_char must be an integer offset into nonempty text")
    if not text[start:].strip():
        raise ValueError("completion must be nonempty")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("completion masking requires reliable fast-tokenizer offsets")
    if tokenizer.eos_token_id is None:
        raise ValueError("completion masking requires an EOS token")
    encoded = tokenizer(
        text,
        truncation=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    specials = encoded.get("special_tokens_mask")
    if offsets is None or specials is None or len(offsets) != len(ids):
        raise ValueError("tokenizer lacks reliable offsets")
    mask = []
    content_positions = []
    previous_start = 0
    terminal_eos = False
    for i, (token, (left, right), special) in enumerate(zip(ids, offsets, specials, strict=True)):
        if special and (left, right) == (0, 0):
            if i == 0 and token == tokenizer.bos_token_id:
                mask.append(0)
                continue
            if i == len(ids) - 1 and token == tokenizer.eos_token_id:
                mask.append(1)
                terminal_eos = True
                continue
            raise ValueError("unsupported special token without reliable offsets")
        if not 0 <= left < right <= len(text) or left < previous_start:
            raise ValueError("tokenizer lacks reliable character offsets")
        previous_start = left
        if left < start < right:
            raise ValueError("token crosses completion boundary")
        active = int(left >= start)
        is_eos = token == tokenizer.eos_token_id and text[left:right] == tokenizer.eos_token
        is_control = token in {tokenizer.bos_token_id, tokenizer.pad_token_id}
        if active and is_control and not is_eos:
            raise ValueError("completion contains BOS or padding special token")
        mask.append(active)
        if active and not is_eos:
            content_positions.append(i)
        if i == len(ids) - 1 and is_eos:
            terminal_eos = True
    if not any(i > 0 for i in content_positions):
        raise ValueError("completion has no content targets after causal shifting")
    if not terminal_eos:
        ids.append(tokenizer.eos_token_id)
        mask.append(1)
    if max_length is None or len(ids) > max_length:
        raise ValueError(f"row has {len(ids)} tokens, exceeding max_seq_length={max_length}")
    return {"input_ids": ids, "completion_mask": mask}


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
    digest = hashlib.sha256()
    supervised = 0
    longest = 0
    for row in prepared:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
        supervised += sum(row["completion_mask"][1:])
        longest = max(longest, len(row["input_ids"]))
    evidence = {
        "completion_only_loss": True,
        "boundary_unit": "unicode_code_point",
        "truncation": False,
        "max_seq_length": training_args.max_length,
        "rows": len(prepared),
        "max_tokens": longest,
        "shifted_supervised_tokens": supervised,
        "prepared_sha256": digest.hexdigest(),
    }
    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "completion_preprocessing.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return prepared
