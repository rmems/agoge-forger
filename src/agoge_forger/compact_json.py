"""JSON text handling for compact edits and lossless source metadata."""

import json
from decimal import Decimal

from ._strict_json import decode_json_object
from .split_schema import canonical_json_bytes

_PAIR_FIELDS = {"instruction", "before", "after"}


def _escaped_character(character: str) -> str:
    codepoint = ord(character)
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    return json.dumps(character, ensure_ascii=True)[1:-1]


def _encode_string(value: str, special_tokens) -> str:
    encoded = json.dumps(value, ensure_ascii=False)
    for token in special_tokens:
        encoded = encoded.replace(token, _escaped_character(token[0]) + token[1:])
    return encoded


def encode_edit(edit: dict[str, str], special_tokens=()) -> str:
    """Break literal tokenizer markers with JSON escapes, without changing decoded text."""
    tokens = sorted(set(filter(None, special_tokens)), key=lambda token: (-len(token), token))
    fields = [f"{json.dumps(key)}: {_encode_string(value, tokens)}" for key, value in edit.items()]
    encoded = "{" + ", ".join(fields) + "}"
    if any(token in encoded for token in tokens):
        raise ValueError("tokenizer control markers collide with the JSON representation")
    return encoded


def _require_metadata_precision(raw: bytes, source) -> None:
    exact = json.loads(raw, parse_float=Decimal)
    exact_metadata = {key: value for key, value in exact.items() if key not in _PAIR_FIELDS}
    metadata = {key: value for key, value in source.items() if key not in _PAIR_FIELDS}
    normalized = json.loads(canonical_json_bytes(metadata), parse_float=Decimal)
    if exact_metadata != normalized:
        raise ValueError("encode precision-sensitive metadata numbers as strings before admission")


def compact_json_line(raw: bytes, rendered) -> bytes:
    """Emit freeze-compatible canonical JSON without silently rounding metadata values."""
    source = decode_json_object(raw, "compact source")
    _require_metadata_precision(raw, source)
    return canonical_json_bytes(rendered) + b"\n"
