"""Versioned SEARCH/REPLACE text completions; no file access or code execution."""

import json
from enum import Enum

from ._strict_json import decode_json_object


class CompletionFormat(str, Enum):
    FROZEN_TEXT = "frozen-text-v1"
    SEARCH_REPLACE = "search-replace-json-v1"


def _unique_position(source: str, search: str) -> int:
    position = source.find(search)
    if position < 0:
        raise ValueError("search context does not match source")
    if source.find(search, position + 1) >= 0:
        raise ValueError("search context is ambiguous")
    return position


def apply_search_replace(source: str, completion: str) -> str:
    """Apply exactly one JSON edit to a string, requiring a unique exact context."""
    edit = decode_json_object(completion.encode("utf-8"), "completion")
    if set(edit) != {"search", "replace"}:
        raise ValueError("completion must have exactly search and replace keys")
    search, replacement = edit["search"], edit["replace"]
    if not isinstance(search, str) or not search or not isinstance(replacement, str):
        raise ValueError("search must be nonempty text and replace must be text")
    if search == replacement:
        raise ValueError("completion must change the source")
    position = _unique_position(source, search)
    return source[:position] + replacement + source[position + len(search) :]


def _common_edges(before: list[str], after: list[str]) -> tuple[int, int]:
    prefix = 0
    while prefix < min(len(before), len(after)) and before[prefix] == after[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(before), len(after)) - prefix and before[-1 - suffix] == after[-1 - suffix]
    ):
        suffix += 1
    return prefix, suffix


def search_replace_completion(before: str, after: str) -> str:
    """Render one changed line span with enough unchanged context to identify it."""
    if not isinstance(before, str) or not before or not isinstance(after, str):
        raise ValueError("before must be nonempty text and after must be text")
    if before == after:
        raise ValueError("before and after must differ")
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    prefix, suffix = _common_edges(old, new)
    # Include one unchanged line on either side, then expand to disambiguate.
    left, context_after = max(0, prefix - 1), min(1, suffix)
    while True:
        search = "".join(old[left : len(old) - suffix + context_after])
        replacement = "".join(new[left : len(new) - suffix + context_after])
        try:
            _unique_position(before, search)
        except ValueError:
            left = max(0, left - 1)
            context_after = min(suffix, context_after + 1)
            continue
        break
    completion = json.dumps({"search": search, "replace": replacement}, ensure_ascii=False)
    if apply_search_replace(before, completion) != after:
        raise ValueError("serialized edit does not reconstruct after exactly")
    return completion


def render_compact_row(row):
    """Convert explicit instruction/before/after fields while retaining identity metadata."""
    collisions = {
        "text",
        "messages",
        "completion_start_char",
        "completion_format",
        "source_text",
    } & row.keys()
    if collisions:
        raise ValueError(f"raw repair pair contains rendered fields: {sorted(collisions)}")
    instruction = row.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty text")
    before, after = row.get("before"), row.get("after")
    completion = search_replace_completion(before, after)
    prompt = (
        f"{instruction}\n\nOriginal text (JSON string):\n"
        f"{json.dumps(before, ensure_ascii=False)}\n\n"
        'Return one JSON object with exactly "search" and "replace" string keys. '
        'The nonempty "search" must match exactly once in the original text.\nEdit:\n'
    )
    metadata = {
        key: value for key, value in row.items() if key not in {"instruction", "before", "after"}
    }
    return {
        **metadata,
        "source_text": before,
        "text": prompt + completion,
        "completion_start_char": len(prompt),
        "completion_format": CompletionFormat.SEARCH_REPLACE.value,
    }


def compact_json_line(raw: bytes, rendered) -> bytes:
    """Serialize rendered fields while retaining original metadata number precision.

    ``raw`` must already have passed strict JSON-object validation. Decode each
    top-level value only to locate its end; copy metadata from the source span.
    """
    text = raw.decode("utf-8").strip()
    decoder = json.JSONDecoder()
    fields = []
    index = 1
    while text[index] != "}":
        index = _skip_whitespace(text, index)
        start = index
        key, index = decoder.raw_decode(text, index)
        index = _skip_whitespace(text, index) + 1  # validated colon
        _, index = decoder.raw_decode(text, _skip_whitespace(text, index))
        if key not in {"instruction", "before", "after"}:
            fields.append(text[start:index])
        index = _skip_whitespace(text, index)
        if text[index] != ",":
            break
        index += 1
    generated = {
        key: rendered[key]
        for key in ("text", "completion_start_char", "completion_format", "source_text")
    }
    fields.append(json.dumps(generated, ensure_ascii=False)[1:-1])
    return ("{" + ",".join(fields) + "}\n").encode("utf-8")


def _skip_whitespace(text: str, index: int) -> int:
    while text[index].isspace():
        index += 1
    return index
