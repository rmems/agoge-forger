"""Versioned SEARCH/REPLACE text completions; no file access or code execution."""

import json
from enum import Enum

from ._strict_json import decode_json_object
from .compact_json import encode_edit


class CompletionFormat(str, Enum):
    FROZEN_TEXT = "frozen-text-v1"
    SEARCH_REPLACE = "search-replace-json-v1"


def _require_text(value, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")  # noqa: TRY004 - schema errors use ValueError.


def _require_nonempty_text(value, name: str) -> None:
    _require_text(value, name)
    if not value:
        raise ValueError(f"{name} must be nonempty text")


def _require_change(before: str, after: str) -> None:
    if before == after:
        raise ValueError("edit must change the source")


def _unique_position(source: str, search: str) -> int:
    position = source.find(search)
    if position < 0:
        raise ValueError("search context does not match source")
    if source.find(search, position + 1) >= 0:
        raise ValueError("search context is ambiguous")
    return position


def _parse_edit(completion: str) -> tuple[str, str]:
    edit = decode_json_object(completion.encode("utf-8"), "completion")
    if set(edit) != {"search", "replace"}:
        raise ValueError("completion must have exactly search and replace keys")
    search, replacement = edit["search"], edit["replace"]
    _require_nonempty_text(search, "search")
    _require_text(replacement, "replace")
    _require_change(search, replacement)
    return search, replacement


def apply_search_replace(source: str, completion: str) -> str:
    """Apply exactly one JSON edit to a string, requiring a unique exact context."""
    search, replacement = _parse_edit(completion)
    position = _unique_position(source, search)
    return source[:position] + replacement + source[position + len(search) :]


def _common_prefix(before: list[str], after: list[str]) -> int:
    for index, (old, new) in enumerate(zip(before, after)):
        if old != new:
            return index
    return min(len(before), len(after))


def _common_edges(before: list[str], after: list[str]) -> tuple[int, int]:
    prefix = _common_prefix(before, after)
    suffix = _common_prefix(before[prefix:][::-1], after[prefix:][::-1])
    return prefix, suffix


def _unique_context(source: str, lines: list[str], prefix: int, suffix: int):
    context = 1
    while True:
        left = max(0, prefix - context)
        right_context = min(suffix, context)
        search = "".join(lines[left : len(lines) - suffix + right_context])
        try:
            _unique_position(source, search)
        except ValueError:
            # Geometric growth bounds rescans; at full-source context the match is unique.
            context *= 2
            continue
        return search, left, right_context


def _edit_span(before: str, after: str) -> dict[str, str]:
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    prefix, suffix = _common_edges(old, new)
    search, left, right_context = _unique_context(before, old, prefix, suffix)
    replacement = "".join(new[left : len(new) - suffix + right_context])
    return {"search": search, "replace": replacement}


def search_replace_completion(before: str, after: str, special_tokens=()) -> str:
    """Render one changed line span with enough unchanged context to identify it."""
    _require_nonempty_text(before, "before")
    _require_text(after, "after")
    _require_change(before, after)
    completion = encode_edit(_edit_span(before, after), special_tokens)
    if apply_search_replace(before, completion) != after:
        raise ValueError("serialized edit does not reconstruct after exactly")
    return completion


def _validate_pair_fields(row) -> None:
    collisions = {
        "text",
        "messages",
        "completion_start_char",
        "completion_format",
        "source_text",
    } & row.keys()
    if collisions:
        raise ValueError(f"raw repair pair contains rendered fields: {sorted(collisions)}")
    _require_nonempty_text(row.get("instruction"), "instruction")
    if not row["instruction"].strip():
        raise ValueError("instruction must be nonempty text")


def _repair_prompt(instruction: str, before: str) -> str:
    return (
        f"{instruction}\n\nOriginal text (JSON string):\n"
        f"{json.dumps(before, ensure_ascii=False)}\n\n"
        'Return one JSON object with exactly "search" and "replace" string keys. '
        'The nonempty "search" must match exactly once in the original text.\nEdit:\n'
    )


def render_compact_row(row, special_tokens=()):
    """Convert explicit pairs, escaping the selected tokenizer's control markers."""
    _validate_pair_fields(row)
    before, after = row.get("before"), row.get("after")
    completion = search_replace_completion(before, after, special_tokens)
    prompt = _repair_prompt(row["instruction"], before)
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
