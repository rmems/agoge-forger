"""JSON decoding helpers for untrusted metadata and immutable provenance inputs."""

from __future__ import annotations

import json
from typing import Any

_MAX_JSON_DEPTH = 128


def decode_bounded_json(raw: str) -> Any:
    """Reject excessive container nesting before allocating the decoded tree.

    Python 3.14's C decoder can accept very deep JSON on large-stack hosts;
    RecursionError alone is no longer a portable metadata resource limit.
    """
    depth = 0
    quoted = False
    escaped = False
    for character in raw:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "{[":
            depth += 1
            if depth > _MAX_JSON_DEPTH:
                raise ValueError("JSON nesting exceeds 128 levels")
        elif character in "}]":
            depth -= 1
    return json.loads(raw)


class DuplicateJsonKey(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


def decode_json_object(
    raw: bytes,
    coordinate: str,
    *,
    object_label: str = "source row",
) -> dict[str, Any]:
    decoded = _decode_utf8(raw, coordinate)
    value = _load_unique_json(decoded, coordinate)
    if not isinstance(value, dict):
        raise ValueError(f"{coordinate}: {object_label} must be a JSON object")  # noqa: TRY004
    return value


def _decode_utf8(raw: bytes, coordinate: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{coordinate}: source line is not UTF-8") from exc


def _load_unique_json(decoded: str, coordinate: str) -> Any:
    try:
        return json.loads(
            decoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_non_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{coordinate}: invalid JSON: {exc}") from exc
    except DuplicateJsonKey as exc:
        raise ValueError(f"{coordinate}: duplicate JSON object key '{exc.key}'") from exc
    except ValueError as exc:
        raise ValueError(f"{coordinate}: {exc}") from exc


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(key)
        result[key] = value
    return result
