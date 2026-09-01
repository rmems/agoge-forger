"""Strict JSON decoding helpers for immutable provenance inputs."""

from __future__ import annotations

import json
from typing import Any


class DuplicateJsonKey(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


def decode_json_object(raw: bytes, coordinate: str) -> dict[str, Any]:
    decoded = _decode_utf8(raw, coordinate)
    value = _load_unique_json(decoded, coordinate)
    if not isinstance(value, dict):
        raise ValueError(f"{coordinate}: source row must be a JSON object")  # noqa: TRY004
    return value


def _decode_utf8(raw: bytes, coordinate: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{coordinate}: source line is not UTF-8") from exc


def _load_unique_json(decoded: str, coordinate: str) -> Any:
    try:
        return json.loads(decoded, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{coordinate}: invalid JSON: {exc}") from exc
    except DuplicateJsonKey as exc:
        raise ValueError(f"{coordinate}: duplicate JSON object key '{exc.key}'") from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(key)
        result[key] = value
    return result
