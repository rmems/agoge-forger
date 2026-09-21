"""Secret-query sanitization and hidden-reasoning stripping for Hermes traces."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

SECRET_QUERY_KEYS = frozenset(
    {
        "access_token",
        "token",
        "client_secret",
        "client_id",
        "authorization",
        "api_key",
        "password",
        "secret",
    }
)

HIDDEN_KEYS = frozenset(
    {
        "reasoning",
        "hidden_reasoning",
        "thinking",
        "thought",
        "thoughts",
        "analysis",
        "scratchpad",
        "chain_of_thought",
        "chain-of-thought",
        "cot",
    }
)

_HIDDEN_TAG = (
    r"(?:think|thought|thinking|analysis|reasoning|scratchpad|chain-of-thought|cot)"
)
_THINK_RE = re.compile(
    rf"<({_HIDDEN_TAG})\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_HIDDEN_OPEN_RE = re.compile(rf"<({_HIDDEN_TAG})\b", re.IGNORECASE)
_HIDDEN_CLOSE_RE = re.compile(rf"</({_HIDDEN_TAG})\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_HEADER_SECRET_RE = re.compile(
    r"(?i)\b(?:authorization|x-api-key|api[-_]?key)\b\s*[:=]"
)


class UnsafeUrlError(ValueError):
    """Raised when a URL cannot be sanitized safely."""


def sanitize_url(url: str) -> str:
    """Drop credentials, secret query keys, and fragments; sort remaining query."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
        username = parts.username
        password = parts.password
    except ValueError as exc:
        raise UnsafeUrlError("invalid_url") from exc
    if username or password:
        raise UnsafeUrlError("credential_url")
    for segment in (parts.path or "").split("/"):
        if segment.casefold() in SECRET_QUERY_KEYS:
            raise UnsafeUrlError("credential_url")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port else host
    query_pairs = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in SECRET_QUERY_KEYS
    ]
    query_pairs.sort(key=lambda item: (item[0], item[1]))
    scheme = parts.scheme or "https"
    return urllib.parse.urlunsplit(
        (scheme, netloc, parts.path or "", urllib.parse.urlencode(query_pairs), "")
    )


def sanitize_query_secrets(value: Any) -> Any:
    """Rewrite URLs embedded in strings so secret query keys cannot leak."""
    if isinstance(value, str):
        return _URL_RE.sub(lambda match: sanitize_url(match.group(0)), value)
    if isinstance(value, list):
        return [sanitize_query_secrets(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_query_secrets(item) for key, item in value.items()}
    return value


def is_hidden_key(key: Any) -> bool:
    return str(key).casefold() in HIDDEN_KEYS


def hidden_markup_remains(value: str) -> bool:
    """True when a hidden tag is unclosed or only partially stripped."""
    stripped = _THINK_RE.sub("", value)
    return (
        _HIDDEN_OPEN_RE.search(stripped) is not None
        or _HIDDEN_CLOSE_RE.search(stripped) is not None
    )


def contains_header_secret(value: str) -> bool:
    return _HEADER_SECRET_RE.search(value) is not None


def strip_hidden_reasoning(value: Any) -> Any:
    """Remove hidden-reasoning fields and producer think-tags from trainable views."""
    if isinstance(value, str):
        return _THINK_RE.sub("", value)
    if isinstance(value, list):
        return [strip_hidden_reasoning(item) for item in value]
    if isinstance(value, dict):
        return {
            key: strip_hidden_reasoning(item)
            for key, item in value.items()
            if not is_hidden_key(key)
        }
    return value
