"""Bounded target-pattern parsing without a backtracking regex engine."""

from __future__ import annotations

from dataclasses import dataclass

_MAX_TARGET_PATTERN_LENGTH = 1_024
_ESCAPABLE_CHARACTERS = frozenset(r"\.^$*+?{}[]()|")
_UNSUPPORTED_CHARACTERS = frozenset(".^$*+?{}[]()|")


def _prefix_end(value: str, prefix: str) -> int | None:
    if not value.startswith(prefix):
        return None
    return len(prefix)


def _ordered_parts_end(value: str, parts: tuple[str, ...], position: int) -> int | None:
    for part in parts:
        match_position = value.find(part, position)
        if match_position < 0:
            return None
        position = match_position + len(part)
    return position


def _suffix_matches(value: str, suffix: str, position: int) -> bool:
    if not suffix:
        return True
    return value.endswith(suffix) and len(value) - len(suffix) >= position


@dataclass(frozen=True)
class SafeTargetPattern:
    """A full-match pattern made only from literals and ``.*`` wildcards."""

    parts: tuple[str, ...]

    def fullmatch(self, value: str) -> bool:
        if len(self.parts) == 1:
            return value == self.parts[0]
        position = _prefix_end(value, self.parts[0])
        if position is None:
            return False
        position = _ordered_parts_end(value, self.parts[1:-1], position)
        return position is not None and _suffix_matches(value, self.parts[-1], position)


@dataclass(frozen=True)
class _PatternToken:
    literal: str | None
    next_position: int


def _has_unescaped_trailing_anchor(target: str) -> bool:
    prefix = target.removesuffix("$")
    preceding_backslashes = len(prefix) - len(prefix.rstrip("\\"))
    return target.endswith("$") and preceding_backslashes % 2 == 0


def _without_anchors(target: str) -> str:
    target = target.removeprefix("^")
    return target[:-1] if target and _has_unescaped_trailing_anchor(target) else target


def _escaped_token(target: str, position: int) -> _PatternToken | None:
    literal_position = position + 1
    if literal_position >= len(target):
        return None
    literal = target[literal_position]
    if literal not in _ESCAPABLE_CHARACTERS:
        return None
    return _PatternToken(literal, literal_position + 1)


def _next_token(target: str, position: int) -> _PatternToken | None:
    if target.startswith(".*", position):
        return _PatternToken(None, position + 2)
    character = target[position]
    if character == "\\":
        return _escaped_token(target, position)
    if character in _UNSUPPORTED_CHARACTERS:
        return None
    return _PatternToken(character, position + 1)


def _append_token(parts: list[list[str]], token: _PatternToken) -> bool:
    if token.literal is not None:
        parts[-1].append(token.literal)
        return True
    if len(parts) > 1 and not parts[-1]:
        return False
    parts.append([])
    return True


def _accepted_token(target: str, position: int, parts: list[list[str]]) -> _PatternToken | None:
    token = _next_token(target, position)
    return token if token is not None and _append_token(parts, token) else None


def _safe_pattern_parts(target: str) -> tuple[str, ...] | None:
    target = _without_anchors(target)
    if not target:
        return None
    parts: list[list[str]] = [[]]
    position = 0
    while position < len(target):
        token = _accepted_token(target, position, parts)
        if token is None:
            return None
        position = token.next_position
    return tuple(map("".join, parts))


def parse_safe_target_pattern(target: str) -> SafeTargetPattern | None:
    """Parse the supported PEFT regex subset, failing closed otherwise."""
    if not target or len(target) > _MAX_TARGET_PATTERN_LENGTH:
        return None
    parts = _safe_pattern_parts(target)
    return SafeTargetPattern(parts) if parts is not None else None
