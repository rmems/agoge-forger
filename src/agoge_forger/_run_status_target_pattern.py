"""Bounded target-pattern parsing without a backtracking regex engine."""

from __future__ import annotations

from dataclasses import dataclass

_MAX_TARGET_PATTERN_LENGTH = 1_024
_ESCAPABLE_CHARACTERS = frozenset(r"\.^$*+?{}[]()|")
_UNSUPPORTED_CHARACTERS = frozenset(".^$*+?{}[]()|")


@dataclass(frozen=True)
class SafeTargetPattern:
    """A full-match pattern made only from literals and ``.*`` wildcards."""

    parts: tuple[str, ...]

    def fullmatch(self, value: str) -> bool:
        if len(self.parts) == 1:
            return value == self.parts[0]
        position = 0
        first = self.parts[0]
        if first:
            if not value.startswith(first):
                return False
            position = len(first)
        for part in self.parts[1:-1]:
            match_position = value.find(part, position)
            if match_position < 0:
                return False
            position = match_position + len(part)
        last = self.parts[-1]
        if not last:
            return True
        return value.endswith(last) and len(value) - len(last) >= position


def _has_unescaped_trailing_anchor(target: str) -> bool:
    prefix = target.removesuffix("$")
    preceding_backslashes = len(prefix) - len(prefix.rstrip("\\"))
    return target.endswith("$") and preceding_backslashes % 2 == 0


def _safe_pattern_parts(target: str) -> tuple[str, ...] | None:
    target = target.removeprefix("^")
    if target and _has_unescaped_trailing_anchor(target):
        target = target[:-1]
    parts: list[list[str]] = [[]]
    valid = bool(target)
    position = 0
    while valid and position < len(target):
        character = target[position]
        if target.startswith(".*", position):
            valid = bool(parts[-1]) or len(parts) == 1
            parts.append([])
            position += 2
        elif character == "\\":
            position += 1
            valid = position < len(target) and target[position] in _ESCAPABLE_CHARACTERS
            if valid:
                parts[-1].append(target[position])
                position += 1
        elif character in _UNSUPPORTED_CHARACTERS:
            valid = False
        else:
            parts[-1].append(character)
            position += 1
    return tuple("".join(part) for part in parts) if valid else None


def parse_safe_target_pattern(target: str) -> SafeTargetPattern | None:
    """Parse the supported PEFT regex subset, failing closed otherwise."""
    if not target or len(target) > _MAX_TARGET_PATTERN_LENGTH:
        return None
    parts = _safe_pattern_parts(target)
    return SafeTargetPattern(parts) if parts is not None else None
