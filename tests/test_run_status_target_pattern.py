"""Semantic tests for the bounded PEFT target-pattern subset."""

import itertools
import re

from agoge_forger._run_status_target_pattern import parse_safe_target_pattern


def test_supported_patterns_are_equivalent_to_regex_fullmatch():
    literals = ("a", r"\.", "q")
    bodies = set(literals)
    bodies.update(f"{left}.*{right}" for left, right in itertools.product(literals, repeat=2))
    bodies.update(
        f"{left}.*{middle}.*{right}"
        for left, middle, right in itertools.product(literals, repeat=3)
    )
    patterns = {
        anchored for body in bodies for anchored in (body, f"^{body}", f"{body}$", f"^{body}$")
    }
    values = ("", "a", ".", "q", "aq", "a.q", "aaaq", "q.a", "prefix.q", "a-tail")

    for pattern, value in itertools.product(patterns, values):
        parsed = parse_safe_target_pattern(pattern)
        assert parsed is not None, pattern
        assert parsed.fullmatch(value) is (re.fullmatch(pattern, value) is not None), (
            pattern,
            value,
        )


def test_every_escaped_regex_punctuation_is_treated_literally():
    for punctuation in r"\.^$*+?{}[]()|":
        pattern = f"\\{punctuation}"
        parsed = parse_safe_target_pattern(pattern)

        assert parsed is not None, pattern
        assert parsed.fullmatch(punctuation)
        assert not parsed.fullmatch(f"x{punctuation}")


def test_unsupported_regex_constructs_fail_closed():
    unsupported = (
        "(a+)+$",
        "a|b",
        "[ab]",
        "a+",
        "a?",
        "a{1}",
        r"\d",
        "trailing\\",
        ".*.*",
        "^$",
    )

    assert all(parse_safe_target_pattern(pattern) is None for pattern in unsupported)
