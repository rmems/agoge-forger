"""Temporary canary for Qodana's configured Python language level."""

from os import PathLike


def normalize_path_for_qodana_canary(
    value: str | PathLike[str] | None,
) -> str | None:
    """Exercise PEP 604 unions and generic path-like annotations."""
    return None if value is None else str(value)
