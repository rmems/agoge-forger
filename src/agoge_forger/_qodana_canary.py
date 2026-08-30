"""Temporary canary for Qodana's configured Python language level."""


def normalize_value_for_qodana_canary(
    value: str | bytes | None,
) -> str | None:
    """Exercise PEP 604 unions under the configured Python language level."""
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else value
