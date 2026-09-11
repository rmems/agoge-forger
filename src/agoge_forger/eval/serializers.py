"""Pinned prompt serializer for the code-repair held-out canary.

The bound callable must stay free of module globals and non-pure builtins so
``SerializerBinding`` can fingerprint it. Row validation lives in the harness.
"""

from collections.abc import Mapping
from typing import Any

SERIALIZER_ID = "code-repair-prompt-v1"
SERIALIZER_VERSION = "1"


def code_repair_prompt(row: Mapping[str, Any]) -> str:
    """Render the prompt prefix from frozen ``text`` and ``completion_start_char``."""

    return str(row["text"])[: int(row["completion_start_char"])]


code_repair_prompt.serializer_id = SERIALIZER_ID  # type: ignore[attr-defined]
code_repair_prompt.serializer_version = SERIALIZER_VERSION  # type: ignore[attr-defined]
