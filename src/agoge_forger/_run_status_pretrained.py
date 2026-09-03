"""Guarded offline Transformers loading for run-status validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def offline_pretrained(
    factory: Any,
    source: str | Path,
    *,
    revision: str | None = None,
) -> Any:
    loader = getattr(factory, "from_pretrained", None)
    if not callable(loader):
        raise TypeError("from_pretrained is not callable")
    revision_kwarg = {} if revision is None else {"revision": revision}
    return loader(
        source,
        **revision_kwarg,
        local_files_only=True,
        trust_remote_code=False,
    )


def tokenizer_usable(candidate: Path) -> bool:
    try:
        tokenizer = offline_pretrained(AutoTokenizer, candidate)  # nosec B615
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return len(tokenizer) > 0 and any(
        token is not None for token in (tokenizer.pad_token, tokenizer.eos_token)
    )
