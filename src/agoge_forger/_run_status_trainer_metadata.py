"""Bounded Transformers Trainer metadata validation."""

from __future__ import annotations

import json
import os
import re
import stat
from errno import ELOOP
from pathlib import Path
from typing import Any

from transformers import TrainerState

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
_MAX_TRAINER_STATE_BYTES = 4 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


def _bounded_regular_file(path: Path, max_bytes: int) -> bytes | None:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return None
    except (FileNotFoundError, NotADirectoryError):
        return None
    flags = os.O_RDONLY
    for flag_name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, flag_name, 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == ELOOP:
            return None
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        content = bytearray()
        while len(content) <= max_bytes:
            remaining = max_bytes + 1 - len(content)
            block = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not block:
                break
            content.extend(block)
        return bytes(content) if len(content) <= max_bytes else None
    finally:
        os.close(descriptor)


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        content = _bounded_regular_file(path, _MAX_TRAINER_STATE_BYTES)
        if content is None:
            return None
        payload: Any = json.loads(content.decode("utf-8"))
    except (MemoryError, RecursionError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _valid_int(value: Any, *, minimum: int | None = None) -> bool:
    return bool(
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _trainer_metadata_usable(payload: dict[str, Any], step: int) -> bool:
    try:
        TrainerState(**payload)
    except TypeError:
        return False
    global_step = payload.get("global_step")
    train_batch_size = payload.get("train_batch_size")
    if not _valid_int(global_step) or not _valid_int(train_batch_size, minimum=1):
        return False
    return global_step == step


def _trainer_state_step(checkpoint: Path) -> int | None:
    payload = _json_object(checkpoint / "trainer_state.json")
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if payload is None or match is None:
        return None
    step = int(match.group(1))
    return step if _trainer_metadata_usable(payload, step) else None
