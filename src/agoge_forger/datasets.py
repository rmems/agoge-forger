import hashlib
import io
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .config import normalize_revision
from .logging import logger
from .path_safety import resolve_existing_path
from .split_schema import sha256_file

_MESSAGE_ROLES = ("user", "assistant", "system", "tool")


def _validate_message(m, index, msg_i):
    if not isinstance(m, dict):
        # Schema errors stay ValueError (1-indexed line) like other normalize_row paths.
        raise ValueError(f"Line {index}: messages[{msg_i}] must be an object.")  # noqa: TRY004
    if not {"role", "content"} <= m.keys():
        raise ValueError(f"Line {index}: messages must contain 'role' and 'content'.")
    if m["role"] not in _MESSAGE_ROLES:
        raise ValueError(f"Line {index}: invalid role '{m['role']}'.")
    if not isinstance(m["content"], str):
        raise ValueError(  # noqa: TRY004
            f"Line {index}: messages[{msg_i}].content must be a string."
        )


def _normalize_messages(row, tokenizer, index):
    if not isinstance(row["messages"], list):
        raise ValueError(f"Line {index}: 'messages' must be a list.")  # noqa: TRY004
    for msg_i, m in enumerate(row["messages"], 1):
        _validate_message(m, index, msg_i)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in row["messages"]])
    return {"text": text}


def _normalize_instruction(row, index):
    if not isinstance(row["instruction"], str):
        raise ValueError(f"Line {index}: 'instruction' must be a string.")  # noqa: TRY004
    text = f"Instruction: {row['instruction']}\n"
    if row.get("input"):
        text += f"Input: {row['input']}\n"
    text += f"Output: {row.get('output', '')}"
    return {"text": text}


def _normalize_text(row, index):
    if not isinstance(row["text"], str):
        raise ValueError(f"Line {index}: 'text' field must be a string.")  # noqa: TRY004
    return row


def normalize_row(row, tokenizer=None, index=0):
    if not isinstance(row, dict):
        # Schema path uses ValueError with 1-indexed line (not bare TypeError).
        raise ValueError(f"Line {index}: JSONL row must be an object.")  # noqa: TRY004
    if "text" in row:
        return _normalize_text(row, index)
    if "messages" in row:
        return _normalize_messages(row, tokenizer, index)
    if "instruction" in row:
        return _normalize_instruction(row, index)
    raise ValueError(
        f"Line {index}: Unknown format. Must contain 'text', 'messages', or 'instruction'."
    )


@dataclass(frozen=True)
class _JsonlDatasetRequest:
    path: str
    sha256: str
    tokenizer: Any


class _HashingReader(io.RawIOBase):
    """Raw reader that hashes exactly the bytes handed to the text decoder."""

    def __init__(self, handle: BinaryIO) -> None:
        super().__init__()
        self._handle = handle
        self.digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self._handle.read(len(buffer))
        size = len(data)
        buffer[:size] = data
        self.digest.update(data)
        return size


def _iter_jsonl_objects(lines: Iterable[str]) -> Iterator[tuple[int, Any]]:
    """Yield ``(line_number, decoded_row)`` for every non-blank JSONL line."""
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {i}: Invalid JSON - {e}") from e
        yield i, row


def _generate_jsonl_rows(request: _JsonlDatasetRequest) -> Iterator[dict[str, Any]]:
    with Path(request.path).open("rb") as handle:
        reader = _HashingReader(handle)
        # Same decoding / universal-newline semantics as ``open(path, "r")``.
        rows_file = io.TextIOWrapper(io.BufferedReader(reader))
        for i, row in _iter_jsonl_objects(rows_file):
            yield normalize_row(row, request.tokenizer, index=i)
        # Fail closed (nothing is committed to the cache) if the bytes read
        # differ from the digest the cache identity was keyed on.
        if reader.digest.hexdigest() != request.sha256:
            raise ValueError(
                f"Dataset file changed after cache identity was established: {request.path}"
            )


def _unwrap_value_error(exc: BaseException) -> None:
    """Re-raise the first ``ValueError`` HF wrapped in ``DatasetGenerationError``."""
    cause = exc.__cause__
    while cause is not None:
        if isinstance(cause, ValueError):
            raise cause from None
        cause = cause.__cause__


def load_jsonl_dataset(path: str, tokenizer=None):
    """Load JSONL via ``Dataset.from_generator`` (disk-backed Arrow cache).

    Empty / blank-only files raise a clear ``ValueError`` without switching to
    an all-in-memory ``from_list`` path that doubles peak RAM on large corpora.
    Schema ``ValueError``s from ``normalize_row`` are re-raised unwrapped (HF
    wraps them in ``DatasetGenerationError``). The file's sha256 is part of the
    cache identity and is re-verified while generating, so a file rewritten at
    the same path is never served from a stale Arrow cache.
    """
    from datasets.exceptions import DatasetGenerationError

    from datasets import Dataset  # type: ignore[attr-defined]

    dataset_path = resolve_existing_path(path, must_be_file=True)

    # Cheap empty check: avoid from_generator SchemaInferenceError on zero rows.
    with dataset_path.open("r") as f:
        if not any(line.strip() for line in f):
            raise ValueError(
                f"Dataset is empty (no valid JSONL rows after skipping blanks): {dataset_path}"
            )

    # The file digest is part of the HF generator cache identity, so rewriting
    # the file at the same path cannot silently reuse stale cached Arrow rows
    # (mirrors split_loaders.load_frozen_dataset).
    request = _JsonlDatasetRequest(
        path=str(dataset_path),
        sha256=sha256_file(dataset_path),
        tokenizer=tokenizer,
    )
    try:
        dataset = Dataset.from_generator(_generate_jsonl_rows, gen_kwargs={"request": request})
    except DatasetGenerationError as exc:
        _unwrap_value_error(exc)
        raise
    # A cache hit skips the generator and its digest check, so confirm the file still
    # holds the bytes the cache was keyed on before returning those rows.
    _require_unchanged(dataset_path, request.sha256)
    return dataset


def _require_unchanged(dataset_path: Path, expected_sha256: str) -> None:
    if sha256_file(dataset_path) != expected_sha256:
        raise ValueError(
            f"Dataset file changed after cache identity was established: {dataset_path}"
        )


def iter_normalized_rows(path: str):
    """Yield ``(line_number, normalized_row)`` without needing a tokenizer.

    ``normalize_row`` picks its output keys from the row *format* and never from
    the tokenizer, so preflight can see exactly what ``load_jsonl_dataset``
    would produce without first paying for ``load_base_model``. Only the
    ``messages`` chat-template *text* depends on the tokenizer; the keys do not.
    """
    dataset_path = resolve_existing_path(path, must_be_file=True)

    with dataset_path.open("r") as f:
        for i, row in _iter_jsonl_objects(f):
            yield i, normalize_row(row, None, index=i)


def dataset_stats(
    path: str,
    model_id: str,
    trust_remote_code: bool = False,
    *,
    revision: str | None = None,
):
    # Lazy import: data-only consumers (consumer_contract, split_materialize)
    # import this module and must not pull transformers/torch, HF datasets, or
    # numpy at import time.
    import numpy as np
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer for dataset stats...")
    # Tokenizer-only token counting; --revision pins the download when given,
    # and the unpinned default matches load_base_model's existing behavior.
    # noinspection PyNoneFunctionAssignment
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 - pinned via --revision when set.
        model_id,
        trust_remote_code=trust_remote_code,
        revision=normalize_revision(revision),
    )

    dataset = load_jsonl_dataset(path, tokenizer)
    lengths: list[int] = []

    for row in dataset:
        # noinspection PyCallingNonCallable
        tokens = tokenizer(row["text"])["input_ids"]
        lengths.append(len(tokens))

    if not lengths:
        raise ValueError(f"Dataset is empty (no tokenizable rows after load): {path}")

    arr = np.asarray(lengths, dtype=np.int64)
    logger.info(f"Dataset Rows: {len(arr)}")
    logger.info(f"Max Tokens: {arr.max()}")
    logger.info(f"Mean Tokens: {arr.mean():.2f}")
    logger.info(f"Min Tokens: {arr.min()}")
