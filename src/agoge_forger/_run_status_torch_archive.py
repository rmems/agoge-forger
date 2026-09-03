"""Non-executing structural checks for current PyTorch ZIP serialization."""

from __future__ import annotations

import io
import pickletools  # nosec B403 - disassembles bytes; never executes or loads pickle
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MAX_PICKLE_METADATA_BYTES = 64 * 1024 * 1024
_MARK = object()
_UNKNOWN = object()


@dataclass
class _MappingKeys:
    values: set[str] = field(default_factory=set)


class _NullWriter(io.StringIO):
    def write(self, text: str) -> int:
        """Discard pickle disassembly output."""
        return len(text)


def _archive_root(names: list[str]) -> str | None:
    roots = {name.partition("/")[0] for name in names}
    return roots.pop() if len(roots) == 1 else None


def _pop_mark(stack: list[object]) -> list[object] | None:
    try:
        position = len(stack) - 1 - stack[::-1].index(_MARK)
    except ValueError:
        return None
    values = stack[position + 1 :]
    del stack[position:]
    return values


def _generic_stack_effect(stack: list[object], opcode: Any) -> bool:
    before = opcode.stack_before
    if pickletools.markobject in before:
        if _pop_mark(stack) is None:
            return False
        pop_count = before.index(pickletools.markobject)
    else:
        pop_count = len(before)
    if len(stack) < pop_count:
        return False
    if pop_count:
        del stack[-pop_count:]
    stack.extend([_UNKNOWN] * len(opcode.stack_after))
    return True


def _add_mapping_items(mapping: object, items: list[object]) -> bool:
    if not isinstance(mapping, _MappingKeys) or len(items) % 2:
        return False
    mapping.values.update(key for key in items[::2] if isinstance(key, str))
    return True


def _apply_mapping_creation_opcode(
    stack: list[object],
    opcode: Any,
) -> bool | None:
    name = opcode.name
    if name == "MARK":
        stack.append(_MARK)
        return True
    if name == "EMPTY_DICT":
        stack.append(_MappingKeys())
        return True
    if name == "DICT":
        items = _pop_mark(stack)
        mapping = _MappingKeys()
        if items is None or not _add_mapping_items(mapping, items):
            return False
        stack.append(mapping)
        return True
    return None


def _apply_mapping_update_opcode(stack: list[object], opcode: Any) -> bool | None:
    if opcode.name == "SETITEM":
        if len(stack) < 3:
            return False
        value, key = stack.pop(), stack.pop()
        return _add_mapping_items(stack[-1], [key, value])
    if opcode.name == "SETITEMS":
        items = _pop_mark(stack)
        return items is not None and bool(stack) and _add_mapping_items(stack[-1], items)
    return None


def _apply_memo_opcode(
    stack: list[object],
    memo: dict[int, object],
    opcode: Any,
    argument: Any,
) -> bool | None:
    name = opcode.name
    if name in {"BINPUT", "LONG_BINPUT", "PUT"}:
        if not stack:
            return False
        memo[int(argument)] = stack[-1]
        return True
    if name == "MEMOIZE":
        if not stack:
            return False
        memo[len(memo)] = stack[-1]
        return True
    if name in {"BINGET", "LONG_BINGET", "GET"}:
        value = memo.get(int(argument))
        if value is None:
            return False
        stack.append(value)
        return True
    return None


def _apply_literal_opcode(
    stack: list[object],
    opcode: Any,
    argument: Any,
) -> bool | None:
    name = opcode.name
    if name == "DUP":
        if not stack:
            return False
        stack.append(stack[-1])
        return True
    if isinstance(argument, str) and ("STRING" in name or "UNICODE" in name):
        stack.append(argument)
        return True
    return None


def _apply_pickle_opcode(
    stack: list[object],
    memo: dict[int, object],
    opcode: Any,
    argument: Any,
) -> bool:
    result = _apply_mapping_creation_opcode(stack, opcode)
    if result is not None:
        return result
    result = _apply_mapping_update_opcode(stack, opcode)
    if result is not None:
        return result
    result = _apply_memo_opcode(stack, memo, opcode, argument)
    if result is not None:
        return result
    result = _apply_literal_opcode(stack, opcode, argument)
    if result is not None:
        return result
    return _generic_stack_effect(stack, opcode)


def _top_level_mapping_keys(payload: bytes) -> set[str] | None:
    stack: list[object] = []
    memo: dict[int, object] = {}
    for opcode, argument, _ in pickletools.genops(payload):
        if opcode.name == "STOP":
            result = stack[-1] if stack else None
            return result.values if isinstance(result, _MappingKeys) else None
        if not _apply_pickle_opcode(stack, memo, opcode, argument):
            return None
    return None


def _pickle_metadata_strings(archive: zipfile.ZipFile, name: str) -> set[str] | None:
    with archive.open(name) as stream:
        payload = stream.read(_MAX_PICKLE_METADATA_BYTES + 1)
    if not payload or len(payload) > _MAX_PICKLE_METADATA_BYTES:
        return None
    try:
        pickletools.dis(payload, out=_NullWriter())
    except (ValueError, EOFError):
        return None
    return _top_level_mapping_keys(payload)


def _validated_archive_metadata(
    archive: zipfile.ZipFile,
    *,
    require_data_record: bool,
) -> set[str] | None:
    names = archive.namelist()
    root = _archive_root(names)
    if root is None:
        return None
    data_name = f"{root}/data.pkl"
    required = {data_name, f"{root}/version", f"{root}/.data/serialization_id"}
    if not required.issubset(names) or archive.testzip() is not None:
        return None
    if require_data_record and not any(name.startswith(f"{root}/data/") for name in names):
        return None
    return _pickle_metadata_strings(archive, data_name)


def torch_zip_metadata(path: Path, *, require_data_record: bool = False) -> set[str] | None:
    """Return static pickle strings from a structurally valid PyTorch ZIP."""
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("rb") as handle, zipfile.ZipFile(handle) as archive:
            return _validated_archive_metadata(
                archive,
                require_data_record=require_data_record,
            )
    except (RuntimeError, zipfile.BadZipFile):
        return None
