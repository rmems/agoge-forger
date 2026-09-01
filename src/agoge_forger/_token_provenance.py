"""Fail-closed provenance bindings for token-statistics callables."""

from __future__ import annotations

import builtins
import inspect
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import CodeType
from typing import Any

from tokenizers import AddedToken
from transformers import PreTrainedTokenizerFast

from .split_schema import (
    Serializer,
    TokenizerLike,
    canonical_json_bytes,
    sha256_bytes,
)

_UNHANDLED = object()
_TOKENIZER_RUNTIME_FIELDS = (
    "clean_up_tokenization_spaces",
    "model_max_length",
    "padding_side",
    "split_special_tokens",
    "truncation_side",
)
_PURE_BUILTINS = frozenset({"bool", "float", "int", "len", "list", "str"})
_TOKENIZER_IMPLEMENTATION_LABEL = "tokenizer implementation"


@dataclass(frozen=True)
class TokenizerBinding:
    """A tokenizer callable inseparably bound to immutable provenance."""

    implementation: TokenizerLike
    expected_tokenizer_id: str | None = None
    expected_tokenizer_revision: str | None = None
    expected_tokenizer_sha256: str | None = None
    tokenizer_id: str = field(init=False)
    tokenizer_revision: str = field(init=False)
    tokenizer_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_callable(self.implementation, "tokenizer")
        tokenizer_id, tokenizer_revision = derive_tokenizer_provenance(self.implementation)
        tokenizer_sha256 = derive_tokenizer_sha256(self.implementation)
        _require_expected(self.expected_tokenizer_id, tokenizer_id, "tokenizer_id")
        _require_expected(
            self.expected_tokenizer_revision,
            tokenizer_revision,
            "tokenizer_revision",
        )
        _require_expected(
            self.expected_tokenizer_sha256,
            tokenizer_sha256,
            "tokenizer_sha256",
        )
        object.__setattr__(self, "tokenizer_id", tokenizer_id)
        object.__setattr__(self, "tokenizer_revision", tokenizer_revision)
        object.__setattr__(self, "tokenizer_sha256", tokenizer_sha256)

    def __call__(self, text: str) -> Any:
        return self.implementation(text)


@dataclass(frozen=True)
class SerializerBinding:
    """A serializer callable inseparably bound to its versioned digest."""

    implementation: Serializer
    expected_serializer_id: str | None = None
    expected_serializer_version: str | None = None
    expected_serializer_sha256: str | None = None
    serializer_id: str = field(init=False)
    serializer_version: str = field(init=False)
    serializer_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_callable(self.implementation, "serializer")
        serializer_id, serializer_version, serializer_sha256 = derive_serializer_provenance(
            self.implementation
        )
        _require_expected(self.expected_serializer_id, serializer_id, "serializer_id")
        _require_expected(
            self.expected_serializer_version,
            serializer_version,
            "serializer_version",
        )
        _require_expected(
            self.expected_serializer_sha256,
            serializer_sha256,
            "serializer_sha256",
        )
        object.__setattr__(self, "serializer_id", serializer_id)
        object.__setattr__(self, "serializer_version", serializer_version)
        object.__setattr__(self, "serializer_sha256", serializer_sha256)

    def __call__(self, row: Mapping[str, Any]) -> str:
        return self.implementation(row)


def derive_tokenizer_provenance(tokenizer: TokenizerLike) -> tuple[str, str]:
    tokenizer_id = _require_identifier(getattr(tokenizer, "name_or_path", None), "name_or_path")
    init_kwargs = _tokenizer_init_kwargs(tokenizer)
    resolved = _resolved_tokenizer_commit(tokenizer, init_kwargs)
    if resolved is None:
        resolved = _unique_revision(
            (getattr(tokenizer, "revision", None), init_kwargs.get("revision")),
            "tokenizer revision",
        )
    if resolved is None:
        raise ValueError("tokenizer does not expose an immutable resolved revision")
    _require_immutable_revision(resolved, "tokenizer revision")
    return tokenizer_id, resolved


def _tokenizer_init_kwargs(tokenizer: TokenizerLike) -> Mapping[str, Any]:
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, Mapping):
        raise TypeError("tokenizer init_kwargs provenance must be a mapping")
    return init_kwargs


def _resolved_tokenizer_commit(
    tokenizer: TokenizerLike, init_kwargs: Mapping[str, Any]
) -> str | None:
    return _unique_revision(
        (
            getattr(tokenizer, "_commit_hash", None),
            getattr(tokenizer, "commit_hash", None),
            init_kwargs.get("_commit_hash"),
            init_kwargs.get("commit_hash"),
        ),
        "resolved tokenizer commit",
    )


def _unique_revision(values: Sequence[Any], label: str) -> str | None:
    present = _present_revisions(values)
    if not present:
        return None
    return _require_unique_revision(present, label)


def _present_revisions(values: Sequence[Any]) -> list[Any]:
    return [value for value in values if value is not None and value != ""]


def _require_unique_revision(values: Sequence[Any], label: str) -> str:
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{label} values must be strings")
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"conflicting {label} values: {sorted(unique)!r}")
    return next(iter(unique))


def derive_tokenizer_sha256(tokenizer: TokenizerLike) -> str:
    tokenizer_type = type(tokenizer)
    target = tokenizer_type.__call__
    code = getattr(target, "__code__", None)
    payload: dict[str, Any] = {
        "class": f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}",
    }
    backend_to_str = getattr(getattr(tokenizer, "backend_tokenizer", None), "to_str", None)
    if isinstance(tokenizer, PreTrainedTokenizerFast):
        if not _is_exact_fast_tokenizer_type(tokenizer_type):
            raise TypeError("fast tokenizer subclasses cannot be fingerprinted fail-closed")
        if not callable(backend_to_str):
            raise TypeError("fast tokenizer does not expose canonical backend serialization")
        payload.update(_fast_tokenizer_state(tokenizer, backend_to_str, code))
    else:
        payload.update(_python_tokenizer_payload(tokenizer, code))
    return sha256_bytes(canonical_json_bytes(payload))


def _is_exact_fast_tokenizer_type(tokenizer_type: type[Any]) -> bool:
    return tokenizer_type is PreTrainedTokenizerFast


def _fast_tokenizer_state(
    tokenizer: TokenizerLike,
    backend_to_str: Callable[[], Any],
    code: CodeType | None,
) -> dict[str, Any]:
    backend_state = _fast_tokenizer_backend_state(backend_to_str)
    get_vocab = _fast_tokenizer_vocabulary_getter(tokenizer)
    special_tokens = _fast_tokenizer_special_tokens(tokenizer)
    payload = {
        "backend": backend_state,
        "vocabulary": _fingerprint_value(get_vocab()),
        "init_kwargs": _fingerprint_value(_tokenizer_init_kwargs(tokenizer)),
        "special_tokens": _fingerprint_value(special_tokens),
        "chat_template": _fingerprint_value(getattr(tokenizer, "chat_template", None)),
        "runtime_configuration": _tokenizer_runtime_configuration(tokenizer),
        "python_instance": _fast_tokenizer_instance_state(tokenizer),
    }
    return _with_callable_code(payload, code)


def _fast_tokenizer_backend_state(backend_to_str: Callable[[], Any]) -> str:
    backend_state = backend_to_str()
    if not isinstance(backend_state, str) or not backend_state:
        raise ValueError("tokenizer backend serialization must be a non-empty string")
    return backend_state


def _fast_tokenizer_vocabulary_getter(tokenizer: TokenizerLike) -> Callable[[], Any]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise TypeError("fast tokenizer does not expose a verifiable vocabulary")
    return get_vocab


def _fast_tokenizer_special_tokens(tokenizer: TokenizerLike) -> Mapping[Any, Any]:
    special_tokens = getattr(tokenizer, "special_tokens_map", None)
    if not isinstance(special_tokens, Mapping):
        raise TypeError("fast tokenizer does not expose canonical special-token state")
    return special_tokens


def _with_callable_code(payload: dict[str, Any], code: CodeType | None) -> dict[str, Any]:
    if code is not None:
        payload["callable_code"] = _code_payload(code)
    return payload


def _fast_tokenizer_instance_state(tokenizer: TokenizerLike) -> Any:
    state = {
        name: value
        for name, value in _tokenizer_instance_state(tokenizer).items()
        if name != "_tokenizer"
    }
    return _fingerprint_value(state)


def _tokenizer_runtime_configuration(tokenizer: TokenizerLike) -> Any:
    state = {
        name: getattr(tokenizer, name)
        for name in _TOKENIZER_RUNTIME_FIELDS
        if hasattr(tokenizer, name)
    }
    return _fingerprint_value(state)


def _python_tokenizer_payload(tokenizer: TokenizerLike, code: CodeType | None) -> dict[str, Any]:
    if code is None:
        raise ValueError("tokenizer implementation code cannot be derived")
    return {
        "callable_code": _code_payload(code),
        "python_state": _fingerprint_value(_python_tokenizer_state(tokenizer)),
    }


def _python_tokenizer_state(tokenizer: TokenizerLike) -> dict[str, Any]:
    return {
        "classes": _tokenizer_class_hierarchy(type(tokenizer)),
        "instance": _tokenizer_instance_state(tokenizer),
        "slots": _tokenizer_slot_state(tokenizer),
    }


def _tokenizer_instance_state(tokenizer: TokenizerLike) -> Mapping[str, Any]:
    namespace = getattr(tokenizer, "__dict__", None)
    if namespace is None:
        return {}
    if not isinstance(namespace, Mapping):
        raise TypeError("tokenizer instance state is not a canonical mapping")
    return namespace


def _tokenizer_class_hierarchy(tokenizer_type: type[Any]) -> dict[str, Any]:
    return {
        _class_identifier(base): {
            "code": _class_code(base),
            "state": _class_state(base),
        }
        for base in tokenizer_type.__mro__
        if base is not object
    }


def _class_identifier(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _class_code(value: type[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, member in vars(value).items():
        code = _class_member_code(member)
        if code is not None:
            result[name] = code
    return result


def _class_member_code(value: Any) -> Any:
    if inspect.isfunction(value):
        return _pure_function_code(value, _TOKENIZER_IMPLEMENTATION_LABEL)
    if isinstance(value, (staticmethod, classmethod)):
        return _pure_function_code(value.__func__, _TOKENIZER_IMPLEMENTATION_LABEL)
    if isinstance(value, property):
        return _property_code(value)
    return None


def _property_code(value: property) -> dict[str, Any]:
    accessors = {"get": value.fget, "set": value.fset, "delete": value.fdel}
    return {
        name: _pure_function_code(accessor, _TOKENIZER_IMPLEMENTATION_LABEL)
        for name, accessor in accessors.items()
        if accessor is not None
    }


def _pure_function_code(function: Callable[..., Any], label: str) -> dict[str, Any]:
    _require_no_external_dependencies(function, label)
    return _code_payload(function.__code__)


def _class_state(value: type[Any]) -> dict[str, Any]:
    return {name: member for name, member in vars(value).items() if _is_class_state(name, member)}


def _is_class_state(name: str, value: Any) -> bool:
    if name.startswith("__") or _class_member_code(value) is not None:
        return False
    return not inspect.isdatadescriptor(value)


def _tokenizer_slot_state(tokenizer: TokenizerLike) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tokenizer_type in type(tokenizer).__mro__:
        for slot in _slot_names(tokenizer_type):
            if slot not in {"__dict__", "__weakref__"} and hasattr(tokenizer, slot):
                result[f"{_class_identifier(tokenizer_type)}.{slot}"] = getattr(tokenizer, slot)
    return result


def _slot_names(value: type[Any]) -> tuple[str, ...]:
    names = getattr(value, "__slots__", ())
    if isinstance(names, str):
        return (names,)
    if not isinstance(names, Sequence) or any(not isinstance(name, str) for name in names):
        raise ValueError("tokenizer slot declaration cannot be fingerprinted")
    return tuple(names)


def derive_serializer_provenance(serializer: Serializer) -> tuple[str, str, str]:
    _require_plain_serializer(serializer)
    serializer_id = _require_identifier(getattr(serializer, "serializer_id", None), "serializer_id")
    serializer_version = _require_identifier(
        getattr(serializer, "serializer_version", None), "serializer_version"
    )
    _require_serializer_independence(serializer)
    fingerprint = canonical_json_bytes(
        {
            "module": getattr(serializer, "__module__", ""),
            "qualified_name": getattr(serializer, "__qualname__", ""),
            "source": _serializer_source(serializer),
            "code": _code_payload(serializer.__code__),
        }
    )
    return serializer_id, serializer_version, sha256_bytes(fingerprint)


def _require_plain_serializer(serializer: Serializer) -> None:
    if not inspect.isfunction(serializer):
        raise TypeError("serializer implementation must be a top-level Python function")
    if serializer.__name__ == "<lambda>" or "<locals>" in serializer.__qualname__:
        raise TypeError("serializer implementation must be a top-level Python function")


def _require_serializer_independence(serializer: Serializer) -> None:
    code = serializer.__code__
    if code.co_freevars:
        raise ValueError("serializer implementation must not depend on closure state")
    if serializer.__defaults__ or serializer.__kwdefaults__:
        raise ValueError("serializer implementation must not depend on mutable defaults")
    _require_no_external_dependencies(serializer, "serializer implementation")


def _require_no_external_dependencies(function: Callable[..., Any], label: str) -> None:
    dependencies = _global_dependencies(function)
    if dependencies:
        raise ValueError(f"{label} must not depend on module globals: {dependencies!r}")
    unsafe_builtins = _unsafe_builtin_dependencies(function)
    if unsafe_builtins:
        raise ValueError(f"{label} uses unsafe builtins: {unsafe_builtins!r}")


def _global_dependencies(function: Callable[..., Any]) -> list[str]:
    return sorted(
        name
        for name in _recursive_code_names(function.__code__)
        if name in function.__globals__ and name != "__builtins__"
    )


def _unsafe_builtin_dependencies(function: Callable[..., Any]) -> list[str]:
    builtin_names = vars(builtins)
    return sorted(
        name
        for name in _recursive_code_names(function.__code__)
        if (name == "__builtins__" or name in builtin_names) and name not in _PURE_BUILTINS
    )


def _recursive_code_names(code: CodeType) -> set[str]:
    names = set(code.co_names)
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(_recursive_code_names(value))
    return names


def _serializer_source(serializer: Serializer) -> str:
    try:
        source = inspect.getsource(serializer)
    except (OSError, TypeError) as exc:
        raise ValueError("serializer implementation source is unavailable") from exc
    return textwrap.dedent(source).replace("\r\n", "\n").strip() + "\n"


def _code_payload(code: CodeType) -> dict[str, Any]:
    return {
        "argcount": code.co_argcount,
        "positional_only_argcount": code.co_posonlyargcount,
        "keyword_only_argcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "bytecode": code.co_code.hex(),
        "exception_table": getattr(code, "co_exceptiontable", b"").hex(),
        "constants": [_fingerprint_value(value) for value in code.co_consts],
        "names": list(code.co_names),
        "variable_names": list(code.co_varnames),
        "free_variables": list(code.co_freevars),
        "cell_variables": list(code.co_cellvars),
    }


def _fingerprint_value(value: Any) -> Any:
    atomic = _fingerprint_atomic(value)
    if atomic is not _UNHANDLED:
        return atomic
    special = _fingerprint_special(value)
    if special is not _UNHANDLED:
        return special
    return _fingerprint_collection(value)


def _fingerprint_atomic(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    return _UNHANDLED


def _fingerprint_special(value: Any) -> Any:
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, slice):
        return {
            "slice": [
                _fingerprint_value(value.start),
                _fingerprint_value(value.stop),
                _fingerprint_value(value.step),
            ]
        }
    if isinstance(value, AddedToken):
        return {"added_token": _fingerprint_value(value.__getstate__())}
    if isinstance(value, CodeType):
        return {"code": _code_payload(value)}
    return _UNHANDLED


def _fingerprint_collection(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return _fingerprint_set(value)
    if isinstance(value, Mapping):
        return _fingerprint_mapping(value)
    raise ValueError(
        "callable fingerprint contains unsupported state: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _fingerprint_set(value: set[Any] | frozenset[Any]) -> dict[str, Any]:
    items = [_fingerprint_value(item) for item in value]
    return {"set": sorted(items, key=canonical_json_bytes)}


def _fingerprint_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    if any(not isinstance(key, str) for key in value):
        raise ValueError("callable fingerprint mappings require string keys")
    return {key: _fingerprint_value(item) for key, item in value.items()}


def _require_callable(value: object, label: str) -> None:
    if not callable(value):
        raise TypeError(f"{label} implementation must be callable")


def _require_expected(expected: str | None, actual: str, label: str) -> None:
    if expected is not None and expected != actual:
        raise ValueError(
            f"{label} assertion does not match callable provenance: "
            f"asserted {expected!r}, derived {actual!r}"
        )


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return value


def _require_immutable_revision(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40,64}", value):
        raise ValueError(f"{label} must be an immutable lowercase revision")
