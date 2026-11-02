"""Bytecode-level purity checks for serializer and tokenizer provenance bindings.

This module deliberately has no top-level imports of ``os`` or ``platform`` so that
function-local imports of those modules cannot be caught by the module-global check.
"""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from agoge_forger.eval.serializers import code_repair_prompt
from agoge_forger.split_contract import SerializerBinding, TokenizerBinding

TOKENIZER_ID = "example/tokenizer"
TOKENIZER_REVISION = "a" * 40


def text(row):
    """Module helper whose name collides with the ``row.text`` attribute below."""

    return str(row["text"])


def encode(value):
    """Module helper whose name collides with the ``.encode`` method below."""

    return list(value)


def _tag(function, serializer_id):
    function.serializer_id = serializer_id
    function.serializer_version = "1"
    return function


def _import_statement_serializer(row):
    import os

    return os.environ.get("AGOGE_SERIALIZER_PROBE", "") + str(row["text"])


def _from_import_serializer(row):
    from os import environ

    return environ.get("AGOGE_SERIALIZER_PROBE", "") + str(row["text"])


def _nested_import_serializer(row):
    def prefix():
        import platform

        return platform.node()

    return prefix() + str(row["text"])


def _format_traversal_serializer(row):
    # Format fields walk attributes inside the string constant, out of the bytecode
    # scan's sight: len.__self__ is the builtins module.
    return f"{len.__self__.__name__}" + str(row["text"])


def _str_format_traversal_serializer(row):
    return str.format("{0.__self__.__name__}", len) + str(row["text"])


def _real_global_serializer(row):
    return text(row)


def _builtins_serializer(row):
    return str(__builtins__) + str(row["text"])


def _builtins_module_import_serializer(row):
    return len.__self__.__import__("os").environ.get("AGOGE_SERIALIZER_PROBE", "") + str(
        row["text"]
    )


def _builtins_module_attribute_serializer(row):
    return str(len.__self__.open) + str(row["text"])


def _function_builtins_attribute_serializer(row):
    probe = (lambda: 0).__builtins__["__import__"]("os").environ.get("AGOGE_SERIALIZER_PROBE", "")
    return probe + str(row["text"])


def _function_globals_attribute_serializer(row):
    return str((lambda: 0).__globals__.get("TOKENIZER_ID", "")) + str(row["text"])


def _frame_globals_serializer(row):
    frame_globals = (value for value in [0]).gi_frame.f_globals
    return str(frame_globals.get("TOKENIZER_ID", "")) + str(row["text"])


def _global_write_serializer(row):
    global _SERIALIZER_CALLS
    _SERIALIZER_CALLS = 1
    return str(row["text"])


def _unresolved_global_serializer(row):
    return str(_LATE_BOUND_PREFIX) + str(row["text"])  # noqa: F821


for _function, _identifier in (
    (_import_statement_serializer, "import-statement"),
    (_from_import_serializer, "from-import"),
    (_nested_import_serializer, "nested-import"),
    (_format_traversal_serializer, "format-traversal"),
    (_str_format_traversal_serializer, "str-format-traversal"),
    (_real_global_serializer, "real-global"),
    (_builtins_serializer, "builtins-access"),
    (_builtins_module_import_serializer, "builtins-module-import"),
    (_builtins_module_attribute_serializer, "builtins-module-attribute"),
    (_function_builtins_attribute_serializer, "function-builtins-attribute"),
    (_function_globals_attribute_serializer, "function-globals-attribute"),
    (_frame_globals_serializer, "frame-globals"),
    (_global_write_serializer, "global-write"),
    (_unresolved_global_serializer, "unresolved-global"),
):
    _tag(_function, _identifier)


class _ImportingTokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __call__(self, value: str):
        import os

        return [ord(char) for char in os.environ.get("AGOGE_TOKENIZER_PROBE", "") + value]


class _ImportingFastTokenizer(PreTrainedTokenizerFast):
    def _encode_plus(self, *args, **kwargs):
        from os import environ

        if environ.get("AGOGE_TOKENIZER_PROBE"):
            return {"input_ids": [999]}
        return super()._encode_plus(*args, **kwargs)


class _BuiltinsModuleTokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __call__(self, value: str):
        environ = len.__self__.__import__("os").environ
        return [ord(char) for char in environ.get("AGOGE_TOKENIZER_PROBE", "") + value]


class _SuperOnlyFastTokenizer(PreTrainedTokenizerFast):
    def _encode_plus(self, *args, **kwargs):
        return super()._encode_plus(*args, **kwargs)


class _SuperInitFastTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _fast_tokenizer(tokenizer_type):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "token": 1}, "[UNK]"))
    tokenizer = tokenizer_type(tokenizer_object=backend)
    tokenizer.name_or_path = TOKENIZER_ID
    tokenizer._commit_hash = TOKENIZER_REVISION
    return tokenizer


@pytest.mark.parametrize(
    "serializer",
    [_import_statement_serializer, _from_import_serializer, _nested_import_serializer],
)
def test_serializer_rejects_import_statements(serializer):
    with pytest.raises(ValueError, match="must not import modules"):
        SerializerBinding(implementation=serializer)


def test_python_tokenizer_rejects_import_statement():
    with pytest.raises(ValueError, match="must not import modules"):
        TokenizerBinding(implementation=_ImportingTokenizer())


def test_fast_tokenizer_subclass_rejects_import_statement():
    with pytest.raises(ValueError, match="must not import modules"):
        TokenizerBinding(implementation=_fast_tokenizer(_ImportingFastTokenizer))


@pytest.mark.parametrize(
    "serializer", [_format_traversal_serializer, _str_format_traversal_serializer]
)
def test_serializer_rejects_format_field_traversal(serializer):
    """``format`` stays a screened name: its fields can walk dunders out of sight."""
    with pytest.raises(ValueError, match="unsafe builtins|module globals"):
        SerializerBinding(implementation=serializer)


def test_real_global_and_builtins_lookups_are_still_rejected():
    with pytest.raises(ValueError, match="module globals"):
        SerializerBinding(implementation=_real_global_serializer)
    with pytest.raises(ValueError, match="unsafe builtins"):
        SerializerBinding(implementation=_builtins_serializer)


@pytest.mark.parametrize(
    ("serializer", "unsafe_name"),
    [
        (_builtins_module_import_serializer, "__self__"),
        (_builtins_module_attribute_serializer, "__self__"),
        (_function_builtins_attribute_serializer, "__builtins__"),
        (_function_globals_attribute_serializer, "__globals__"),
        (_frame_globals_serializer, "gi_frame"),
    ],
)
def test_serializer_rejects_attribute_escape_hatches(serializer, unsafe_name):
    with pytest.raises(ValueError, match=f"unsafe builtins: .*'{unsafe_name}'"):
        SerializerBinding(implementation=serializer)


def test_python_tokenizer_rejects_attribute_escape_hatch():
    with pytest.raises(ValueError, match="unsafe builtins: .*'__self__'"):
        TokenizerBinding(implementation=_BuiltinsModuleTokenizer())


def test_serializer_rejects_module_global_assignment():
    with pytest.raises(
        ValueError, match="must not assign module globals: \\['_SERIALIZER_CALLS'\\]"
    ):
        SerializerBinding(implementation=_global_write_serializer)


def test_serializer_rejects_unresolved_global_lookup():
    with pytest.raises(ValueError, match="module globals: \\['_LATE_BOUND_PREFIX'\\]"):
        SerializerBinding(implementation=_unresolved_global_serializer)


@pytest.mark.parametrize("tokenizer_type", [_SuperOnlyFastTokenizer, _SuperInitFastTokenizer])
def test_super_stays_allowed_only_for_fast_tokenizer_subclasses(tokenizer_type):
    binding = TokenizerBinding(implementation=_fast_tokenizer(tokenizer_type))

    assert binding.tokenizer_id == TOKENIZER_ID


def test_shipped_code_repair_prompt_still_binds():
    binding = SerializerBinding(implementation=code_repair_prompt)

    assert binding({"text": "abcdef", "completion_start_char": 3}) == "abc"
