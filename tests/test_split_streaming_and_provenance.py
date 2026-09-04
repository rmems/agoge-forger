import json
import os
import tracemalloc
import weakref
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

from agoge_forger import split_loaders, split_token_stats, split_validation
from agoge_forger.split_contract import (
    SerializerBinding,
    SplitMaterializationSpec,
    SplitPolicy,
    TokenizerBinding,
    TokenStatisticsDerivation,
    TokenStatisticsSpec,
    TokenStatSplit,
    canonical_json_bytes,
    iter_frozen_records,
    materialize_split,
    sha256_bytes,
    validate_split_manifest,
    write_token_statistics,
)

TOKENIZER_ID = "example/tokenizer"
TOKENIZER_REVISION = "a" * 40
SERIALIZER_ID = "messages-v1"
SERIALIZER_VERSION = "1"
GLOBAL_TOKENIZER_MODE = "bytes"
FAST_TOKENIZER_MODE = "backend"


class _Tokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __call__(self, text: str):
        return {"input_ids": list(text.encode())}


class _FloatingTokenizer:
    name_or_path = TOKENIZER_ID
    revision = "main"

    def __call__(self, text: str):
        return [text]


class _AlternateTokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __call__(self, text: str):
        return [text]


class _MutatingTokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __init__(self):
        self.mode = "initial"

    def __call__(self, text: str):
        self.mode = "mutated"
        return list(text.encode())


class _GlobalStateTokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __call__(self, text: str):
        if GLOBAL_TOKENIZER_MODE == "bytes":
            return list(text.encode())
        return [999]


class _NestedGlobalStateTokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __call__(self, text: str):
        def encode():
            return list(text.encode()) if GLOBAL_TOKENIZER_MODE == "bytes" else [999]

        return encode()


class _DefaultDependentTokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __call__(self, text: str, prefix: int = 1):
        return {"input_ids": [prefix, *text.encode()]}


class _KeywordDefaultDependentTokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __call__(self, text: str, *, suffix: int = 2):
        return {"input_ids": [*text.encode(), suffix]}


class _StaticDefaultDependentTokenizer(_Tokenizer):
    @staticmethod
    def mode(fallback: str = "bytes"):
        return fallback


class _FakeBackend:
    def to_str(self):
        return '{"model":"forged"}'


class _FastShapedTokenizer:
    name_or_path = TOKENIZER_ID
    _commit_hash = TOKENIZER_REVISION

    def __init__(self, mode: str):
        self.mode = mode
        self.init_kwargs = {}
        self.special_tokens_map = {}
        self.chat_template = None
        self.backend_tokenizer = _FakeBackend()

    def get_vocab(self):
        return {"token": 1}

    def __call__(self, text: str):
        return list(text.encode()) if self.mode == "bytes" else [999]


class _GlobalDependentFastTokenizer(PreTrainedTokenizerFast):
    def _encode_plus(self, *args, **kwargs):
        if FAST_TOKENIZER_MODE == "backend":
            return super()._encode_plus(*args, **kwargs)
        return {"input_ids": [999]}


class _ClassStateFastTokenizer(PreTrainedTokenizerFast):
    MODE = "backend"

    def _encode_plus(self, *args, **kwargs):
        if __class__.MODE == "backend":
            return super()._encode_plus(*args, **kwargs)
        return {"input_ids": [999]}


def _serializer(row):
    return str(row["text"])


_serializer.serializer_id = SERIALIZER_ID
_serializer.serializer_version = SERIALIZER_VERSION


def _environment_serializer(row):
    return __import__("os").environ.get("AGOGE_SERIALIZER_PROBE", "") + str(row["text"])


_environment_serializer.serializer_id = "environment-dependent"
_environment_serializer.serializer_version = "1"


def _nested_environment_serializer(row):
    def environment_prefix():
        return __import__("os").environ.get("AGOGE_SERIALIZER_PROBE", "")

    prefix = environment_prefix()
    return prefix + str(row["text"])


_nested_environment_serializer.serializer_id = "nested-environment-dependent"
_nested_environment_serializer.serializer_version = "1"


def _derivation(**spec_updates) -> TokenStatisticsDerivation:
    tokenizer = TokenizerBinding(implementation=_Tokenizer())
    serializer = SerializerBinding(implementation=_serializer)
    spec = TokenStatisticsSpec(
        model_id="example/model",
        model_revision="c" * 40,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_revision=tokenizer.tokenizer_revision,
        tokenizer_sha256=tokenizer.tokenizer_sha256,
        serializer_id=serializer.serializer_id,
        serializer_version=serializer.serializer_version,
        serializer_sha256=serializer.serializer_sha256,
    ).model_copy(update=spec_updates)
    return TokenStatisticsDerivation(
        tokenizer=tokenizer,
        serializer=serializer,
        spec=spec,
    )


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("tokenizer_id", "example/unrelated-tokenizer"),
        ("tokenizer_revision", "d" * 40),
        ("tokenizer_sha256", "f" * 64),
        ("serializer_id", "unrelated-serializer"),
        ("serializer_version", "2"),
        ("serializer_sha256", "e" * 64),
    ],
)
def test_token_statistics_rejects_callable_provenance_mismatch(tmp_path, field, mismatched_value):
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match=field):
        write_token_statistics(
            tmp_path / "missing-manifest.json",
            output,
            _derivation(**{field: mismatched_value}),
        )

    assert not output.exists()


@pytest.mark.parametrize("field", ["tokenizer", "serializer"])
def test_token_statistics_requires_explicit_callable_bindings(field):
    values = {
        "tokenizer": TokenizerBinding(
            implementation=_Tokenizer(),
        ),
        "serializer": SerializerBinding(
            implementation=_serializer,
        ),
        "spec": _derivation().spec,
    }
    values[field] = _Tokenizer() if field == "tokenizer" else _serializer

    with pytest.raises(TypeError, match=f"{field} must be a .*Binding"):
        TokenStatisticsDerivation(**values)


def test_callable_binding_assertions_cannot_lie(tmp_path):
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="tokenizer_id assertion"):
        TokenizerBinding(
            implementation=_Tokenizer(),
            expected_tokenizer_id="example/forged-tokenizer",
        )
    with pytest.raises(ValueError, match="serializer_version assertion"):
        SerializerBinding(
            implementation=_serializer,
            expected_serializer_version="forged-version",
        )

    assert not output.exists()


def test_bindings_reject_unresolvable_callable_provenance():
    with pytest.raises(ValueError, match="immutable"):
        TokenizerBinding(implementation=_FloatingTokenizer())
    with pytest.raises(TypeError, match="top-level Python function"):
        SerializerBinding(implementation=lambda row: str(row["text"]))


def test_python_tokenizer_rejects_module_global_behavior():
    with pytest.raises(ValueError, match="module globals"):
        TokenizerBinding(implementation=_GlobalStateTokenizer())


def test_python_tokenizer_rejects_nested_module_global_behavior():
    with pytest.raises(ValueError, match="module globals"):
        TokenizerBinding(implementation=_NestedGlobalStateTokenizer())


@pytest.mark.parametrize(
    "tokenizer_type",
    [
        _DefaultDependentTokenizer,
        _KeywordDefaultDependentTokenizer,
        _StaticDefaultDependentTokenizer,
    ],
)
def test_python_tokenizer_rejects_default_dependent_behavior(tokenizer_type):
    with pytest.raises(ValueError, match="must not depend on defaults"):
        TokenizerBinding(implementation=tokenizer_type())


def test_fast_shaped_non_hugging_face_tokenizer_cannot_hide_instance_state():
    with pytest.raises(ValueError, match="module globals|unsupported state"):
        TokenizerBinding(implementation=_FastShapedTokenizer("bytes"))


def test_hugging_face_fast_subclass_rejects_module_global_behavior():
    tokenizer = _fast_tokenizer(
        {"[UNK]": 0, "global": 1},
        tokenizer_type=_GlobalDependentFastTokenizer,
    )

    with pytest.raises(ValueError, match="module globals"):
        TokenizerBinding(implementation=tokenizer)


def test_hugging_face_fast_subclass_rejects_closure_behavior():
    mode = "backend"

    class ClosureDependentFastTokenizer(PreTrainedTokenizerFast):
        def _encode_plus(self, *args, **kwargs):
            if mode == "backend":
                return super()._encode_plus(*args, **kwargs)
            return {"input_ids": [999]}

    tokenizer = _fast_tokenizer(
        {"[UNK]": 0, "closure": 1},
        tokenizer_type=ClosureDependentFastTokenizer,
    )

    with pytest.raises(ValueError, match="closure state"):
        TokenizerBinding(implementation=tokenizer)


def test_hugging_face_fast_subclass_binds_class_state():
    tokenizer = _fast_tokenizer(
        {"[UNK]": 0, "class-state": 1},
        tokenizer_type=_ClassStateFastTokenizer,
    )
    first = TokenizerBinding(implementation=tokenizer)

    _ClassStateFastTokenizer.MODE = "alternate"
    try:
        second = TokenizerBinding(implementation=tokenizer)
    finally:
        _ClassStateFastTokenizer.MODE = "backend"

    assert first.tokenizer_sha256 != second.tokenizer_sha256


def test_serializer_rejects_dynamic_import_builtin():
    with pytest.raises(ValueError, match="unsafe builtins"):
        SerializerBinding(implementation=_environment_serializer)


def test_serializer_rejects_nested_dynamic_import_builtin():
    with pytest.raises(ValueError, match="unsafe builtins"):
        SerializerBinding(implementation=_nested_environment_serializer)


def _fast_tokenizer(vocabulary, tokenizer_type=PreTrainedTokenizerFast):
    backend = Tokenizer(WordLevel(vocabulary, "[UNK]"))
    tokenizer = tokenizer_type(tokenizer_object=backend)
    tokenizer.name_or_path = TOKENIZER_ID
    tokenizer._commit_hash = TOKENIZER_REVISION
    return tokenizer


def test_hugging_face_fast_state_and_model_specific_subclasses_are_bound(monkeypatch):
    first_tokenizer = _fast_tokenizer({"[UNK]": 0, "first": 1})
    first = TokenizerBinding(implementation=first_tokenizer)
    second = TokenizerBinding(implementation=_fast_tokenizer({"[UNK]": 0, "second": 1}))

    assert first.tokenizer_sha256 != second.tokenizer_sha256
    first_tokenizer.add_special_tokens({"additional_special_tokens": ["<probe>"]})
    with_added_token = TokenizerBinding(implementation=first_tokenizer)
    assert with_added_token.tokenizer_sha256 != first.tokenizer_sha256

    class HiddenFastTokenizer(PreTrainedTokenizerFast):
        pass

    subclass = TokenizerBinding(
        implementation=_fast_tokenizer(
            {"[UNK]": 0, "hidden": 1}, tokenizer_type=HiddenFastTokenizer
        )
    )

    assert subclass.tokenizer_sha256 != first.tokenizer_sha256

    def changed_encode(self, *args, **kwargs):
        return {"input_ids": [7]}

    monkeypatch.setattr(HiddenFastTokenizer, "_encode_plus", changed_encode)
    changed_subclass = TokenizerBinding(
        implementation=_fast_tokenizer(
            {"[UNK]": 0, "hidden": 1}, tokenizer_type=HiddenFastTokenizer
        )
    )

    assert changed_subclass.tokenizer_sha256 != subclass.tokenizer_sha256


def test_shipped_transformers_fast_subclass_can_be_bound():
    tokenizer = _fast_tokenizer(
        {"[UNK]": 0, "shipped": 1},
        tokenizer_type=GPT2TokenizerFast,
    )

    binding = TokenizerBinding(implementation=tokenizer)

    assert len(binding.tokenizer_sha256) == 64


def test_tokenizer_binding_fingerprints_materially_distinct_callables():
    first = TokenizerBinding(implementation=_Tokenizer())
    second = TokenizerBinding(implementation=_AlternateTokenizer())

    assert first.tokenizer_id == second.tokenizer_id
    assert first.tokenizer_revision == second.tokenizer_revision
    assert first("same text")["input_ids"] != second("same text")
    assert first.tokenizer_sha256 != second.tokenizer_sha256


def test_tokenizer_spec_cannot_be_reused_for_different_same_revision_callable(tmp_path):
    first = _derivation()
    second = TokenizerBinding(implementation=_AlternateTokenizer())
    derivation = TokenStatisticsDerivation(
        tokenizer=second,
        serializer=first.serializer,
        spec=first.spec,
    )
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="tokenizer_sha256"):
        write_token_statistics(tmp_path / "missing-manifest.json", output, derivation)

    assert first.tokenizer.tokenizer_id == second.tokenizer_id
    assert first.tokenizer.tokenizer_revision == second.tokenizer_revision
    assert first.tokenizer.tokenizer_sha256 != second.tokenizer_sha256
    assert not output.exists()


def test_callable_provenance_is_rederived_at_write_time(tmp_path):
    derivation = _derivation()
    derivation.tokenizer.implementation.name_or_path = "example/mutated-tokenizer"
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="tokenizer_id changed after"):
        write_token_statistics(tmp_path / "missing-manifest.json", output, derivation)

    assert not output.exists()


def test_tokenizer_state_is_rederived_after_counting_before_output(tmp_path):
    manifest_path, _ = _materialized_manifest(tmp_path)
    tokenizer = TokenizerBinding(implementation=_MutatingTokenizer())
    serializer = SerializerBinding(implementation=_serializer)
    spec = TokenStatisticsSpec(
        model_id="example/model",
        model_revision="c" * 40,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_revision=tokenizer.tokenizer_revision,
        tokenizer_sha256=tokenizer.tokenizer_sha256,
        serializer_id=serializer.serializer_id,
        serializer_version=serializer.serializer_version,
        serializer_sha256=serializer.serializer_sha256,
    )
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="tokenizer_sha256 changed after"):
        write_token_statistics(
            manifest_path,
            output,
            TokenStatisticsDerivation(
                tokenizer=tokenizer,
                serializer=serializer,
                spec=spec,
            ),
        )

    assert not output.exists()


def test_token_statistics_hashes_the_manifest_snapshot_it_counted(tmp_path, monkeypatch):
    manifest_path, _ = _materialized_manifest(tmp_path)
    original_snapshot = manifest_path.read_bytes()
    replacement_payload = json.loads(original_snapshot)
    replacement_payload["source"]["dataset_version"] = "replaced-v2"
    replacement_path = tmp_path / "replacement-manifest.json"
    replacement_path.write_bytes(canonical_json_bytes(replacement_payload) + b"\n")
    original_for_split = split_token_stats._TokenCounter.for_split
    replaced = False

    def count_then_replace(self, path, manifest, split):
        nonlocal replaced
        result = original_for_split(self, path, manifest, split)
        if not replaced:
            os.replace(replacement_path, manifest_path)
            replaced = True
        return result

    monkeypatch.setattr(split_token_stats._TokenCounter, "for_split", count_then_replace)

    statistics = write_token_statistics(
        manifest_path,
        tmp_path / "token-statistics.json",
        _derivation(),
    )

    assert replaced
    assert statistics.split_manifest_sha256 == sha256_bytes(original_snapshot)
    assert statistics.split_manifest_sha256 != sha256_bytes(manifest_path.read_bytes())


def test_token_statistics_cleans_partial_staging_after_write_failure(tmp_path, monkeypatch):
    manifest_path, _ = _materialized_manifest(tmp_path)
    output = tmp_path / "token-statistics.json"

    def fail_after_partial_write(path, payload):
        path.write_bytes(payload[:8])
        raise OSError("synthetic token-statistics write failure")

    monkeypatch.setattr(
        split_token_stats,
        "_write_token_statistics_payload",
        fail_after_partial_write,
        raising=False,
    )

    with pytest.raises(OSError, match="synthetic token-statistics write failure"):
        write_token_statistics(manifest_path, output, _derivation())

    assert not output.exists()
    assert not list(tmp_path.glob(".token-statistics.json.*"))


def test_token_counter_aggregates_large_splits_without_retaining_lengths(monkeypatch):
    record_count = 100_000

    def records(*_args):
        for index in range(record_count):
            yield {"text": str(index + 300)}

    monkeypatch.setattr(split_token_stats, "iter_materialized_records", records)
    counter = split_token_stats._TokenCounter(
        tokenizer=lambda text: range(int(text)),
        serializer=lambda row: row["text"],
        context_limit=50_000,
    )

    tracemalloc.start()
    try:
        statistics = counter.for_split(Path("manifest.json"), object(), "train")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert statistics.record_count == record_count
    assert statistics.minimum_tokens == 300
    assert statistics.maximum_tokens == record_count + 299
    assert peak_bytes < 1_000_000


@pytest.mark.parametrize(
    "total_tokens",
    [19, 20, 40, 41],
)
def test_token_stat_split_rejects_totals_outside_declared_bounds(total_tokens):
    with pytest.raises(ValueError, match="total_tokens"):
        TokenStatSplit(
            record_count=2,
            total_tokens=total_tokens,
            minimum_tokens=10,
            maximum_tokens=20,
            truncated_records=0,
        )


def test_serializer_provenance_is_rederived_at_write_time(tmp_path, monkeypatch):
    derivation = _derivation()
    monkeypatch.setattr(_serializer, "serializer_version", "mutated-version")
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="serializer_version changed after"):
        write_token_statistics(tmp_path / "missing-manifest.json", output, derivation)

    assert not output.exists()


def test_serializer_fingerprint_covers_executable_code(tmp_path, monkeypatch):
    derivation = _derivation()
    mutated_constants = tuple(
        "canonical_id" if value == "text" else value for value in _serializer.__code__.co_consts
    )
    monkeypatch.setattr(
        _serializer,
        "__code__",
        _serializer.__code__.replace(co_consts=mutated_constants),
    )
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="serializer_sha256 changed after"):
        write_token_statistics(tmp_path / "missing-manifest.json", output, derivation)

    assert not output.exists()


def _materialized_manifest(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "canonical_id": f"sample-{index:03d}",
            "lineage_id": f"lineage-{index:03d}",
            "group_id": f"group-{index:03d}",
            "text": f"Deterministic example {index}",
        }
        for index in range(90)
    ]
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    output = tmp_path / "snapshot"
    manifest = materialize_split(
        source,
        output,
        SplitMaterializationSpec(
            source_repository="example/dataset",
            source_revision="f" * 40,
            dataset_version="v1",
            source_path="data/source.jsonl",
            split_policy=SplitPolicy(
                seed=99,
                salt="streaming-regression-v1",
                weights={"train": 6, "validation": 2, "held_out": 2},
            ),
        ),
    )
    return output / "split_manifest.json", manifest


def test_frozen_loader_does_not_retain_materialized_source_records(tmp_path, monkeypatch):
    manifest_path, manifest = _materialized_manifest(tmp_path)
    original = split_validation.iter_source_records
    record_references: list[weakref.ReferenceType] = []
    peak_live_records = 0

    def tracked_records(*args, **kwargs):
        nonlocal peak_live_records
        for record in original(*args, **kwargs):
            record_references.append(weakref.ref(record))
            live_records = sum(reference() is not None for reference in record_references)
            peak_live_records = max(peak_live_records, live_records)
            yield record

    monkeypatch.setattr(split_validation, "iter_source_records", tracked_records)

    loaded = list(iter_frozen_records(manifest_path, "train"))

    assert len(loaded) == manifest.splits["train"].record_count
    assert peak_live_records <= 2


def test_manifest_validation_rejects_downstream_rendered_representation(tmp_path):
    manifest_path, manifest = _materialized_manifest(tmp_path)
    artifact = manifest.splits["train"]
    artifact_path = manifest_path.parent / artifact.path
    rows = [json.loads(line) for line in artifact_path.read_text().splitlines()]
    converted = [
        {key: value for key, value in row.items() if key != "text"}
        | {"messages": [{"role": "user", "content": row["text"]}]}
        for row in rows
    ]
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in converted)
    artifact_path.write_bytes(payload)
    updated_artifact = artifact.model_copy(update={"sha256": sha256_bytes(payload)})
    updated_manifest = manifest.model_copy(
        update={"splits": {**manifest.splits, "train": updated_artifact}}
    )
    manifest_path.write_bytes(
        canonical_json_bytes(updated_manifest.model_dump(mode="json")) + b"\n"
    )

    with pytest.raises(ValueError, match="new split snapshots require pre-rendered 'text'"):
        validate_split_manifest(manifest_path)


def test_declared_split_symlink_is_rejected(tmp_path):
    manifest_path, manifest = _materialized_manifest(tmp_path)
    artifact_path = manifest_path.parent / manifest.splits["train"].path
    target_path = artifact_path.with_name("train-real.jsonl")
    artifact_path.rename(target_path)
    artifact_path.symlink_to(target_path.name)

    with pytest.raises(ValueError, match="without following a symlink"):
        validate_split_manifest(manifest_path)


def test_split_validation_reports_unsupported_dirfd_as_controlled_error(tmp_path, monkeypatch):
    manifest_path, _ = _materialized_manifest(tmp_path)

    def unsupported_descriptor(root, relative_path):
        raise NotImplementedError("dir_fd unavailable")

    monkeypatch.setattr(split_validation, "_open_split_descriptor", unsupported_descriptor)

    with pytest.raises(ValueError, match=r"cannot be validated safely.*dir_fd") as raised:
        validate_split_manifest(manifest_path)

    assert isinstance(raised.value.__cause__, NotImplementedError)


def test_split_validation_stages_snapshot_beside_manifest(tmp_path, monkeypatch):
    manifest_path, _ = _materialized_manifest(tmp_path)
    original_mkstemp = split_validation.tempfile.mkstemp
    observed_directories = []

    def beside_manifest(*args, **kwargs):
        observed_directories.append(kwargs.get("dir"))
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(split_validation.tempfile, "mkstemp", beside_manifest)

    validate_split_manifest(manifest_path)

    assert observed_directories
    assert set(observed_directories) == {manifest_path.parent}


def test_split_validation_accepts_configured_writable_staging(tmp_path, monkeypatch):
    manifest_path, _ = _materialized_manifest(tmp_path)
    staging = tmp_path / "writable-staging"
    staging.mkdir()
    original_mkstemp = split_validation.tempfile.mkstemp
    original_temporary_directory = split_validation.tempfile.TemporaryDirectory
    observed_file_directories = []
    observed_source_directories = []

    def configured_staging(*args, **kwargs):
        observed_file_directories.append(kwargs.get("dir"))
        return original_mkstemp(*args, **kwargs)

    def configured_source_staging(*args, **kwargs):
        observed_source_directories.append(kwargs.get("dir"))
        return original_temporary_directory(*args, **kwargs)

    monkeypatch.setenv("AGOGE_VALIDATION_STAGING_DIR", str(staging))
    monkeypatch.setattr(split_validation.tempfile, "mkstemp", configured_staging)
    monkeypatch.setattr(
        split_validation.tempfile,
        "TemporaryDirectory",
        configured_source_staging,
    )

    validate_split_manifest(manifest_path, source_path=tmp_path / "source.jsonl")

    assert set(observed_file_directories) == {staging}
    assert set(observed_source_directories) == {staging}
    assert list(staging.iterdir()) == []


def test_split_validation_reports_unwritable_staging(tmp_path, monkeypatch):
    manifest_path, _ = _materialized_manifest(tmp_path)

    def deny_staging(*args, **kwargs):
        raise PermissionError("read-only mount")

    monkeypatch.setattr(split_validation.tempfile, "mkstemp", deny_staging)

    with pytest.raises(ValueError, match="set AGOGE_VALIDATION_STAGING_DIR") as raised:
        validate_split_manifest(manifest_path)

    assert isinstance(raised.value.__cause__, PermissionError)


@pytest.mark.parametrize("duplicate", ["schema", "nested-split"])
def test_split_manifest_rejects_duplicate_json_keys(tmp_path, duplicate):
    manifest_path, _ = _materialized_manifest(tmp_path)
    payload = manifest_path.read_bytes()
    if duplicate == "schema":
        payload = payload.replace(
            b"{",
            b'{"schema_version":"agoge.split-manifest.v1",',
            1,
        )
    else:
        payload = payload.replace(b'"held_out":{', b'"held_out":{},"held_out":{', 1)
    manifest_path.write_bytes(payload)

    with pytest.raises(ValueError, match="invalid split manifest JSON"):
        validate_split_manifest(manifest_path)


def test_frozen_loader_rechecks_selected_artifact_after_manifest_validation(tmp_path, monkeypatch):
    manifest_path, manifest = _materialized_manifest(tmp_path)
    artifact_path = manifest_path.parent / manifest.splits["train"].path
    original = split_loaders.validate_split_manifest

    def validate_then_replace(path):
        validated = original(path)
        artifact_path.write_bytes(b'{"canonical_id":"unvalidated"}\n')
        return validated

    monkeypatch.setattr(split_loaders, "validate_split_manifest", validate_then_replace)

    with pytest.raises(ValueError, match="train digest mismatch"):
        list(iter_frozen_records(manifest_path, "train"))


def test_artifact_replacement_during_snapshot_validation_fails_closed(tmp_path, monkeypatch):
    manifest_path, manifest = _materialized_manifest(tmp_path)
    artifact_path = manifest_path.parent / manifest.splits["train"].path
    replacement = tmp_path / "identical-replacement.jsonl"
    replacement.write_bytes(artifact_path.read_bytes())
    original = split_validation.iter_source_records
    replaced = False

    def replacing_records(*args, **kwargs):
        nonlocal replaced
        for record in original(*args, **kwargs):
            if not replaced and Path(args[0]).name.startswith("agoge-train-snapshot-"):
                os.replace(replacement, artifact_path)
                replaced = True
            yield record

    monkeypatch.setattr(split_validation, "iter_source_records", replacing_records)

    with pytest.raises(ValueError, match="train artifact changed while"):
        validate_split_manifest(manifest_path)

    assert replaced
