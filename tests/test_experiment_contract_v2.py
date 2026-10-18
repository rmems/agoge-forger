"""Tests for ``agoge.experiment-contract.v2`` and version dispatch (#188).

The selfgen-repair-v1 draft under ``tests/fixtures/experiment-contract-v2/`` is
the design from #187 with every not-yet-known pin left as ``<fill>``. These tests
resolve the placeholders in memory; the fixture itself keeps them.
"""

from __future__ import annotations

import json
import types
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from agoge_forger.eval.experiment_contract import (
    ExperimentContract,
    ExperimentContractError,
    ExperimentContractV2,
    SplitPolicyPin,
    build_experiment_contract,
    parse_experiment_contract,
    split_pin_from_manifest,
    validate_experiment_contract,
)
from tests.evaluation_contract_cases import frozen_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "experiment-contract-v2"
    / "selfgen-v1-experiment-contract.json"
)
V1_CONTRACTS = (
    "reports/granite-4.1-first-sft/rm760-code-repair-v1-rm760-20260914/experiment-contract.json",
    "reports/granite-4.1-r2egym-sft/r2egym-cap512-last-assistant-v1/experiment-contract.json",
)
PLACEHOLDER = "<fill>"
HEX_A = "a" * 64
REVISION = "0123456789abcdef0123456789abcdef01234567"

FieldPath = tuple[str | int, ...]

# Every <fill> in the draft, with a type-correct stand-in value.
RESOLUTIONS: dict[FieldPath, object] = {
    ("agoge_git_sha",): REVISION,
    ("model", "parent_adapter", "experiment_id"): "rm760-code-repair-v1-rm760-20260914",
    ("model", "parent_adapter", "adapter_sha256"): HEX_A,
    ("dataset", "factory_revision"): REVISION,
    ("dataset", "export_manifest_sha256"): HEX_A,
    ("dataset", "agoge_jsonl_sha256"): HEX_A,
    ("dataset", "self_generated_rows", "provenance_required", 0): "proposer_adapter_sha256",
    ("split", "split_manifest_path"): "pinned-split/split_manifest.json",
    ("split", "split_manifest_sha256"): HEX_A,
    ("split", "train_split_sha256"): HEX_A,
    ("split", "validation_split_sha256"): HEX_A,
    ("split", "held_out_split_sha256"): HEX_A,
    ("split", "train_record_count"): 129,
    ("split", "validation_record_count"): 10,
    ("split", "held_out_record_count"): 16,
    ("split", "split_seed"): 20260908,
    ("split", "split_salt"): "python-repair-v2",
    ("split", "train_weight"): 80,
    ("split", "validation_weight"): 10,
    ("split", "held_out_weight"): 10,
    ("serializer", "serializer_id"): "code-repair-prompt-v1",
    ("serializer", "serializer_version"): "1",
    ("serializer", "serializer_sha256"): HEX_A,
    ("training", "batch_size"): 1,
    ("training", "gradient_accumulation_steps"): 8,
    ("training", "seed"): 42,
    ("evaluation", "context_window"): 4096,
    ("evaluation", "max_new_tokens"): 576,
    ("evaluation", "decoding", "seed"): 17,
    ("evaluation", "decoding", "max_new_tokens"): 576,
    ("self_generation", "seed_catalog", "catalog_sha256"): HEX_A,
    ("self_generation", "seed_catalog", "held_out_denylist_sha256"): HEX_A,
    ("self_generation", "propose", "decoding", "seed"): 7,
    ("self_generation", "propose", "decoding", "max_new_tokens"): 576,
    ("self_generation", "propose", "decoding", "top_p"): 0.95,
    ("self_generation", "verify_task", "verifier_id"): "doctest-task-gate-v1",
    ("self_generation", "solve", "output_format"): "search-replace-diff",
}

# The fields #188 adds; each must be required by the v2 schema.
ISSUE_FIELDS: tuple[FieldPath, ...] = (
    ("hypothesis",),
    ("evaluation", "arms"),
    ("evaluation", "runs"),
    ("evaluation", "interval"),
    ("evaluation", "paired_test"),
    ("self_generation",),
    *(
        ("self_generation", name)
        for name in (
            "seed_catalog",
            "propose",
            "verify_task",
            "solve",
            "verify_solution",
            "difficulty_band",
            "iterations",
            "stop_rules",
        )
    ),
    ("dataset", "self_generated_rows"),
    ("dataset", "self_generated_rows", "max_fraction_of_train"),
    ("dataset", "self_generated_rows", "min_survival_rate_to_continue"),
    ("dataset", "self_generated_rows", "provenance_required"),
    ("model", "parent_adapter"),
    ("model", "parent_adapter", "experiment_id"),
    ("model", "parent_adapter", "adapter_sha256"),
    ("training", "init"),
)


def _load_draft() -> dict[str, Any]:
    return json.loads(DRAFT_PATH.read_text(encoding="utf-8"))


def _placeholder_paths(value: object, prefix: FieldPath = ()) -> Iterator[FieldPath]:
    if value == PLACEHOLDER:
        yield prefix
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _placeholder_paths(child, (*prefix, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _placeholder_paths(child, (*prefix, index))


def _parent_of(payload: Any, path: FieldPath) -> Any:
    for part in path[:-1]:
        payload = payload[part]
    return payload


def _resolved_draft() -> dict[str, Any]:
    draft = _load_draft()
    for path, value in RESOLUTIONS.items():
        _parent_of(draft, path)[path[-1]] = value
    return draft


def _nested_model(annotation: object) -> type[BaseModel] | None:
    """Return the model type behind ``Model``, ``Model | None``, or ``tuple[Model, ...]``."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = typing.get_origin(annotation)
    if origin in (types.UnionType, typing.Union, tuple):
        for argument in typing.get_args(annotation):
            nested = _nested_model(argument)
            if nested is not None:
                return nested
    return None


def _required_paths(model: type[BaseModel], payload: dict[str, Any], prefix: FieldPath = ()):
    for name, info in model.model_fields.items():
        path = (*prefix, name)
        if info.is_required():
            yield path
        nested = _nested_model(info.annotation)
        child = payload.get(name)
        if nested is None:
            continue
        if isinstance(child, dict):
            yield from _required_paths(nested, child, path)
        elif isinstance(child, list) and child and isinstance(child[0], dict):
            yield from _required_paths(nested, child[0], (*path, 0))


def _dotted(path: FieldPath) -> str:
    text = ""
    for part in path:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else part)
    return text


REQUIRED_PATHS = tuple(_required_paths(ExperimentContractV2, _resolved_draft()))


def _without_self_generation() -> dict[str, Any]:
    contract = _resolved_draft()
    contract["self_generation"] = None
    contract["dataset"]["self_generated_rows"] = None
    contract["model"]["parent_adapter"] = None
    contract["evaluation"]["arms"] = [
        {"arm_id": "g0", "role": "base", "training_rows": "none"},
        {"arm_id": "g1", "role": "treatment", "training_rows": "base"},
    ]
    contract["evaluation"]["paired_test"]["comparisons"] = [{"treatment": "g1", "control": "g0"}]
    return contract


def _rejects(contract: dict[str, Any], message: str) -> None:
    with pytest.raises(ExperimentContractError, match=message):
        parse_experiment_contract(contract)


def test_draft_placeholders_match_the_resolution_table():
    assert set(_placeholder_paths(_load_draft())) == set(RESOLUTIONS)


def test_selfgen_draft_validates_once_placeholders_are_resolved():
    contract = parse_experiment_contract(_resolved_draft())
    assert isinstance(contract, ExperimentContractV2)
    assert contract.training.init == "fresh"
    assert contract.self_generation is not None
    assert contract.self_generation.propose.decoding.do_sample is True
    assert [arm.arm_id for arm in contract.evaluation.arms] == ["g1", "g2-accumulate", "g2-control"]


def test_selfgen_draft_with_placeholders_is_rejected():
    with pytest.raises(ExperimentContractError) as caught:
        parse_experiment_contract(_load_draft())
    message = str(caught.value)
    assert "agoge_git_sha:" in message
    assert "model.parent_adapter.adapter_sha256:" in message
    assert "self_generation.propose.decoding.top_p:" in message


def test_every_issue_field_is_required():
    missing = [path for path in ISSUE_FIELDS if path not in REQUIRED_PATHS]
    assert missing == []


@pytest.mark.parametrize("path", REQUIRED_PATHS, ids=_dotted)
def test_missing_required_field_is_named(path):
    contract = _resolved_draft()
    del _parent_of(contract, path)[path[-1]]
    expected = f"{_dotted(path)}: missing required field"
    with pytest.raises(ExperimentContractError) as caught:
        parse_experiment_contract(contract)
    assert expected in str(caught.value)


def test_missing_schema_version_is_named():
    contract = _resolved_draft()
    del contract["schema_version"]
    _rejects(contract, r"schema_version: missing required field")


def test_unknown_schema_version_is_rejected():
    contract = _resolved_draft()
    contract["schema_version"] = "agoge.experiment-contract.v3"
    _rejects(contract, r"schema_version: unsupported experiment contract version")


def test_v2_contract_without_self_generation_validates():
    contract = parse_experiment_contract(_without_self_generation())
    assert isinstance(contract, ExperimentContractV2)
    assert contract.self_generation is None


def test_stacked_init_requires_parent_adapter():
    contract = _without_self_generation()
    contract["training"]["init"] = "stacked"
    _rejects(contract, r"training.init='stacked' requires model.parent_adapter")


def test_self_generation_requires_row_policy():
    contract = _resolved_draft()
    contract["dataset"]["self_generated_rows"] = None
    _rejects(contract, r"self_generation and dataset.self_generated_rows must both be set")


def test_self_generated_arm_requires_self_generation():
    contract = _without_self_generation()
    contract["evaluation"]["arms"][1]["training_rows"] = "base_plus_self_generated"
    _rejects(contract, r"base_plus_self_generated rows but self_generation is null")


def test_paired_comparison_must_name_declared_arms():
    contract = _resolved_draft()
    contract["evaluation"]["paired_test"]["comparisons"][0]["control"] = "g0"
    _rejects(contract, r"comparisons\[0\].control names undeclared arm 'g0'")


def test_paired_comparison_needs_two_arms():
    contract = _resolved_draft()
    contract["evaluation"]["paired_test"]["comparisons"][0]["control"] = "g2-accumulate"
    _rejects(contract, r"evaluation.paired_test.comparisons\[0\]: .*two different arms")


def test_arm_ids_must_be_unique():
    contract = _resolved_draft()
    contract["evaluation"]["arms"][2]["arm_id"] = "g1"
    _rejects(contract, r"evaluation.arms\[\].arm_id contains duplicate entry 'g1'")


def test_proposer_must_be_a_declared_arm():
    contract = _resolved_draft()
    contract["self_generation"]["propose"]["proposer_arm"] = "g9"
    _rejects(contract, r"self_generation.propose.proposer_arm names undeclared arm 'g9'")


def test_difficulty_band_must_be_ordered():
    contract = _resolved_draft()
    contract["self_generation"]["difficulty_band"]["min_pass_rate"] = 0.9
    _rejects(contract, r"min_pass_rate must be below max_pass_rate")


def test_evaluation_decoding_stays_greedy():
    contract = _resolved_draft()
    contract["evaluation"]["decoding"].update({"do_sample": True, "temperature": 0.7})
    _rejects(contract, r"experiment contract requires greedy decoding")


def test_unknown_v2_field_is_rejected():
    contract = _resolved_draft()
    contract["self_generation"]["temperature"] = 0.7
    _rejects(contract, r"self_generation.temperature: Extra inputs are not permitted")


@pytest.mark.parametrize("relative_path", V1_CONTRACTS)
def test_committed_v1_contracts_still_validate(relative_path):
    contract = validate_experiment_contract(REPO_ROOT / relative_path)
    assert isinstance(contract, ExperimentContract)
    assert contract.schema_version == "agoge.experiment-contract.v1"


def test_v1_contract_rejects_v2_fields():
    payload = json.loads((REPO_ROOT / V1_CONTRACTS[0]).read_text(encoding="utf-8"))
    payload["hypothesis"] = "v2-only field"
    _rejects(payload, r"hypothesis: Extra inputs are not permitted")


def _frozen_v2_contract(tmp_path: Path) -> ExperimentContractV2:
    manifest_path, _manifest = frozen_manifest(tmp_path)
    anchor = tmp_path / "experiment"
    anchor.mkdir()
    split = split_pin_from_manifest(
        manifest_path,
        contract_anchor=anchor,
        policy=SplitPolicyPin(
            split_seed=99,
            split_salt="paired-evaluation-v1",
            train_weight=6,
            validation_weight=2,
            held_out_weight=2,
        ),
    )
    payload = _resolved_draft()
    payload["split"] = split.model_dump(mode="json")
    contract = parse_experiment_contract(payload)
    assert isinstance(contract, ExperimentContractV2)
    return contract


def test_freeze_writes_and_validates_v2_contract(tmp_path):
    destination = tmp_path / "experiment" / "experiment-contract.json"
    build_experiment_contract(contract_path=destination, contract=_frozen_v2_contract(tmp_path))
    validated = validate_experiment_contract(destination)
    assert isinstance(validated, ExperimentContractV2)
    assert validated.experiment_id == "selfgen-repair-v1"


def test_freeze_revalidates_v2_contract_before_writing(tmp_path):
    destination = tmp_path / "experiment" / "experiment-contract.json"
    contract = _frozen_v2_contract(tmp_path)
    tampered = contract.model_copy(update={"hypothesis": "", "self_generation": None})
    with pytest.raises(ExperimentContractError, match="hypothesis:"):
        build_experiment_contract(contract_path=destination, contract=tampered)
    assert not destination.exists()
