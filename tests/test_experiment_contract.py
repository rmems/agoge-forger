"""Tests for pre-registered experiment contracts and G0-only held-out eval."""

from pathlib import Path

import pytest

from agoge_forger.eval.experiment_contract import (
    ExperimentContract,
    build_experiment_contract,
    load_experiment_contract,
    split_pin_from_manifest,
    validate_experiment_contract,
)
from agoge_forger.eval.harness import HeldOutEvalRuntime, run_g0_base_eval
from agoge_forger.eval.score import OBJECTIVE_SCORING_VERSION
from agoge_forger.eval.serializers import (
    code_repair_prompt,
)
from agoge_forger.split_contract import SerializerBinding
from tests.evaluation_contract_cases import MODEL_REPOSITORY, MODEL_REVISION, frozen_manifest
from tests.held_out_eval_cases import canary_evaluation_case, scripted_generator


def _serializer_pin() -> dict[str, str]:
    binding = SerializerBinding(implementation=code_repair_prompt)
    return {
        "serializer_id": binding.serializer_id,
        "serializer_version": binding.serializer_version,
        "serializer_sha256": binding.serializer_sha256,
    }


def _experiment_contract(tmp_path: Path, manifest_path: Path) -> ExperimentContract:
    anchor = tmp_path / "experiment"
    anchor.mkdir()
    split = split_pin_from_manifest(
        manifest_path,
        contract_anchor=anchor,
        split_seed=99,
        split_salt="paired-evaluation-v1",
        train_weight=6,
        validation_weight=2,
        held_out_weight=2,
    )
    return ExperimentContract(
        experiment_id="test-experiment",
        agoge_git_sha=MODEL_REVISION,
        model={
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "trust_remote_code": False,
        },
        dataset={
            "factory_repository": "rmems/synthetic-factory",
            "factory_revision": MODEL_REVISION,
            "export_manifest_sha256": "a" * 64,
            "agoge_jsonl_sha256": "b" * 64,
        },
        split=split,
        serializer=_serializer_pin(),
        training={
            "completion_only_loss": True,
            "max_seq_length": 2048,
            "target_modules": ("q_proj",),
            "learning_rate": 1e-4,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 1,
            "seed": 42,
        },
        evaluation={
            "context_window": 512,
            "max_new_tokens": 128,
            "decoding": {
                "do_sample": False,
                "seed": 17,
                "max_new_tokens": 128,
                "temperature": 0,
                "top_p": 1,
            },
            "scoring_version": OBJECTIVE_SCORING_VERSION,
            "truncation_policy": "mark_unsupported",
        },
    )


def test_experiment_contract_rejects_host_absolute_split_path(tmp_path):
    manifest_path, _manifest = frozen_manifest(tmp_path)
    contract = _experiment_contract(tmp_path, manifest_path)
    broken = contract.model_copy(
        update={
            "split": contract.split.model_copy(
                update={"split_manifest_path": "/tmp/split_manifest.json"}
            )
        }
    )
    with pytest.raises(ValueError, match="portable relative paths"):
        ExperimentContract.model_validate(broken.model_dump(mode="json"))


def test_build_and_validate_experiment_contract(tmp_path):
    manifest_path, _manifest = frozen_manifest(tmp_path)
    destination = tmp_path / "experiment" / "experiment-contract.json"
    contract = _experiment_contract(tmp_path, manifest_path)
    build_experiment_contract(contract_path=destination, contract=contract)
    loaded = load_experiment_contract(destination)
    assert loaded.experiment_id == "test-experiment"
    validate_experiment_contract(destination)


def test_g0_only_bundle_without_sft_artifact(tmp_path):
    manifest_path, _manifest, base, _sft = canary_evaluation_case(tmp_path)
    output = tmp_path / "g0-bundle"
    published = run_g0_base_eval(
        manifest_path=manifest_path,
        output_dir=output,
        experiment_id="canary-g0",
        base=base,
        runtime=HeldOutEvalRuntime(generator=scripted_generator("all-correct")),
    )
    assert published == output
    assert (output / "g0-base" / "generations.jsonl").is_file()
    assert (output / "g0-base" / "metrics.json").is_file()
    assert (output / "g0-contract.json").is_file()
    metrics = (output / "g0-base" / "metrics.json").read_text(encoding="utf-8")
    assert '"n_scored"' in metrics
