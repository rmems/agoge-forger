"""Protocol tests for the frozen Granite 4.1 first-SFT comparison.

These tests do not load Granite weights. Fixture generations prove ordering
and artifact publication only.
"""

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agoge_forger.cli import app
from agoge_forger.eval.score import GenerationRecord
from agoge_forger.experiment.execute import ExperimentPhases, execute_experiment
from agoge_forger.experiment.freeze import freeze_experiment_contract
from agoge_forger.experiment.local import build_training_config, observe_completed_run
from agoge_forger.experiment.reload import assert_adapter_files, clean_reload
from agoge_forger.experiment.reload import main as reload_main
from agoge_forger.experiment.schema import (
    RegisteredBudget,
    ReloadObservation,
    SoftwareEnvironment,
    TrainingResults,
)
from agoge_forger.split_contract import canonical_json_bytes, iter_frozen_records, sha256_file
from tests.evaluation_contract_cases import frozen_manifest

COMMIT = "b" * 40
REVISION = "a" * 40
TOKENIZER_SHA = "ab" * 32


def _qualification(path: Path, *, verdict: str = "pass") -> None:
    payload = {
        "schema_version": "agoge.model-compatibility.v1",
        "model_repository": "ibm-granite/granite-4.1-3b-base",
        "model_revision": REVISION,
        "tokenizer_repository": "ibm-granite/granite-4.1-3b-base",
        "tokenizer_revision": REVISION,
        "tokenizer_sha256": TOKENIZER_SHA,
        "target_modules": ["q_proj", "v_proj"],
        "verdict": verdict,
        "lifecycle": "load-train-save-reload-generate",
        "trust_remote_code": False,
    }
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _freeze(tmp_path: Path, *, verdict: str = "pass"):
    manifest_path, _manifest = frozen_manifest(tmp_path)
    qualification = tmp_path / "qualification.json"
    _qualification(qualification, verdict=verdict)
    destination = tmp_path / "granite-4.1-first-sft" / "granite-4-1-first"
    contract = freeze_experiment_contract(
        experiment_id="granite-4-1-first",
        output_dir=destination,
        split_manifest=manifest_path,
        qualification_report=qualification,
        agoge_commit=COMMIT,
    )
    return contract, destination / "experiment-contract.json"


def _generations(contract, *, correct: set[int] | None) -> tuple[GenerationRecord, ...]:
    records = []
    held_out = iter_frozen_records(contract.split.split_manifest_path, "held_out")
    for index, row in enumerate(held_out):
        expected = str(row["text"])[int(row["completion_start_char"]) :]
        completion = expected if correct is None or index in correct else expected + " no"
        records.append(
            GenerationRecord(
                task_id=str(row["canonical_id"]),
                prompt="prompt",
                expected_completion=expected,
                completion=completion,
                status="ok",
            )
        )
    return tuple(records)


def _completed() -> TrainingResults:
    tokens = 128
    seconds = 4.0
    return TrainingResults(
        status="completed",
        accepted_records=8,
        accepted_tokens=tokens,
        train_loss=0.4,
        validation_loss=None,
        wall_clock_seconds=seconds,
        tokens_per_second=tokens / seconds,
        peak_vram_bytes=4096,
        adapter_bytes=16,
        adapter_sha256="c" * 64,
        checkpoint_bytes=0,
        environment=SoftwareEnvironment(
            python_version="3.12.0",
            torch_version="2.12.0",
            cuda_version="13.0",
            gpu_name="protocol-fixture",
            transformer_version="5.0.0",
        ),
    )


def _phases(
    calls: list[str],
    *,
    base_correct: set[int] | None = frozenset(),
    sft_correct: set[int] | None = None,
    training: TrainingResults | None = None,
    reload_ok: bool = True,
    fail_g0: bool = False,
) -> ExperimentPhases:
    def evaluate_base(contract):
        calls.append("g0")
        if fail_g0:
            raise RuntimeError("g0 device unavailable")
        return _generations(contract, correct=base_correct)

    def train(contract, adapter_dir: Path):
        del contract
        calls.append("train")
        assert (adapter_dir.parent.parent / "g0-base" / "metrics.json").is_file()
        return training if training is not None else _completed()

    def reload(contract, adapter_dir: Path):
        del contract, adapter_dir
        calls.append("reload")
        if reload_ok:
            return ReloadObservation(status="reloaded")
        return ReloadObservation(status="failed", failure_reason="clean reload failed")

    def evaluate_sft(contract, adapter_dir: Path):
        del adapter_dir
        calls.append("g1")
        return _generations(contract, correct=sft_correct)

    return ExperimentPhases(evaluate_base, train, reload, evaluate_sft)


def test_freeze_records_split_digest_and_refuses_a_second_write(tmp_path: Path):
    contract, contract_path = _freeze(tmp_path)
    assert contract.model_repository == "ibm-granite/granite-4.1-3b-base"
    assert contract.budget.target_modules == ("q_proj", "v_proj")
    assert contract.budget.completion_only_loss is True
    assert contract.decoding.scoring_version == "exact-match-v1"
    assert (contract_path.parent / "experiment-contract.sha256").is_file()
    with pytest.raises(FileExistsError):
        freeze_experiment_contract(
            experiment_id=contract.experiment_id,
            output_dir=contract_path.parent,
            split_manifest=contract.split.split_manifest_path,
            qualification_report=contract.blockers.qualification_report_path,
            agoge_commit=COMMIT,
        )


def test_freeze_blocks_failed_qualification(tmp_path: Path):
    with pytest.raises(ValueError, match="qualification verdict is pass"):
        _freeze(tmp_path, verdict="fail")


def test_registered_budget_rejects_a_retuned_learning_rate():
    with pytest.raises(ValueError, match="training budget drifted"):
        RegisteredBudget(target_modules=("q_proj",), learning_rate=0.001)


def test_execute_scores_g1_only_after_g0_train_and_reload(tmp_path: Path):
    _contract, contract_path = _freeze(tmp_path)
    calls: list[str] = []
    outcome = execute_experiment(
        contract_path,
        agoge_commit=COMMIT,
        worktree_clean=True,
        phases=_phases(calls),
    )
    assert outcome.status == "scored"
    assert calls == ["g0", "train", "reload", "g1"]
    experiment = contract_path.parent
    for name in (
        "g0-base/generations.jsonl",
        "g0-base/metrics.json",
        "g1-sft/generations.jsonl",
        "g1-sft/reload.json",
        "comparison.json",
        "regressions.jsonl",
        "training-results.json",
        "report.md",
        "reproduction.md",
    ):
        assert (experiment / name).is_file()
    comparison = json.loads((experiment / "comparison.json").read_text())
    report = (experiment / "report.md").read_text()
    assert comparison["status"] == "scored"
    assert comparison["conclusion"] == "improved"
    assert comparison["n_regressed"] == 0
    assert "Measured result: **improved**." in report
    assert "protocol-fixture" in report
    assert "Training loss is diagnostic only" in report
    assert (experiment / "regressions.jsonl").read_text() == ""
    assert "run-granite-first-sft" in (experiment / "reproduction.md").read_text()


def test_g0_failure_does_not_train(tmp_path: Path):
    _contract, contract_path = _freeze(tmp_path)
    calls: list[str] = []
    outcome = execute_experiment(
        contract_path,
        agoge_commit=COMMIT,
        worktree_clean=True,
        phases=_phases(calls, fail_g0=True),
    )
    assert outcome.status == "g0_failed"
    assert calls == ["g0"]
    experiment = contract_path.parent
    comparison = json.loads((experiment / "comparison.json").read_text())
    assert comparison["status"] == "not_run"
    assert comparison["conclusion"] is None
    report = (experiment / "report.md").read_text()
    assert "G0 failed. G1 training did not start." in report
    assert "g0 device unavailable" in report
    assert (experiment / "g0-base" / "failure.json").is_file()


def test_training_failure_does_not_score_g1(tmp_path: Path):
    _contract, contract_path = _freeze(tmp_path)
    calls: list[str] = []
    failed = TrainingResults(status="failed", failure_reason="CUDA is required but not available.")
    outcome = execute_experiment(
        contract_path,
        agoge_commit=COMMIT,
        worktree_clean=True,
        phases=_phases(calls, training=failed),
    )
    assert outcome.status == "training_failed"
    assert calls == ["g0", "train"]
    report = (contract_path.parent / "report.md").read_text()
    assert "G1 training did not complete" in report
    assert "Measured result: **improved**." not in report


def test_reload_failure_preserves_training_without_a_conclusion(tmp_path: Path):
    _contract, contract_path = _freeze(tmp_path)
    calls: list[str] = []
    outcome = execute_experiment(
        contract_path,
        agoge_commit=COMMIT,
        worktree_clean=True,
        phases=_phases(calls, reload_ok=False),
    )
    assert outcome.status == "reload_failed"
    assert calls == ["g0", "train", "reload"]
    training = json.loads((contract_path.parent / "training-results.json").read_text())
    assert training["status"] == "completed"
    assert "Clean reload failed" in (contract_path.parent / "report.md").read_text()


def test_mixed_outcomes_record_regressions(tmp_path: Path):
    _contract, contract_path = _freeze(tmp_path)
    outcome = execute_experiment(
        contract_path,
        agoge_commit=COMMIT,
        worktree_clean=True,
        phases=_phases([], base_correct={1}, sft_correct={0}),
    )
    assert outcome.status == "scored"
    comparison = json.loads((contract_path.parent / "comparison.json").read_text())
    assert comparison["conclusion"] == "mixed"
    assert comparison["n_improved"] >= 1
    assert comparison["n_regressed"] >= 1
    regressions = (contract_path.parent / "regressions.jsonl").read_text().strip().splitlines()
    assert len(regressions) == comparison["n_regressed"]


def test_execute_refuses_a_dirty_or_moved_commit(tmp_path: Path):
    _contract, contract_path = _freeze(tmp_path)
    with pytest.raises(ValueError, match="clean Agoge worktree"):
        execute_experiment(
            contract_path,
            agoge_commit=COMMIT,
            worktree_clean=False,
            phases=_phases([]),
        )
    with pytest.raises(ValueError, match="does not match frozen commit"):
        execute_experiment(
            contract_path,
            agoge_commit="d" * 40,
            worktree_clean=True,
            phases=_phases([]),
        )


def test_execute_refuses_a_tampered_contract_digest(tmp_path: Path):
    _contract, contract_path = _freeze(tmp_path)
    digest = contract_path.parent / "experiment-contract.sha256"
    digest.write_text(("0" * 64) + "  experiment-contract.json\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        execute_experiment(
            contract_path,
            agoge_commit=COMMIT,
            worktree_clean=True,
            phases=_phases([]),
        )


def test_training_config_uses_the_frozen_train_split(tmp_path: Path):
    contract, _contract_path = _freeze(tmp_path)
    adapter_dir = tmp_path / "granite-4.1-first-sft" / "granite-4-1-first" / "g1-sft" / "adapter"
    config = build_training_config(contract, adapter_dir)
    assert config.training.completion_only_loss is True
    assert config.training.learning_rate == 0.0001
    assert config.lora.target_modules == ["q_proj", "v_proj"]
    assert config.lora.target_modules_mode == "explicit"
    assert config.trust_remote_code is False
    assert sha256_file(Path(config.dataset_path)) == contract.split.train_sha256
    assert config.run_name == "adapter"


def test_observe_completed_run_reads_preprocessing_and_adapter(tmp_path: Path):
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"weights")
    (tmp_path / "completion_preprocessing.json").write_text(
        json.dumps({"rows": 3, "shifted_supervised_tokens": 30})
    )
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"loss": 1.5, "step": 1}, {"train_loss": 0.2}]})
    )
    observed = observe_completed_run(
        tmp_path,
        2.0,
        peak_vram_bytes=2048,
        gpu_name="protocol-fixture",
    )
    assert observed.status == "completed"
    assert observed.accepted_records == 3
    assert observed.accepted_tokens == 30
    assert observed.tokens_per_second == 15.0
    assert observed.train_loss == 0.2
    assert observed.adapter_sha256 == sha256_file(weights)
    assert observed.environment is not None
    assert observed.environment.gpu_name == "protocol-fixture"


def test_clean_reload_does_not_spawn_when_the_adapter_is_missing(tmp_path: Path):
    def runner(*_args, **_kwargs):
        raise AssertionError("clean reload spawned a process")

    observed = clean_reload(
        contract_path=tmp_path / "experiment-contract.json",
        adapter_dir=tmp_path / "missing",
        runner=runner,
    )
    assert observed.status == "failed"
    assert observed.failure_reason == "adapter directory is missing"


def test_clean_reload_reads_the_child_status(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"status": "reloaded"}\n',
            stderr="",
        )

    observed = clean_reload(
        contract_path=tmp_path / "experiment-contract.json",
        adapter_dir=adapter,
        runner=runner,
    )
    assert observed.status == "reloaded"


def test_reload_main_stops_before_weights_when_files_are_missing(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        assert_adapter_files(adapter)
    assert (
        reload_main(["--contract", str(tmp_path / "missing.json"), "--adapter", str(adapter)]) == 1
    )


def test_cli_freeze_and_run_use_the_git_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path, _manifest = frozen_manifest(tmp_path)
    qualification = tmp_path / "qualification.json"
    _qualification(qualification)
    destination = tmp_path / "granite-4.1-first-sft" / "granite-4-1-first"
    monkeypatch.setattr(
        "agoge_forger._cli_experiment.get_git_info",
        lambda: {"commit": COMMIT, "dirty": False},
    )
    runner = CliRunner()
    frozen = runner.invoke(
        app,
        [
            "freeze-granite-first-sft",
            "--experiment-id",
            "granite-4-1-first",
            "--output-dir",
            str(destination),
            "--split-manifest",
            str(manifest_path),
            "--qualification-report",
            str(qualification),
        ],
    )
    assert frozen.exit_code == 0, frozen.output
    contract_path = destination / "experiment-contract.json"

    def phases(_contract_path: Path):
        return _phases([])

    monkeypatch.setattr("agoge_forger._cli_experiment.local_phases", phases)
    ran = runner.invoke(app, ["run-granite-first-sft", "--contract", str(contract_path)])
    assert ran.exit_code == 0, ran.output
    assert "improved" in (destination / "report.md").read_text()


def test_cli_freeze_refuses_a_dirty_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path, _manifest = frozen_manifest(tmp_path)
    qualification = tmp_path / "qualification.json"
    _qualification(qualification)
    monkeypatch.setattr(
        "agoge_forger._cli_experiment.get_git_info",
        lambda: {"commit": COMMIT, "dirty": True},
    )
    result = CliRunner().invoke(
        app,
        [
            "freeze-granite-first-sft",
            "--experiment-id",
            "granite-4-1-first",
            "--output-dir",
            str(tmp_path / "granite-4.1-first-sft" / "granite-4-1-first"),
            "--split-manifest",
            str(manifest_path),
            "--qualification-report",
            str(qualification),
        ],
    )
    assert result.exit_code == 1
    assert "clean Agoge worktree" in result.output
