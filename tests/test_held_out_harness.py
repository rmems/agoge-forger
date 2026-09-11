import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agoge_forger.eval.bundle import BUNDLE_FILES
from agoge_forger.eval.contract import compose_evaluation_contract
from agoge_forger.eval.generate import PreparedTask, apply_context_window, generation_kwargs
from agoge_forger.eval.harness import HeldOutEvalRuntime, run_held_out_eval
from agoge_forger.eval.score import OBJECTIVE_SCORING_VERSION
from tests.held_out_eval_cases import canary_evaluation_case, scripted_generator

pytestmark = pytest.mark.usefixtures("cached_test_base_config")


def _run(manifest_path, output, base, sft, **runtime):
    return run_held_out_eval(
        manifest_path=manifest_path,
        output_dir=output,
        arms=(base, sft),
        runtime=HeldOutEvalRuntime(**runtime),
    )


def _bundle_paths(output: Path) -> list[Path]:
    return [output / relative for relative in BUNDLE_FILES]


def test_held_out_harness_writes_auditable_bundle(tmp_path):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    output = tmp_path / "eval" / "canary"
    published = _run(manifest_path, output, base, sft, generator=scripted_generator())
    assert published == output
    for path in _bundle_paths(output):
        assert path.is_file(), path
    comparison = json.loads((output / "comparison.json").read_bytes())
    assert set(comparison["outcomes"].values()) <= {"improved", "regressed", "tied", "invalid"}
    assert comparison["n_improved"] >= 1
    assert comparison["n_regressed"] >= 1
    assert comparison["n_tied"] >= 1
    assert comparison["n_invalid"] >= 1
    assert comparison["conclusion"] == "mixed"
    report = (output / "report.md").read_text()
    assert "**Conclusion:** mixed" in report
    assert "Heuristic or model-judge signals were not used" in report
    contract = json.loads((output / "contract.json").read_bytes())
    assert contract["schema_version"] == "agoge.evaluation-contract.v2"
    assert contract["base"]["scoring_version"] == OBJECTIVE_SCORING_VERSION


def test_held_out_harness_refuses_overwrite(tmp_path):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    output = tmp_path / "eval" / "canary"
    _run(manifest_path, output, base, sft, generator=scripted_generator("all-correct"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(manifest_path, output, base, sft, generator=scripted_generator("all-correct"))


def test_held_out_harness_rejects_scoring_version_drift(tmp_path):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    drifted = {
        "scoring_version": "heuristic-judge-v1",
    }
    with pytest.raises(ValueError, match="scoring_version must be exact-match-v1"):
        _run(
            manifest_path,
            tmp_path / "eval" / "canary",
            base.model_copy(update=drifted),
            sft.model_copy(update=drifted),
            generator=scripted_generator("all-correct"),
        )


def test_compose_still_fails_closed_on_decoding_drift(tmp_path):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    drifted_decoding = base.decoding.model_copy(update={"seed": 99})
    with pytest.raises(ValidationError, match="non-comparable"):
        compose_evaluation_contract(
            manifest_path=manifest_path,
            contract_path=tmp_path / "eval" / "contract.json",
            base=base,
            sft=sft.model_copy(update={"decoding": drifted_decoding}),
        )


def test_scripted_all_correct_is_null_delta(tmp_path):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    output = tmp_path / "eval" / "null"
    _run(manifest_path, output, base, sft, generator=scripted_generator("all-correct"))
    comparison = json.loads((output / "comparison.json").read_bytes())
    assert comparison["conclusion"] == "null"
    assert comparison["n_improved"] == 0
    assert comparison["n_regressed"] == 0
    assert (output / "regressions.jsonl").read_bytes() == b""


class _CountingTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) or 1))}


def test_mark_unsupported_does_not_truncate(tmp_path):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    narrow = base.model_copy(update={"context_window": 1, "truncation_policy": "mark_unsupported"})
    sft_narrow = sft.model_copy(
        update={"context_window": 1, "truncation_policy": "mark_unsupported"}
    )
    output = tmp_path / "eval" / "unsupported"
    _run(
        manifest_path,
        output,
        narrow,
        sft_narrow,
        generator=scripted_generator("all-correct"),
        tokenizer=_CountingTokenizer(),
    )
    comparison = json.loads((output / "comparison.json").read_bytes())
    assert comparison["conclusion"] == "inconclusive"
    assert comparison["n_invalid"] == comparison["n_tasks"]


def test_apply_context_window_rejects_overflow():
    task = PreparedTask("t", "one two three", "x", "ready")
    with pytest.raises(ValueError, match="exceeds context_window"):
        apply_context_window(
            task, _CountingTokenizer(), context_window=1, truncation_policy="reject"
        )


def test_generation_kwargs_omit_temperature_when_greedy():
    from agoge_forger.eval.contract import DecodingContract

    decoding = DecodingContract(do_sample=False, seed=1, max_new_tokens=8, temperature=0, top_p=1)

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

    kwargs = generation_kwargs(decoding, _Tok())
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
