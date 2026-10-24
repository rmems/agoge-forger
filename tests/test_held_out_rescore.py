"""Post-hoc re-scoring and scorer selection for #183."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agoge_forger import _cli_eval as cli_eval
from agoge_forger.cli import app
from agoge_forger.eval.compare import compare_arms
from agoge_forger.eval.harness import HeldOutEvalRuntime, run_held_out_eval
from agoge_forger.eval.rescore import rescore_paired_bundle
from agoge_forger.eval.score import (
    DOCTEST_V1_SCORING_VERSION,
    EXACT_MATCH_V2_SCORING_VERSION,
    OBJECTIVE_SCORING_VERSION,
    SUPPORTED_SCORING_VERSIONS,
    ExampleScore,
    GenerationRecord,
)
from tests.held_out_eval_cases import canary_evaluation_case, scripted_generator

MODULE = 'def inc(x):\n    """\n    >>> inc(1)\n    2\n    """\n    return x + 1\n'
RM760 = (
    Path(__file__).resolve().parents[1]
    / "reports/granite-4.1-first-sft/rm760-code-repair-v1-rm760-20260914"
)


def _generation(task_id, completion) -> dict:
    return GenerationRecord(
        task_id=task_id,
        prompt="Fix:\n",
        expected_completion=MODULE,
        completion=completion,
        status="ok",
    ).model_dump(mode="json")


def _write_bundle(root: Path, base: list[str], sft: list[str]) -> Path:
    root.mkdir(parents=True)
    task_ids = [f"t{i}" for i in range(len(base))]
    (root / "contract.json").write_text(
        json.dumps({"logical_task_ids": task_ids}), encoding="utf-8"
    )
    for arm, completions in (("base", base), ("sft", sft)):
        (root / arm).mkdir()
        rows = [_generation(f"t{i}", text) for i, text in enumerate(completions)]
        (root / arm / "generations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return root


@pytest.fixture
def fenced_base_bundle(tmp_path) -> Path:
    fenced = f"```python\n{MODULE}```"
    return _write_bundle(
        tmp_path / "bundle", [fenced, fenced], [MODULE, "def inc(x):\n    return x"]
    )


def test_registry_keeps_v1_as_the_default():
    assert OBJECTIVE_SCORING_VERSION == "exact-match-v1"
    assert SUPPORTED_SCORING_VERSIONS == (
        "exact-match-v1",
        EXACT_MATCH_V2_SCORING_VERSION,
        DOCTEST_V1_SCORING_VERSION,
    )


def test_v1_rescore_shows_the_fence_confound_and_v2_removes_it(fenced_base_bundle):
    v1 = json.loads(
        rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v1").read_bytes()
    )
    v2 = json.loads(
        rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v2").read_bytes()
    )
    assert v1["arms"]["base"]["n_correct"] == 0
    assert v1["conclusion"] == "improved"
    assert v2["arms"]["base"]["n_correct"] == 2
    assert v2["arms"]["sft"]["n_correct"] == 1
    assert v2["conclusion"] == "regressed"
    assert v2["scoring_version"] == "exact-match-v2"


def test_doctest_rescore_records_per_task_details(fenced_base_bundle):
    path = rescore_paired_bundle(fenced_base_bundle, scoring_version="doctest-v1")
    payload = json.loads(path.read_bytes())
    assert path == fenced_base_bundle / "comparison.doctest-v1.json"
    base_row, _ = payload["scores"]["base"]
    assert base_row == {"task_id": "t0", "verdict": "correct", "n_examples": 1, "n_failed": 0}
    sft_failed = payload["scores"]["sft"][1]
    assert sft_failed["verdict"] == "incorrect"
    assert sft_failed["reason"] == "candidate has no doctest examples"
    assert payload["post_hoc"].startswith("Post-hoc re-score")


def test_rescore_leaves_the_bundle_untouched_and_never_overwrites(fenced_base_bundle):
    def digest():
        return {
            str(p.relative_to(fenced_base_bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(fenced_base_bundle.rglob("*.jsonl"))
        }

    before = digest()
    out = fenced_base_bundle / "elsewhere" / "result.json"
    rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v2", output=out)
    assert digest() == before
    assert out.is_file()
    payload = json.loads(out.read_bytes())
    assert set(payload["rescored_from"]) == {"base/generations.jsonl", "sft/generations.jsonl"}
    with pytest.raises(FileExistsError):
        rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v2", output=out)


def test_rescore_rejects_bad_inputs(tmp_path, fenced_base_bundle):
    with pytest.raises(ValueError, match="unsupported objective scoring version"):
        rescore_paired_bundle(fenced_base_bundle, scoring_version="heuristic-judge-v1")
    with pytest.raises(FileNotFoundError, match="generations not found"):
        rescore_paired_bundle(tmp_path / "missing", scoring_version="exact-match-v2")
    (fenced_base_bundle / "sft" / "generations.jsonl").write_text('{"task_id": 1}\n')
    with pytest.raises(ValueError, match="not a generation record"):
        rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v2")


def test_rescore_cli(fenced_base_bundle, tmp_path):
    out = tmp_path / "cli-result.json"
    runner = CliRunner()
    ok = runner.invoke(
        app,
        [
            "rescore-held-out",
            str(fenced_base_bundle),
            "--scoring-version",
            "exact-match-v2",
            "--output",
            str(out),
        ],
    )
    assert ok.exit_code == 0, ok.output
    assert json.loads(out.read_bytes())["scoring_version"] == "exact-match-v2"
    bad = runner.invoke(
        app, ["rescore-held-out", str(fenced_base_bundle), "--scoring-version", "nope-v9"]
    )
    assert bad.exit_code == 1


@pytest.mark.usefixtures("cached_test_base_config")
@pytest.mark.parametrize("version", [EXACT_MATCH_V2_SCORING_VERSION, DOCTEST_V1_SCORING_VERSION])
def test_contract_can_select_a_new_scorer(tmp_path, version):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    arms = tuple(arm.model_copy(update={"scoring_version": version}) for arm in (base, sft))
    output = tmp_path / "eval" / version
    run_held_out_eval(
        manifest_path=manifest_path,
        output_dir=output,
        arms=arms,
        runtime=HeldOutEvalRuntime(generator=scripted_generator("all-correct")),
    )
    assert json.loads((output / "comparison.json").read_bytes())["scoring_version"] == version
    assert f"Scoring version: `{version}`" in (output / "report.md").read_text()


@pytest.mark.usefixtures("cached_test_base_config")
def test_eval_cli_forwards_scoring_version(tmp_path, monkeypatch):
    manifest_path, _manifest, base, sft = canary_evaluation_case(tmp_path)
    seen: dict[str, object] = {}

    def fake_arms(inputs):
        seen["inputs"] = inputs
        return base, sft, object()

    monkeypatch.setattr("agoge_forger._cli_eval._evaluation_arms", fake_arms)
    monkeypatch.setattr(
        "agoge_forger._cli_eval.run_held_out_eval", lambda **kwargs: Path(kwargs["output_dir"])
    )
    args = [
        "held-out-eval",
        "--split-manifest",
        str(manifest_path),
        "--sft-artifact",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--base-model-id",
        "example/base-model",
        "--base-revision",
        "abcdef0123456789abcdef0123456789abcdef01",
        "--context-window",
        "4096",
    ]
    assert CliRunner().invoke(app, args).exit_code == 0
    assert seen["inputs"].scoring_version == OBJECTIVE_SCORING_VERSION  # type: ignore[attr-defined]
    assert CliRunner().invoke(app, [*args, "--scoring-version", "doctest-v1"]).exit_code == 0
    assert seen["inputs"].scoring_version == "doctest-v1"  # type: ignore[attr-defined]


def _verdicts(payload) -> dict:
    return {
        arm: [
            (row["task_id"], row["verdict"], row.get("n_examples"), row.get("n_failed"))
            for row in rows
        ]
        for arm, rows in payload["scores"].items()
    }


@pytest.mark.skipif(not RM760.is_dir(), reason="RM-760 report bundle not present")
def test_rm760_v1_rescore_reproduces_the_committed_comparison(tmp_path):
    out = rescore_paired_bundle(
        RM760 / "g1-eval", scoring_version="exact-match-v1", output=tmp_path / "v1.json"
    )
    committed = json.loads((RM760 / "comparison.json").read_bytes())
    rescored = json.loads(out.read_bytes())
    assert {key: rescored[key] for key in committed} == committed


@pytest.mark.skipif(not RM760.is_dir(), reason="RM-760 report bundle not present")
@pytest.mark.parametrize("version", ["exact-match-v2", "doctest-v1"])
def test_rm760_committed_secondary_scoring_is_reproducible(tmp_path, version):
    out = rescore_paired_bundle(
        RM760 / "g1-eval", scoring_version=version, output=tmp_path / f"{version}.json"
    )
    committed = json.loads((RM760 / f"comparison.{version}.json").read_bytes())
    rescored = json.loads(out.read_bytes())
    for key in ("arms", "outcomes", "conclusion", "delta_accuracy", "rescored_from"):
        assert rescored[key] == committed[key], key
    assert _verdicts(rescored) == _verdicts(committed)


def test_compare_arms_rejects_scores_from_another_scorer():
    score = ExampleScore(task_id="t0", scoring_version="exact-match-v1", verdict="correct")
    with pytest.raises(ValueError, match="requested exact-match-v2 but scores came from"):
        compare_arms((score,), (score,), scoring_version="exact-match-v2")
    mixed = score.model_copy(update={"scoring_version": "exact-match-v2"})
    with pytest.raises(ValueError, match="scoring versions drifted|requested"):
        compare_arms((score,), (mixed,), scoring_version="exact-match-v2")
    assert compare_arms((score,), (score,)).scoring_version == "exact-match-v1"


def test_rescore_refuses_a_bundle_that_omits_contract_tasks(fenced_base_bundle):
    for arm in ("base", "sft"):
        path = fenced_base_bundle / arm / "generations.jsonl"
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover exactly the 2 task"):
        rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v2")
    assert not (fenced_base_bundle / "comparison.exact-match-v2.json").exists()


def test_rescore_refuses_a_bundle_without_its_contract(fenced_base_bundle):
    (fenced_base_bundle / "contract.json").unlink()
    with pytest.raises(FileNotFoundError, match="bundle contract not found"):
        rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v2")


def test_rescored_from_hashes_the_bytes_that_were_scored(fenced_base_bundle, monkeypatch):
    path = fenced_base_bundle / "base" / "generations.jsonl"
    scored_bytes = path.read_bytes()
    reads = []
    original = Path.read_bytes

    def read_then_change(self):
        content = original(self)
        if self == path:
            reads.append(content)
            self.write_bytes(b"")  # the file changes right after it is read
        return content

    monkeypatch.setattr(Path, "read_bytes", read_then_change)
    out = rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v2")
    payload = json.loads(original(out))
    assert len(reads) == 1
    assert (
        payload["rescored_from"]["base/generations.jsonl"]
        == hashlib.sha256(scored_bytes).hexdigest()
    )


def test_a_failed_write_leaves_nothing_behind_and_can_be_retried(fenced_base_bundle, monkeypatch):
    real_fsync = os.fsync

    def failing_fsync(_fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="disk full"):
        rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v2")
    assert not list(fenced_base_bundle.glob("comparison.*")) and not list(
        fenced_base_bundle.glob(".comparison*")
    )
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert rescore_paired_bundle(fenced_base_bundle, scoring_version="exact-match-v2").is_file()


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("held-out-eval", ["--sft-artifact", "a"]),
        ("g0-held-out-eval", ["--experiment-id", "e", "--max-new-tokens", "8"]),
    ],
)
def test_an_unknown_scorer_is_rejected_before_any_tokenizer_work(monkeypatch, command, extra):
    def must_not_load(*_args, **_kwargs):
        raise AssertionError("tokenizer loaded for an invalid scorer")

    monkeypatch.setattr(cli_eval, "_load_pinned_tokenizer", must_not_load)
    result = CliRunner().invoke(
        app,
        [
            command,
            "--split-manifest", "m.json",
            "--output-dir", "out",
            "--base-model-id", "b",
            "--base-revision", "r",
            "--context-window", "8",
            "--scoring-version", "nope-v9",
            *extra,
        ],
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    assert "unsupported objective scoring version" in result.output
