"""Held-out paired evaluation command."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import typer
from transformers import AutoTokenizer

from ._cli_app import CLI_PATH_ERRORS, app, exit_on_error
from ._token_provenance import TokenizerBinding, attach_pinned_tokenizer_revision
from .eval.contract import (
    ArtifactIndexReference,
    DecodingContract,
    EvaluationArm,
    held_out_task_ids,
    logical_task_set_sha256,
)
from .eval.harness import G0BaseEvalSpec, HeldOutEvalRuntime, run_g0_base_eval, run_held_out_eval
from .eval.score import OBJECTIVE_SCORING_VERSION
from .eval.serializers import code_repair_prompt
from .logging import logger
from .path_safety import resolve_absent_output_directory, resolve_existing_path
from .split_contract import SerializerBinding, sha256_file
from .split_validation import validate_split_manifest_snapshot

TruncationPolicy = Literal["reject", "mark_unsupported", "left", "right"]
_TRUST_REMOTE_CODE_HELP = "Trust remote code from the model repo"


class _ArmCommon(TypedDict):
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_sha256: str
    serializer_id: str
    serializer_version: str
    serializer_sha256: str
    logical_task_set_sha256: str
    context_window: int
    truncation_policy: TruncationPolicy
    decoding: DecodingContract
    scoring_version: str


@dataclass(frozen=True)
class _ModelEvalOptions:
    """CLI options shared by ``held-out-eval`` and ``g0-held-out-eval``."""

    base_model_id: str
    base_revision: str
    tokenizer_id: str | None
    tokenizer_revision: str | None
    context_window: int
    max_new_tokens: int
    seed: int
    truncation_policy: TruncationPolicy
    device_map: str
    trust_remote_code: bool


@dataclass(frozen=True)
class _HeldOutEvalRequest:
    split_manifest: str
    sft_artifact: str
    output_dir: str
    options: _ModelEvalOptions


@dataclass(frozen=True)
class _G0HeldOutEvalRequest:
    split_manifest: str
    output_dir: str
    experiment_id: str
    options: _ModelEvalOptions


@dataclass(frozen=True)
class _HeldOutArmInputs:
    manifest_path: Path
    artifact_root: Path
    base_model_id: str
    base_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    context_window: int
    max_new_tokens: int
    seed: int
    truncation_policy: TruncationPolicy
    trust_remote_code: bool


def _infer_artifact_kind(root: Path) -> Literal["peft_adapter", "merged_model"]:
    if (root / "adapter_config.json").is_file():
        return "peft_adapter"
    if (root / "config.json").is_file():
        return "merged_model"
    raise ValueError(f"SFT artifact is neither a PEFT adapter nor a merged model: {root}")


def _load_pinned_tokenizer(
    tokenizer_id: str, tokenizer_revision: str, trust_remote_code: bool
) -> Any:
    # noinspection PyNoneFunctionAssignment
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 - revision is pinned below.
        tokenizer_id,
        trust_remote_code=trust_remote_code,
        revision=tokenizer_revision,
    )
    attach_pinned_tokenizer_revision(tokenizer, tokenizer_revision)
    return tokenizer


def _arm_common(inputs: _HeldOutArmInputs, tokenizer: Any, task_digest: str) -> _ArmCommon:
    tokenizer_binding = TokenizerBinding(implementation=tokenizer)
    serializer = SerializerBinding(implementation=code_repair_prompt)
    return {
        "tokenizer_repository": tokenizer_binding.tokenizer_id,
        "tokenizer_revision": tokenizer_binding.tokenizer_revision,
        "tokenizer_sha256": tokenizer_binding.tokenizer_sha256,
        "serializer_id": serializer.serializer_id,
        "serializer_version": serializer.serializer_version,
        "serializer_sha256": serializer.serializer_sha256,
        "logical_task_set_sha256": task_digest,
        "context_window": inputs.context_window,
        "truncation_policy": inputs.truncation_policy,
        "decoding": DecodingContract(
            do_sample=False,
            seed=inputs.seed,
            max_new_tokens=inputs.max_new_tokens,
            temperature=0,
            top_p=1,
        ),
        "scoring_version": OBJECTIVE_SCORING_VERSION,
    }


def _arm_inputs(
    options: _ModelEvalOptions, manifest_path: Path, artifact_root: Path
) -> _HeldOutArmInputs:
    return _HeldOutArmInputs(
        manifest_path=manifest_path,
        artifact_root=artifact_root,
        base_model_id=options.base_model_id,
        base_revision=options.base_revision,
        tokenizer_id=options.tokenizer_id or options.base_model_id,
        tokenizer_revision=options.tokenizer_revision or options.base_revision,
        context_window=options.context_window,
        max_new_tokens=options.max_new_tokens,
        seed=options.seed,
        truncation_policy=options.truncation_policy,
        trust_remote_code=options.trust_remote_code,
    )


def _eval_runtime(options: _ModelEvalOptions, tokenizer: Any) -> HeldOutEvalRuntime:
    return HeldOutEvalRuntime(
        tokenizer=tokenizer,
        trust_remote_code=options.trust_remote_code,
        device_map=options.device_map,
    )


def _base_arm(inputs: _HeldOutArmInputs) -> tuple[EvaluationArm, _ArmCommon, Any]:
    manifest = validate_split_manifest_snapshot(
        inputs.manifest_path, inputs.manifest_path.read_bytes()
    )
    tokenizer = _load_pinned_tokenizer(
        inputs.tokenizer_id, inputs.tokenizer_revision, inputs.trust_remote_code
    )
    common = _arm_common(inputs, tokenizer, logical_task_set_sha256(held_out_task_ids(manifest)))
    base = EvaluationArm(
        role="causal_base",
        model_repository=inputs.base_model_id,
        model_revision=inputs.base_revision,
        **common,
    )
    return base, common, tokenizer


def _evaluation_arms(inputs: _HeldOutArmInputs) -> tuple[EvaluationArm, EvaluationArm, Any]:
    base, common, tokenizer = _base_arm(inputs)
    index_path = inputs.artifact_root / "artifact_index.json"
    if not index_path.is_file():
        raise ValueError(f"SFT artifact requires artifact_index.json: {index_path}")
    artifact = ArtifactIndexReference(
        kind=_infer_artifact_kind(inputs.artifact_root),
        artifact_index_path=str(index_path),
        artifact_index_sha256=sha256_file(index_path),
    )
    sft = EvaluationArm(
        role="causal_sft",
        model_repository=inputs.base_model_id,
        model_revision=inputs.base_revision,
        artifact=artifact,
        **common,
    )
    return base, sft, tokenizer


def _g0_base_arm(inputs: _HeldOutArmInputs) -> tuple[EvaluationArm, Any]:
    base, _common, tokenizer = _base_arm(inputs)
    return base, tokenizer


def _run_held_out_eval_command(request: _HeldOutEvalRequest) -> None:
    try:
        manifest_path = resolve_existing_path(request.split_manifest, must_be_file=True)
        artifact_root = resolve_existing_path(request.sft_artifact, must_be_dir=True)
        destination = resolve_absent_output_directory(request.output_dir)
        base, sft, tokenizer = _evaluation_arms(
            _arm_inputs(request.options, manifest_path, artifact_root)
        )
        published = run_held_out_eval(
            manifest_path=manifest_path,
            output_dir=destination,
            arms=(base, sft),
            runtime=_eval_runtime(request.options, tokenizer),
        )
    except (*CLI_PATH_ERRORS, TypeError) as exc:
        exit_on_error(exc)
    logger.info("wrote held-out eval bundle %s", published)


def _run_g0_held_out_eval_command(request: _G0HeldOutEvalRequest) -> None:
    try:
        manifest_path = resolve_existing_path(request.split_manifest, must_be_file=True)
        destination = resolve_absent_output_directory(request.output_dir)
        base, tokenizer = _g0_base_arm(
            _arm_inputs(request.options, manifest_path, manifest_path.parent)
        )
        spec = G0BaseEvalSpec(
            manifest_path=manifest_path,
            output_dir=destination,
            experiment_id=request.experiment_id,
            base=base,
        )
        published = run_g0_base_eval(spec, runtime=_eval_runtime(request.options, tokenizer))
    except (*CLI_PATH_ERRORS, TypeError) as exc:
        exit_on_error(exc)
    logger.info("wrote G0 held-out eval bundle %s", published)


@app.command("g0-held-out-eval")
def g0_held_out_eval(
    split_manifest: str = typer.Option(..., help="Frozen split_manifest.json"),
    output_dir: str = typer.Option(..., help="New g0/<run_name> bundle directory"),
    experiment_id: str = typer.Option(..., help="Pre-registered experiment identifier"),
    base_model_id: str = typer.Option(..., help="Pinned base model id or local snapshot"),
    base_revision: str = typer.Option(..., help="Immutable 40-64 hex base revision"),
    tokenizer_id: str | None = typer.Option(
        None, help="Tokenizer id (defaults to --base-model-id)"
    ),
    tokenizer_revision: str | None = typer.Option(
        None, help="Tokenizer revision (defaults to --base-revision)"
    ),
    context_window: int = typer.Option(..., help="Prompt token budget for the base arm"),
    max_new_tokens: int = typer.Option(..., help="Decoding max_new_tokens for the base arm"),
    seed: int = typer.Option(17, help="Decoding seed"),
    truncation_policy: Annotated[
        TruncationPolicy, typer.Option(help="reject or mark_unsupported")
    ] = "mark_unsupported",
    device_map: str = typer.Option("auto", help="Transformers device_map"),
    trust_remote_code: bool = typer.Option(False, help=_TRUST_REMOTE_CODE_HELP),
):
    """Run base-only generation on frozen held-out membership before SFT exists."""
    options = _ModelEvalOptions(
        base_model_id,
        base_revision,
        tokenizer_id,
        tokenizer_revision,
        context_window,
        max_new_tokens,
        seed,
        truncation_policy,
        device_map,
        trust_remote_code,
    )
    _run_g0_held_out_eval_command(
        _G0HeldOutEvalRequest(split_manifest, output_dir, experiment_id, options)
    )


@app.command("held-out-eval")
def held_out_eval(
    split_manifest: str = typer.Option(..., help="Frozen split_manifest.json"),
    sft_artifact: str = typer.Option(..., help="PEFT adapter or merged-model bundle"),
    output_dir: str = typer.Option(..., help="New eval/<run_name> bundle directory"),
    base_model_id: str = typer.Option(..., help="Pinned base model id or local snapshot"),
    base_revision: str = typer.Option(..., help="Immutable 40-64 hex base revision"),
    tokenizer_id: str | None = typer.Option(
        None, help="Tokenizer id (defaults to --base-model-id)"
    ),
    tokenizer_revision: str | None = typer.Option(
        None, help="Tokenizer revision (defaults to --base-revision)"
    ),
    context_window: int = typer.Option(..., help="Prompt token budget for both arms"),
    max_new_tokens: int = typer.Option(128, help="Decoding max_new_tokens for both arms"),
    seed: int = typer.Option(17, help="Decoding seed for both arms"),
    truncation_policy: Annotated[
        TruncationPolicy, typer.Option(help="reject or mark_unsupported")
    ] = "mark_unsupported",
    device_map: str = typer.Option("auto", help="Transformers device_map"),
    trust_remote_code: bool = typer.Option(False, help=_TRUST_REMOTE_CODE_HELP),
):
    """Run base then SFT generation on frozen held-out membership and write the eval bundle."""
    options = _ModelEvalOptions(
        base_model_id,
        base_revision,
        tokenizer_id,
        tokenizer_revision,
        context_window,
        max_new_tokens,
        seed,
        truncation_policy,
        device_map,
        trust_remote_code,
    )
    _run_held_out_eval_command(
        _HeldOutEvalRequest(split_manifest, sft_artifact, output_dir, options)
    )
