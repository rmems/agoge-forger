"""Held-out paired evaluation command."""

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
from .eval.harness import run_held_out_eval
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


def _infer_artifact_kind(root: Path) -> Literal["peft_adapter", "merged_model"]:
    if (root / "adapter_config.json").is_file():
        return "peft_adapter"
    if (root / "config.json").is_file():
        return "merged_model"
    raise ValueError(f"SFT artifact is neither a PEFT adapter nor a merged model: {root}")


def _evaluation_arms(
    *,
    manifest_path: Path,
    artifact_root: Path,
    base_model_id: str,
    base_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    context_window: int,
    max_new_tokens: int,
    seed: int,
    truncation_policy: TruncationPolicy,
    trust_remote_code: bool,
) -> tuple[EvaluationArm, EvaluationArm, Any]:
    manifest = validate_split_manifest_snapshot(manifest_path, manifest_path.read_bytes())
    task_digest = logical_task_set_sha256(held_out_task_ids(manifest))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=trust_remote_code,
        revision=tokenizer_revision,
    )
    attach_pinned_tokenizer_revision(tokenizer, tokenizer_revision)
    tokenizer_binding = TokenizerBinding(implementation=tokenizer)
    serializer = SerializerBinding(implementation=code_repair_prompt)
    decoding = DecodingContract(
        do_sample=False,
        seed=seed,
        max_new_tokens=max_new_tokens,
        temperature=0,
        top_p=1,
    )
    index_path = artifact_root / "artifact_index.json"
    if not index_path.is_file():
        raise ValueError(f"SFT artifact requires artifact_index.json: {index_path}")
    common: _ArmCommon = {
        "tokenizer_repository": tokenizer_binding.tokenizer_id,
        "tokenizer_revision": tokenizer_binding.tokenizer_revision,
        "tokenizer_sha256": tokenizer_binding.tokenizer_sha256,
        "serializer_id": serializer.serializer_id,
        "serializer_version": serializer.serializer_version,
        "serializer_sha256": serializer.serializer_sha256,
        "logical_task_set_sha256": task_digest,
        "context_window": context_window,
        "truncation_policy": truncation_policy,
        "decoding": decoding,
        "scoring_version": OBJECTIVE_SCORING_VERSION,
    }
    artifact = ArtifactIndexReference(
        kind=_infer_artifact_kind(artifact_root),
        artifact_index_path=str(index_path),
        artifact_index_sha256=sha256_file(index_path),
    )
    base = EvaluationArm(
        role="causal_base",
        model_repository=base_model_id,
        model_revision=base_revision,
        **common,
    )
    sft = EvaluationArm(
        role="causal_sft",
        model_repository=base_model_id,
        model_revision=base_revision,
        artifact=artifact,
        **common,
    )
    return base, sft, tokenizer


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
    try:
        manifest_path = resolve_existing_path(split_manifest, must_be_file=True)
        artifact_root = resolve_existing_path(sft_artifact, must_be_dir=True)
        destination = resolve_absent_output_directory(output_dir)
        base, sft, tokenizer = _evaluation_arms(
            manifest_path=manifest_path,
            artifact_root=artifact_root,
            base_model_id=base_model_id,
            base_revision=base_revision,
            tokenizer_id=tokenizer_id or base_model_id,
            tokenizer_revision=tokenizer_revision or base_revision,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            seed=seed,
            truncation_policy=truncation_policy,
            trust_remote_code=trust_remote_code,
        )
        published = run_held_out_eval(
            manifest_path=manifest_path,
            output_dir=destination,
            base=base,
            sft=sft,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
    except (*CLI_PATH_ERRORS, TypeError) as exc:
        exit_on_error(exc)
    logger.info("wrote held-out eval bundle %s", published)
