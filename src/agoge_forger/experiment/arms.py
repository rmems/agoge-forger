"""Build held-out evaluation arms from the frozen Granite contract."""

from __future__ import annotations

from typing import Literal

from ..eval.contract import ArtifactIndexReference, DecodingContract, EvaluationArm
from .schema import GraniteFirstSftContract

ArmRole = Literal["causal_base", "causal_sft"]


def base_arm(contract: GraniteFirstSftContract) -> EvaluationArm:
    return evaluation_arm(contract, role="causal_base", artifact=None)


def sft_arm(contract: GraniteFirstSftContract, artifact: ArtifactIndexReference) -> EvaluationArm:
    return evaluation_arm(contract, role="causal_sft", artifact=artifact)


def evaluation_arm(
    contract: GraniteFirstSftContract,
    *,
    role: ArmRole,
    artifact: ArtifactIndexReference | None,
) -> EvaluationArm:
    decoding = contract.decoding
    return EvaluationArm(
        role=role,
        model_repository=contract.model_repository,
        model_revision=contract.model_revision,
        artifact=artifact,
        tokenizer_repository=contract.tokenizer_repository,
        tokenizer_revision=contract.tokenizer_revision,
        tokenizer_sha256=contract.tokenizer_sha256,
        serializer_id=contract.serializer_id,
        serializer_version=contract.serializer_version,
        serializer_sha256=contract.serializer_sha256,
        logical_task_set_sha256=contract.split.logical_task_set_sha256,
        context_window=decoding.context_window,
        truncation_policy=decoding.truncation_policy,
        decoding=DecodingContract(
            do_sample=False,
            seed=decoding.seed,
            max_new_tokens=decoding.max_new_tokens,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
        ),
        scoring_version=decoding.scoring_version,
    )
