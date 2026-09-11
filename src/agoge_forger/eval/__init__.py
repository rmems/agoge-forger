"""Public evaluation-contract, artifact-validation, and held-out harness interfaces."""

from ._artifact_schema import ArtifactProducerProvenance
from ._artifact_validation import (
    VerifiedAdapterSource,
    require_adapter_source_tensor_schema,
    verified_adapter_source,
)
from .contract import (
    DecodingContract,
    EvaluationArm,
    PairedEvaluationContract,
    build_evaluation_contract,
    compose_evaluation_contract,
    validate_evaluation_contract,
)
from .harness import run_held_out_eval
from .score import OBJECTIVE_SCORING_VERSION

__all__ = [
    "OBJECTIVE_SCORING_VERSION",
    "ArtifactProducerProvenance",
    "DecodingContract",
    "EvaluationArm",
    "PairedEvaluationContract",
    "VerifiedAdapterSource",
    "build_evaluation_contract",
    "compose_evaluation_contract",
    "require_adapter_source_tensor_schema",
    "run_held_out_eval",
    "validate_evaluation_contract",
    "verified_adapter_source",
]
