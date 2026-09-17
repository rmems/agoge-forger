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
from .experiment_contract import (
    ExperimentContract,
    build_experiment_contract,
    load_experiment_contract,
    split_pin_from_manifest,
    validate_experiment_contract,
)
from .harness import run_g0_base_eval, run_held_out_eval
from .score import OBJECTIVE_SCORING_VERSION

__all__ = [
    "OBJECTIVE_SCORING_VERSION",
    "ArtifactProducerProvenance",
    "DecodingContract",
    "EvaluationArm",
    "ExperimentContract",
    "PairedEvaluationContract",
    "VerifiedAdapterSource",
    "build_evaluation_contract",
    "build_experiment_contract",
    "compose_evaluation_contract",
    "load_experiment_contract",
    "require_adapter_source_tensor_schema",
    "run_g0_base_eval",
    "run_held_out_eval",
    "split_pin_from_manifest",
    "validate_evaluation_contract",
    "validate_experiment_contract",
    "verified_adapter_source",
]
