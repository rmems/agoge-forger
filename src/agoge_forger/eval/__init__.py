"""Public evaluation-contract, artifact-validation, and held-out harness interfaces."""

from ._artifact_schema import (
    ArtifactIndex,
    ArtifactIndexEntry,
    ArtifactIndexReference,
    ArtifactProducerProvenance,
    FrozenEvaluationModel,
    portable_artifact_path,
    portable_contract_reference,
)
from ._artifact_validation import (
    VerifiedAdapterSource,
    require_adapter_source_tensor_schema,
    verified_adapter_source,
)
from ._descriptor_bundle import (
    EntryIdentity,
    hash_relative_file,
    open_bundle,
    read_relative_file,
    require_descriptor_support,
    scan_bundle,
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
    "ArtifactIndex",
    "ArtifactIndexEntry",
    "ArtifactIndexReference",
    "ArtifactProducerProvenance",
    "DecodingContract",
    "EntryIdentity",
    "EvaluationArm",
    "FrozenEvaluationModel",
    "PairedEvaluationContract",
    "VerifiedAdapterSource",
    "build_evaluation_contract",
    "compose_evaluation_contract",
    "hash_relative_file",
    "open_bundle",
    "portable_artifact_path",
    "portable_contract_reference",
    "read_relative_file",
    "require_adapter_source_tensor_schema",
    "require_descriptor_support",
    "run_held_out_eval",
    "scan_bundle",
    "validate_evaluation_contract",
    "verified_adapter_source",
]
