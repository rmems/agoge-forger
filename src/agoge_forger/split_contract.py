"""Public facade for immutable source-level split contracts.

``SplitManifest`` remains the single authoritative partition schema. The
implementation is separated by responsibility so materialization, validation,
and derivative token statistics can evolve without duplicating that schema.
"""

from .split_materialize import materialize_split
from .split_schema import (
    SPLIT_ALGORITHM_VERSION,
    SPLIT_MANIFEST_VERSION,
    SPLIT_NAMES,
    TOKEN_STATS_VERSION,
    CanonicalIdentityPolicy,
    LeakageAudit,
    SourceFile,
    SplitArtifact,
    SplitManifest,
    SplitMaterializationSpec,
    SplitMember,
    SplitName,
    SplitPolicy,
    TokenStatistics,
    TokenStatisticsSpec,
    TokenStatSplit,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .split_token_stats import TokenStatisticsDerivation, write_token_statistics
from .split_validation import (
    iter_frozen_records,
    load_frozen_dataset,
    load_split_manifest,
    validate_split_manifest,
)

__all__ = [
    "SPLIT_ALGORITHM_VERSION",
    "SPLIT_MANIFEST_VERSION",
    "SPLIT_NAMES",
    "TOKEN_STATS_VERSION",
    "CanonicalIdentityPolicy",
    "LeakageAudit",
    "SourceFile",
    "SplitArtifact",
    "SplitManifest",
    "SplitMaterializationSpec",
    "SplitMember",
    "SplitName",
    "SplitPolicy",
    "TokenStatSplit",
    "TokenStatistics",
    "TokenStatisticsDerivation",
    "TokenStatisticsSpec",
    "canonical_json_bytes",
    "iter_frozen_records",
    "load_frozen_dataset",
    "load_split_manifest",
    "materialize_split",
    "sha256_bytes",
    "sha256_file",
    "validate_split_manifest",
    "write_token_statistics",
]
