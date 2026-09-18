"""Public facade for deterministic token-budget mixture composition."""

from pathlib import Path

from ._strict_json import decode_json_object
from .mixture_allocate import allocate_token_quotas
from .mixture_compose import compose_mixture
from .mixture_ledger import load_token_ledger, write_token_ledger
from .mixture_ledger_schema import TokenLedger
from .mixture_result_schema import MixtureManifest
from .mixture_schema import (
    MIXTURE_ALGORITHM_VERSION,
    MIXTURE_MANIFEST_VERSION,
    TOKEN_LEDGER_VERSION,
    MixtureCompositionSpec,
    MixturePolicy,
    MixtureSourceSpec,
)

__all__ = [
    "MIXTURE_ALGORITHM_VERSION",
    "MIXTURE_MANIFEST_VERSION",
    "TOKEN_LEDGER_VERSION",
    "MixtureCompositionSpec",
    "MixtureManifest",
    "MixturePolicy",
    "MixtureSourceSpec",
    "TokenLedger",
    "allocate_token_quotas",
    "compose_mixture",
    "load_mixture_spec",
    "load_token_ledger",
    "write_token_ledger",
]


def load_mixture_spec(path: str | Path) -> MixtureCompositionSpec:
    spec_path = Path(path).expanduser().resolve(strict=True)
    value = decode_json_object(spec_path.read_bytes(), str(spec_path), object_label="mixture spec")
    return MixtureCompositionSpec.model_validate(value)
